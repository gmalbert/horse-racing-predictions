#!/usr/bin/env python
"""
Fetch odds from Betfair Exchange API.

Reads racecards JSON and fetches exchange odds for each race, outputting CSV
compatible with merge_odds_from_csv.py.

Betfair API docs: https://docs.developer.betfair.com/

Usage:
  python scripts/fetch_odds_betfair.py --date 2025-12-22
  python scripts/fetch_odds_betfair.py --date 2025-12-22 --output data/raw/betfair_odds.csv
"""

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from betfairlightweight import APIClient
from betfairlightweight.filters import (
    market_filter,
    price_projection,
)
from dotenv import load_dotenv


def normalize_name(name: str) -> str:
    """Normalize horse/venue name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)     # Collapse whitespace
    return name


def create_betfair_client() -> APIClient:
    """Create and login to Betfair API client."""
    load_dotenv()
    
    username = os.getenv('BETFAIR_USERNAME')
    password = os.getenv('BETFAIR_PASSWORD')
    app_key = os.getenv('BETFAIR_APP_KEY')
    
    if not all([username, password, app_key]):
        raise ValueError(
            "Missing Betfair credentials. Add to .env file:\n"
            "  BETFAIR_USERNAME=your_username\n"
            "  BETFAIR_PASSWORD=your_password\n"
            "  BETFAIR_APP_KEY=your_app_key"
        )
    
    # Create client
    client = APIClient(
        username=username,
        password=password,
        app_key=app_key,
    )
    
    # Login
    client.login()
    print(f"✓ Logged in to Betfair as {username}")
    
    return client


def find_betfair_markets(client: APIClient, venue: str, race_date: str) -> List:
    """
    Find Betfair markets for a specific venue and date.
    
    Args:
        client: Betfair API client
        venue: Course name (e.g., "Kempton")
        race_date: Date in YYYY-MM-DD format
    
    Returns:
        List of market catalogue objects
    """
    # Parse date
    race_dt = datetime.strptime(race_date, '%Y-%m-%d')
    
    # Create time filter (whole day)
    time_from = race_dt.replace(hour=0, minute=0, second=0)
    time_to = race_dt.replace(hour=23, minute=59, second=59)
    
    # Create market filter
    mkt_filter = market_filter(
        event_type_ids=['7'],  # 7 = Horse Racing
        market_countries=['GB', 'IE'],
        market_type_codes=['WIN'],  # WIN market (to-win odds)
        market_start_time={
            'from': time_from.isoformat() + 'Z',
            'to': time_to.isoformat() + 'Z',
        },
        venues=[venue],
    )
    
    # Create price projection
    price_proj = price_projection(
        price_data=['EX_BEST_OFFERS'],  # Exchange best back/lay prices
    )
    
    # List markets
    try:
        markets = client.betting.list_market_catalogue(
            filter=mkt_filter,
            market_projection=['RUNNER_DESCRIPTION', 'EVENT', 'MARKET_START_TIME'],
            max_results=100,
        )
        return markets
    except Exception as e:
        print(f"Warning: Failed to fetch markets for {venue}: {e}")
        return []


def get_market_odds(client: APIClient, market_id: str) -> Dict[str, float]:
    """
    Get current odds for all runners in a market.
    
    Returns:
        Dict mapping selection_id -> best back price (decimal odds)
    """
    try:
        # Get market book with prices
        market_books = client.betting.list_market_book(
            market_ids=[market_id],
            price_projection=price_projection(
                price_data=['EX_BEST_OFFERS'],
            ),
        )
        
        if not market_books:
            return {}
        
        market_book = market_books[0]
        odds_map = {}
        
        for runner in market_book.runners:
            selection_id = runner.selection_id
            
            # Get best available back price (what you can back at)
            if runner.ex and runner.ex.available_to_back:
                best_back = runner.ex.available_to_back[0]
                odds_map[str(selection_id)] = best_back.price
        
        return odds_map
        
    except Exception as e:
        print(f"Warning: Failed to get odds for market {market_id}: {e}")
        return {}


def match_betfair_runner_to_horse(runner_name: str, horses: List[Dict]) -> Optional[Dict]:
    """
    Match Betfair runner name to horse in racecards.
    
    Args:
        runner_name: Runner name from Betfair
        horses: List of horses from racecards
    
    Returns:
        Matched horse dict or None
    """
    runner_norm = normalize_name(runner_name)
    
    # Try exact match first
    for horse in horses:
        horse_norm = normalize_name(horse.get('name', ''))
        if horse_norm == runner_norm:
            return horse
    
    # Try partial match
    for horse in horses:
        horse_norm = normalize_name(horse.get('name', ''))
        if runner_norm in horse_norm or horse_norm in runner_norm:
            return horse
    
    return None


def fetch_odds_for_race(client: APIClient, race: Dict, race_date: str) -> List[Dict]:
    """
    Fetch Betfair odds for a single race.
    
    Returns:
        List of dicts with: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    race_id = race.get('race_id')
    venue = race.get('course', '')
    off_time = race.get('off_time', '')
    runners = race.get('runners', [])
    
    if not race_id or not venue or not runners:
        return []
    
    print(f"Fetching Betfair odds for {venue} {off_time}...", end=' ')
    
    # Find Betfair markets for this venue
    markets = find_betfair_markets(client, venue, race_date)
    
    if not markets:
        print(f"No Betfair markets found")
        return []
    
    # Parse off_time to match against market start times
    try:
        race_time = datetime.strptime(f"{race_date} {off_time}", '%Y-%m-%d %H:%M')
    except:
        print(f"Invalid off_time: {off_time}")
        return []
    
    # Find the market closest to this race time
    best_market = None
    min_time_diff = timedelta(hours=24)
    
    for market in markets:
        market_time = market.market_start_time.replace(tzinfo=None)
        time_diff = abs(market_time - race_time)
        
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            best_market = market
    
    if not best_market or min_time_diff > timedelta(minutes=30):
        print(f"No matching market (closest: {min_time_diff})")
        return []
    
    # Get odds for this market
    odds_map = get_market_odds(client, best_market.market_id)
    
    if not odds_map:
        print(f"No odds available")
        return []
    
    # Match Betfair runners to our horses
    odds_data = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    for runner in best_market.runners:
        selection_id = str(runner.selection_id)
        runner_name = runner.runner_name
        
        # Match to horse in racecards
        matched_horse = match_betfair_runner_to_horse(runner_name, runners)
        
        if not matched_horse:
            continue
        
        # Get odds for this selection
        odds = odds_map.get(selection_id)
        
        if not odds:
            continue
        
        odds_data.append({
            'race_id': race_id,
            'horse_id': matched_horse.get('horse_id'),
            'horse_name': matched_horse.get('name'),
            'bookmaker_odds': odds,
            'bookmaker': 'Betfair',
            'odds_timestamp': timestamp,
        })
    
    print(f"OK ({len(odds_data)} horses matched)")
    return odds_data


def fetch_all_odds(client: APIClient, racecards_path: Path, race_date: str) -> pd.DataFrame:
    """
    Fetch Betfair odds for all races in racecards JSON.
    
    Returns:
        DataFrame with columns: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    if not racecards_path.exists():
        raise FileNotFoundError(f"Racecards not found: {racecards_path}")
    
    with open(racecards_path, 'r', encoding='utf-8') as f:
        racecards = json.load(f)
    
    all_odds = []
    
    # Iterate through nested structure: region -> course -> time -> race
    for region in racecards:
        for course in racecards[region]:
            for off_time, race in racecards[region][course].items():
                odds = fetch_odds_for_race(client, race, race_date)
                all_odds.extend(odds)
    
    if not all_odds:
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    return pd.DataFrame(all_odds)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch odds from Betfair Exchange API.'
    )
    parser.add_argument(
        '--date',
        required=True,
        help='Date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--output',
        help='Output CSV path (default: data/raw/betfair_odds_YYYY-MM-DD.csv)'
    )
    
    args = parser.parse_args()
    
    # Paths
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/betfair_odds_{args.date}.csv")
    
    print(f"\nFetching Betfair odds for {args.date}...")
    print(f"Reading racecards from {racecards_path}")
    
    # Create Betfair client
    try:
        client = create_betfair_client()
    except Exception as e:
        print(f"\n❌ Failed to connect to Betfair: {e}")
        return
    
    try:
        # Fetch odds
        df_odds = fetch_all_odds(client, racecards_path, args.date)
        
        if len(df_odds) == 0:
            print("\n⚠️  No odds found!")
            print("\nPossible reasons:")
            print("1. No Betfair markets available for this date/venues")
            print("2. Venue names don't match between racecards and Betfair")
            print("3. Race times differ significantly")
            return
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_odds.to_csv(output_path, index=False)
        
        print(f"\n✓ Fetched {len(df_odds)} odds entries")
        print(f"✓ Saved to {output_path}")
        
        # Summary
        print(f"\nSummary:")
        print(f"  Total odds: {len(df_odds)}")
        print(f"  Unique races: {df_odds['race_id'].nunique()}")
        print(f"  Unique horses: {df_odds['horse_id'].nunique()}")
        print(f"  Average odds: {df_odds['bookmaker_odds'].mean():.2f}")
        
        print(f"\nNext step:")
        print(f"  python scripts/merge_odds_from_csv.py --date {args.date} --odds-csv {output_path}")
        
    finally:
        # Logout
        client.logout()
        print("\n✓ Logged out from Betfair")


if __name__ == '__main__':
    main()
