#!/usr/bin/env python
"""
Fetch horse racing odds from Scrapebet API.

Scrapebet provides global odds data from multiple bookmakers.
API docs: https://scrapebet.com/api

Usage:
  python scripts/fetch_odds_scrapebet.py --date 2025-12-22
  python scripts/fetch_odds_scrapebet.py --date 2025-12-22 --sport horseracing
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


def normalize_name(name: str) -> str:
    """Normalize horse/venue name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def get_scrapebet_client():
    """Get Scrapebet API key from environment."""
    load_dotenv()
    
    api_key = os.getenv('SCRAPEBET_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Missing Scrapebet API key. Add to .env file:\n"
            "  SCRAPEBET_API_KEY=your_api_key"
        )
    
    return api_key


def fetch_scrapebet_odds(api_key: str, sport: str = 'horseracing', region: str = 'uk') -> List[Dict]:
    """
    Fetch odds from Scrapebet API.
    
    Args:
        api_key: Scrapebet API key
        sport: Sport type (e.g., 'horseracing', 'horse_racing')
        region: Region code (e.g., 'uk', 'us', 'au')
    
    Returns:
        List of events with odds data
    """
    # Scrapebet API endpoint (adjust based on actual API documentation)
    base_url = 'https://api.scrapebet.com/v1/odds'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    params = {
        'sport': sport,
        'region': region,
        'markets': 'win',  # Win market (to-win odds)
    }
    
    try:
        print(f"Fetching odds from Scrapebet API...")
        resp = requests.get(base_url, headers=headers, params=params, timeout=30)
        
        if resp.status_code == 401:
            raise ValueError("Invalid Scrapebet API key. Check your credentials.")
        
        if resp.status_code == 403:
            raise ValueError("Access forbidden. Check your Scrapebet subscription plan.")
        
        if resp.status_code != 200:
            raise ValueError(f"Scrapebet API error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        
        # Handle different response formats
        if isinstance(data, dict):
            events = data.get('events', []) or data.get('data', [])
        else:
            events = data
        
        return events
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch from Scrapebet: {e}")


def match_scrapebet_event_to_race(event: Dict, racecards: Dict) -> Optional[Dict]:
    """
    Match Scrapebet event to race in racecards.
    
    Args:
        event: Event from Scrapebet API
        racecards: Racecards data structure
    
    Returns:
        Matched race dict or None
    """
    event_name = event.get('event_name', '') or event.get('name', '')
    commence_time = event.get('commence_time') or event.get('start_time')
    venue = event.get('venue', '') or event.get('location', '')
    
    venue_norm = normalize_name(venue)
    
    # Parse commence time
    try:
        if commence_time:
            event_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            event_time = event_dt.strftime('%H:%M')
        else:
            event_time = None
    except:
        event_time = None
    
    # Search through racecards for matching venue and time
    for region in racecards:
        for course in racecards[region]:
            course_norm = normalize_name(course)
            
            # Check if venue matches
            if venue_norm and (venue_norm in course_norm or course_norm in venue_norm):
                for off_time, race in racecards[region][course].items():
                    # Check if time matches (within tolerance)
                    if event_time and off_time == event_time:
                        return race
    
    return None


def extract_odds_from_event(event: Dict, race: Dict) -> List[Dict]:
    """
    Extract odds from Scrapebet event and match to horses.
    
    Returns:
        List of dicts with: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    race_id = race.get('race_id')
    runners = race.get('runners', [])
    
    if not race_id or not runners:
        return []
    
    odds_data = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Extract bookmakers and odds from event
    bookmakers = event.get('bookmakers', []) or event.get('odds', [])
    
    for bookmaker_data in bookmakers:
        bookmaker_name = bookmaker_data.get('name') or bookmaker_data.get('bookmaker') or 'Scrapebet'
        
        # Get markets
        markets = bookmaker_data.get('markets', [])
        
        for market in markets:
            # Look for win/moneyline market
            market_key = market.get('key', '').lower()
            if 'win' not in market_key and 'h2h' not in market_key and 'moneyline' not in market_key:
                continue
            
            # Get outcomes (horses with odds)
            outcomes = market.get('outcomes', [])
            
            for outcome in outcomes:
                horse_name_api = outcome.get('name', '')
                odds = outcome.get('price') or outcome.get('odds')
                
                if not horse_name_api or not odds:
                    continue
                
                # Match to horse in racecards
                matched_horse = None
                horse_name_norm = normalize_name(horse_name_api)
                
                for runner in runners:
                    runner_norm = normalize_name(runner.get('name', ''))
                    if runner_norm == horse_name_norm or horse_name_norm in runner_norm or runner_norm in horse_name_norm:
                        matched_horse = runner
                        break
                
                if matched_horse:
                    odds_data.append({
                        'race_id': race_id,
                        'horse_id': matched_horse.get('horse_id'),
                        'horse_name': matched_horse.get('name'),
                        'bookmaker_odds': float(odds),
                        'bookmaker': bookmaker_name,
                        'odds_timestamp': timestamp,
                    })
    
    return odds_data


def fetch_all_odds(api_key: str, racecards_path: Path, sport: str = 'horseracing') -> pd.DataFrame:
    """
    Fetch Scrapebet odds for all races in racecards JSON.
    
    Returns:
        DataFrame with columns: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    if not racecards_path.exists():
        raise FileNotFoundError(f"Racecards not found: {racecards_path}")
    
    with open(racecards_path, 'r', encoding='utf-8') as f:
        racecards = json.load(f)
    
    # Fetch odds from Scrapebet
    events = fetch_scrapebet_odds(api_key, sport)
    
    if not events:
        print("No events returned from Scrapebet API")
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    print(f"✓ Received {len(events)} events from Scrapebet")
    
    all_odds = []
    matched_races = 0
    
    # Match events to races and extract odds
    for event in events:
        matched_race = match_scrapebet_event_to_race(event, racecards)
        
        if matched_race:
            matched_races += 1
            odds = extract_odds_from_event(event, matched_race)
            all_odds.extend(odds)
            
            if odds:
                print(f"✓ Matched race {matched_race.get('race_id')}: {len(odds)} odds entries")
    
    print(f"\nMatched {matched_races} races from Scrapebet events")
    
    if not all_odds:
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    return pd.DataFrame(all_odds)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch horse racing odds from Scrapebet API.'
    )
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--sport', default='horseracing', help='Sport key (default: horseracing)')
    parser.add_argument('--output', help='Output CSV path')
    
    args = parser.parse_args()
    
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/scrapebet_odds_{args.date}.csv")
    
    print(f"\nFetching Scrapebet odds for {args.date}...")
    print(f"Reading racecards from {racecards_path}")
    
    try:
        api_key = get_scrapebet_client()
    except Exception as e:
        print(f"\n❌ {e}")
        print("\nGet your API key from: https://scrapebet.com/")
        return
    
    try:
        df_odds = fetch_all_odds(api_key, racecards_path, args.sport)
        
        if len(df_odds) == 0:
            print("\n⚠️  No odds found!")
            print("\nPossible reasons:")
            print("1. Scrapebet doesn't have data for this date/region")
            print("2. Sport key is incorrect (try: 'horseracing', 'horse_racing', 'racing')")
            print("3. Venue/time matching failed")
            print("4. No markets available")
            print("\nTry using --sport with different values or check Scrapebet API docs.")
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
        print(f"  Bookmakers: {', '.join(df_odds['bookmaker'].unique())}")
        print(f"  Average odds: {df_odds['bookmaker_odds'].mean():.2f}")
        
        print(f"\nNext step:")
        print(f"  python scripts/merge_odds_from_csv.py --date {args.date} --odds-csv {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. Your Scrapebet API key is valid")
        print("2. Your subscription plan includes horse racing data")
        print("3. The API endpoint and parameters are correct")
        print("4. Check Scrapebet documentation: https://scrapebet.com/api")


if __name__ == '__main__':
    main()
