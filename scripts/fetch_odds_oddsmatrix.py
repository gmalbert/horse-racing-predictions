#!/usr/bin/env python
"""
Fetch horse racing odds from OddsMatrix API.

OddsMatrix provides professional horse racing data and odds from global bookmakers.
API docs: https://oddsmatrix.com/api-documentation

Usage:
  python scripts/fetch_odds_oddsmatrix.py --date 2025-12-22
  python scripts/fetch_odds_oddsmatrix.py --date 2025-12-22 --region uk
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


def get_oddsmatrix_credentials():
    """Get OddsMatrix API credentials from environment."""
    load_dotenv()
    
    api_key = os.getenv('ODDSMATRIX_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Missing OddsMatrix API key. Add to .env file:\n"
            "  ODDSMATRIX_API_KEY=your_api_key"
        )
    
    return api_key


def fetch_oddsmatrix_events(api_key: str, date_str: str, region: str = 'gb') -> List[Dict]:
    """
    Fetch horse racing events from OddsMatrix API.
    
    Args:
        api_key: OddsMatrix API key
        date_str: Date in YYYY-MM-DD format
        region: Region code (gb, us, au, ie, etc.)
    
    Returns:
        List of racing events with odds
    """
    # OddsMatrix API endpoint
    base_url = 'https://api.oddsmatrix.com/v1/horseracing/events'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    params = {
        'date': date_str,
        'region': region,
        'include_odds': 'true',
        'market_type': 'win',
    }
    
    try:
        print(f"Fetching events from OddsMatrix API...")
        resp = requests.get(base_url, headers=headers, params=params, timeout=30)
        
        if resp.status_code == 401:
            raise ValueError("Invalid OddsMatrix API key. Check your credentials.")
        
        if resp.status_code == 403:
            raise ValueError("Access forbidden. Check your OddsMatrix subscription plan.")
        
        if resp.status_code == 429:
            raise ValueError("Rate limit exceeded. Too many API requests.")
        
        if resp.status_code != 200:
            raise ValueError(f"OddsMatrix API error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        
        # Handle different response formats
        if isinstance(data, dict):
            events = data.get('events', []) or data.get('data', []) or data.get('races', [])
        else:
            events = data
        
        return events
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch from OddsMatrix: {e}")


def match_oddsmatrix_event_to_race(event: Dict, racecards: Dict) -> Optional[Dict]:
    """
    Match OddsMatrix event to race in racecards.
    
    Args:
        event: Event from OddsMatrix API
        racecards: Racecards data structure
    
    Returns:
        Matched race dict or None
    """
    # Extract event details (adjust keys based on actual API response)
    venue = (event.get('venue') or event.get('track') or 
             event.get('course') or event.get('location') or '')
    
    start_time = (event.get('start_time') or event.get('post_time') or 
                  event.get('off_time') or event.get('commence_time'))
    
    venue_norm = normalize_name(venue)
    
    # Parse start time
    try:
        if start_time:
            if isinstance(start_time, str):
                # Try parsing ISO format
                event_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                event_time = event_dt.strftime('%H:%M')
            else:
                event_time = None
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
                    # Check if time matches
                    if event_time and off_time == event_time:
                        return race
                    # If no time match, return first race at this venue (fallback)
                    if not event_time:
                        return race
    
    return None


def extract_odds_from_oddsmatrix_event(event: Dict, race: Dict) -> List[Dict]:
    """
    Extract odds from OddsMatrix event and match to horses.
    
    Returns:
        List of dicts with: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    race_id = race.get('race_id')
    runners = race.get('runners', [])
    
    if not race_id or not runners:
        return []
    
    odds_data = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Extract runners/selections from OddsMatrix event
    # The API might use 'runners', 'selections', 'horses', or 'participants'
    event_runners = (event.get('runners') or event.get('selections') or 
                     event.get('horses') or event.get('participants') or [])
    
    # Also check for odds in different structures
    if not event_runners:
        # Sometimes odds are nested under markets/bookmakers
        bookmakers = event.get('bookmakers', []) or event.get('odds', [])
        
        for bookmaker_data in bookmakers:
            bookmaker_name = (bookmaker_data.get('name') or 
                            bookmaker_data.get('bookmaker') or 'OddsMatrix')
            
            markets = bookmaker_data.get('markets', [])
            
            for market in markets:
                # Look for win market
                market_key = (market.get('key') or market.get('type') or '').lower()
                if 'win' not in market_key:
                    continue
                
                outcomes = market.get('outcomes', []) or market.get('selections', [])
                
                for outcome in outcomes:
                    horse_name_api = outcome.get('name') or outcome.get('runner')
                    odds = outcome.get('price') or outcome.get('odds')
                    
                    if not horse_name_api or not odds:
                        continue
                    
                    # Match to horse in racecards
                    matched_horse = match_horse_by_name(horse_name_api, runners)
                    
                    if matched_horse:
                        odds_data.append({
                            'race_id': race_id,
                            'horse_id': matched_horse.get('horse_id'),
                            'horse_name': matched_horse.get('name'),
                            'bookmaker_odds': float(odds),
                            'bookmaker': bookmaker_name,
                            'odds_timestamp': timestamp,
                        })
    else:
        # Direct runner format
        for runner_data in event_runners:
            horse_name_api = runner_data.get('name') or runner_data.get('horse')
            odds = (runner_data.get('odds') or runner_data.get('price') or 
                   runner_data.get('win_odds'))
            bookmaker = runner_data.get('bookmaker', 'OddsMatrix')
            
            if not horse_name_api or not odds:
                continue
            
            # Match to horse in racecards
            matched_horse = match_horse_by_name(horse_name_api, runners)
            
            if matched_horse:
                odds_data.append({
                    'race_id': race_id,
                    'horse_id': matched_horse.get('horse_id'),
                    'horse_name': matched_horse.get('name'),
                    'bookmaker_odds': float(odds),
                    'bookmaker': bookmaker,
                    'odds_timestamp': timestamp,
                })
    
    return odds_data


def match_horse_by_name(api_name: str, runners: List[Dict]) -> Optional[Dict]:
    """Match horse name from API to runner in racecards."""
    api_norm = normalize_name(api_name)
    
    # Try exact match first
    for runner in runners:
        runner_norm = normalize_name(runner.get('name', ''))
        if runner_norm == api_norm:
            return runner
    
    # Try partial match
    for runner in runners:
        runner_norm = normalize_name(runner.get('name', ''))
        if api_norm in runner_norm or runner_norm in api_norm:
            return runner
    
    return None


def fetch_all_odds(api_key: str, racecards_path: Path, date_str: str, region: str = 'gb') -> pd.DataFrame:
    """
    Fetch OddsMatrix odds for all races in racecards JSON.
    
    Returns:
        DataFrame with columns: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    if not racecards_path.exists():
        raise FileNotFoundError(f"Racecards not found: {racecards_path}")
    
    with open(racecards_path, 'r', encoding='utf-8') as f:
        racecards = json.load(f)
    
    # Fetch events from OddsMatrix
    events = fetch_oddsmatrix_events(api_key, date_str, region)
    
    if not events:
        print("No events returned from OddsMatrix API")
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    print(f"✓ Received {len(events)} events from OddsMatrix")
    
    all_odds = []
    matched_races = 0
    
    # Match events to races and extract odds
    for event in events:
        matched_race = match_oddsmatrix_event_to_race(event, racecards)
        
        if matched_race:
            matched_races += 1
            odds = extract_odds_from_oddsmatrix_event(event, matched_race)
            all_odds.extend(odds)
            
            if odds:
                print(f"✓ Matched race {matched_race.get('race_id')}: {len(odds)} odds entries")
            else:
                print(f"⚠ Matched race {matched_race.get('race_id')} but no odds found")
        else:
            venue = event.get('venue') or event.get('track') or 'Unknown'
            print(f"✗ No match for event at {venue}")
    
    print(f"\nMatched {matched_races}/{len(events)} races from OddsMatrix events")
    
    if not all_odds:
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    return pd.DataFrame(all_odds)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch horse racing odds from OddsMatrix API.'
    )
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--region', default='gb', help='Region code (gb, us, au, ie)')
    parser.add_argument('--output', help='Output CSV path')
    
    args = parser.parse_args()
    
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/oddsmatrix_odds_{args.date}.csv")
    
    print(f"\nFetching OddsMatrix odds for {args.date}...")
    print(f"Reading racecards from {racecards_path}")
    
    try:
        api_key = get_oddsmatrix_credentials()
    except Exception as e:
        print(f"\n❌ {e}")
        print("\nGet your API key from: https://oddsmatrix.com/")
        return
    
    try:
        df_odds = fetch_all_odds(api_key, racecards_path, args.date, args.region)
        
        if len(df_odds) == 0:
            print("\n⚠️  No odds found!")
            print("\nPossible reasons:")
            print("1. OddsMatrix doesn't have data for this date/region")
            print("2. Venue/time matching failed")
            print("3. No odds available in the API response")
            print("4. Region code is incorrect (try: gb, us, au, ie)")
            print("\nCheck OddsMatrix API documentation and response format.")
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
        print("1. Your OddsMatrix API key is valid")
        print("2. Your subscription plan includes horse racing data")
        print("3. The API endpoint and parameters are correct")
        print("4. Check OddsMatrix documentation: https://oddsmatrix.com/api-documentation")


if __name__ == '__main__':
    main()
