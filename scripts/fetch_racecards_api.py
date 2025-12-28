#!/usr/bin/env python
"""
Fetch racecards from The Racing API.

This uses your Racing API credentials (no web scraping, no blocking issues).
Outputs JSON in the same format as the external scraper.

Usage:
  python scripts/fetch_racecards_api.py --date 2025-12-28
  python scripts/fetch_racecards_api.py --date 2025-12-28 --region GB
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


def get_credentials():
    """Load Racing API credentials from environment."""
    load_dotenv()
    
    username = os.getenv('RACING_API_USERNAME')
    password = os.getenv('RACING_API_PASSWORD')
    
    if not username or not password:
        raise ValueError(
            "Racing API credentials not found. Add to .env file:\n"
            "  RACING_API_USERNAME=your_username\n"
            "  RACING_API_PASSWORD=your_password"
        )
    
    return username, password


def fetch_racecards_from_api(date_str, region=None):
    """
    Fetch racecards from The Racing API for a specific date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        region: Optional region filter (GB, IE, US, etc.)
    
    Returns:
        List of race data from API
    """
    username, password = get_credentials()
    
    base_url = "https://api.theracingapi.com/v1"
    endpoint = f"{base_url}/racecards"
    
    params = {'date': date_str}
    if region:
        params['region'] = region
    
    print(f"Fetching racecards from The Racing API for {date_str}...")
    if region:
        print(f"Region filter: {region}")
    
    try:
        response = requests.get(
            endpoint,
            auth=(username, password),
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # API might return races directly or nested under 'races' or 'racecards'
        if isinstance(data, dict):
            races = data.get('racecards') or data.get('races') or data.get('data') or []
        else:
            races = data
        
        return races
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("❌ Authentication failed. Check your Racing API credentials in .env")
        elif response.status_code == 404:
            print(f"❌ No racecards found for {date_str}")
        else:
            print(f"❌ HTTP error {response.status_code}: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return []


def transform_to_standard_format(api_races, date_str):
    """
    Transform Racing API format to the standard nested format:
    {region: {course: {off_time: race}}}
    """
    races = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for race in api_races:
        # Extract race details (adjust keys based on actual API response)
        race_id = race.get('race_id') or race.get('id')
        course = race.get('course') or race.get('track') or race.get('venue')
        region = race.get('region') or race.get('country') or 'GB'
        
        # Parse race time
        race_time = race.get('off_time') or race.get('start_time') or race.get('post_time')
        if race_time:
            try:
                dt = datetime.fromisoformat(race_time.replace('Z', '+00:00'))
                off_time = dt.strftime('%H:%M')
            except:
                off_time = race_time[:5] if len(race_time) >= 5 else '00:00'
        else:
            off_time = '00:00'
        
        # Build standardized race object
        race_obj = {
            'race_id': int(race_id) if race_id else None,
            'date': date_str,
            'off_time': off_time,
            'course': course,
            'course_id': race.get('course_id'),
            'region': region,
            'race_name': race.get('race_name') or race.get('name') or '',
            'race_type': race.get('race_type') or 'Flat',
            'distance': race.get('distance') or '',
            'distance_f': race.get('distance_furlongs'),
            'distance_y': race.get('distance_yards'),
            'going': race.get('going') or '',
            'surface': race.get('surface') or 'Turf',
            'race_class': race.get('class') or race.get('race_class'),
            'pattern': race.get('pattern') or '',
            'age_band': race.get('age_band'),
            'rating_band': race.get('rating_band'),
            'field_size': race.get('field_size') or race.get('num_runners'),
            'prize': race.get('prize') or race.get('prize_money'),
            'handicap': race.get('handicap', False),
            'href': race.get('url') or '',
            'runners': []
        }
        
        # Extract runners
        runners = race.get('runners') or race.get('horses') or []
        for runner in runners:
            runner_obj = {
                'horse_id': runner.get('horse_id'),
                'name': runner.get('horse_name') or runner.get('name') or '',
                'number': runner.get('number') or runner.get('saddle_cloth'),
                'draw': runner.get('draw'),
                'age': runner.get('age'),
                'jockey': runner.get('jockey_name') or runner.get('jockey') or '',
                'jockey_id': runner.get('jockey_id'),
                'trainer': runner.get('trainer_name') or runner.get('trainer') or '',
                'trainer_id': runner.get('trainer_id'),
                'ofr': runner.get('official_rating') or runner.get('ofr'),
                'rpr': runner.get('rpr'),
                'ts': runner.get('topspeed') or runner.get('ts'),
                'lbs': runner.get('weight_lbs') or runner.get('weight'),
                'headgear': runner.get('headgear'),
                'form': runner.get('form') or '',
                'last_run': runner.get('days_since_last_run') or runner.get('last_run'),
                'non_runner': runner.get('non_runner', False),
            }
            race_obj['runners'].append(runner_obj)
        
        # Store in nested structure
        races[region][course][off_time] = race_obj
    
    # Convert defaultdict to regular dict for JSON serialization
    return {
        region: {
            course: dict(times)
            for course, times in courses.items()
        }
        for region, courses in races.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fetch racecards from The Racing API.'
    )
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--region', help='Region filter (GB, IE, US, etc.)')
    
    args = parser.parse_args()
    
    try:
        # Fetch from API
        api_races = fetch_racecards_from_api(args.date, args.region)
        
        if not api_races:
            print("\n⚠️  No racecards found")
            print("Possible reasons:")
            print("1. No races scheduled for this date")
            print("2. API credentials invalid")
            print("3. Region filter excluded all races")
            sys.exit(1)
        
        print(f"✓ Received {len(api_races)} races from API")
        
        # Transform to standard format
        racecards = transform_to_standard_format(api_races, args.date)
        
        # Count total races
        total_races = sum(
            len(times)
            for region in racecards.values()
            for course in region.values()
            for times in [course]
        )
        
        # Save to JSON
        output_dir = Path('data/raw')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'racecards_{args.date}.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(racecards, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {total_races} races to {output_path}")
        
        # Summary
        print(f"\nSummary:")
        print(f"  Regions: {', '.join(racecards.keys())}")
        print(f"  Total races: {total_races}")
        
        # Count runners
        total_runners = sum(
            len(race.get('runners', []))
            for region in racecards.values()
            for course in region.values()
            for race in course.values()
        )
        print(f"  Total runners: {total_runners}")
        
        print(f"\nNext step:")
        print(f"  python scripts/predict_todays_races.py --date {args.date}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
