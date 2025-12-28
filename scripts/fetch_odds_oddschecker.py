#!/usr/bin/env python
"""
Fetch horse racing odds from Oddschecker (US-accessible).

Oddschecker aggregates odds from multiple bookmakers and works globally.
Reads racecards JSON and scrapes odds, outputting CSV compatible with merge_odds_from_csv.py.

Usage:
  python scripts/fetch_odds_oddschecker.py --date 2025-12-22
  python scripts/fetch_odds_oddschecker.py --date 2025-12-22 --region uk
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from lxml import html


def normalize_name(name: str) -> str:
    """Normalize horse/venue name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def get_oddschecker_race_url(course: str, race_time: str, date: str, region: str = 'uk') -> Optional[str]:
    """
    Construct Oddschecker URL for a race.
    
    Oddschecker URLs typically follow pattern:
    https://www.oddschecker.com/horse-racing/{course}/{time}-race
    """
    # Normalize course name for URL (lowercase, replace spaces with dashes)
    course_slug = course.lower().replace(' ', '-').replace("'", "")
    
    # Parse time (HH:MM) to create race slug
    try:
        time_obj = datetime.strptime(race_time, '%H:%M')
        time_slug = time_obj.strftime('%H%M')
    except:
        return None
    
    # Oddschecker URL format
    base_url = f"https://www.oddschecker.com/horse-racing/{course_slug}/{time_slug}"
    
    return base_url


def fetch_oddschecker_odds(url: str, race_id: int, runners: List[Dict]) -> List[Dict]:
    """
    Fetch odds from Oddschecker for a specific race.
    
    Returns list of dicts with: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
    except:
        return []
    
    try:
        doc = html.fromstring(resp.content)
    except:
        return []
    
    odds_data = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Oddschecker shows odds in a table with horses as rows and bookmakers as columns
    # Try to extract odds for each runner
    
    for runner in runners:
        horse_name = runner.get('name', '')
        horse_id = runner.get('horse_id')
        horse_norm = normalize_name(horse_name)
        
        if not horse_id or not horse_name:
            continue
        
        # Try to find the horse's row in the odds table
        # Oddschecker typically uses class names like 'diff-row' or 'eventTableRow'
        
        # Look for elements containing horse name
        horse_rows = doc.xpath(f".//tr[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{horse_norm}')]")
        
        if not horse_rows:
            continue
        
        # Get the first bookmaker's odds from this row
        # Oddschecker shows odds in <td> elements with class containing 'odds'
        odds_cells = horse_rows[0].xpath(".//td[contains(@class, 'odds')]//span[@class='odds']")
        
        if odds_cells:
            odds_text = odds_cells[0].text_content().strip()
            decimal_odds = convert_fractional_to_decimal(odds_text)
            
            if decimal_odds:
                # Get bookmaker name from column header or default to 'Oddschecker'
                bookmaker = 'Oddschecker Best'
                
                odds_data.append({
                    'race_id': race_id,
                    'horse_id': horse_id,
                    'horse_name': horse_name,
                    'bookmaker_odds': decimal_odds,
                    'bookmaker': bookmaker,
                    'odds_timestamp': timestamp,
                })
    
    return odds_data


def convert_fractional_to_decimal(fractional: str) -> Optional[float]:
    """Convert fractional odds to decimal."""
    fractional = fractional.strip().upper()
    
    if fractional in ['EVENS', 'EVS', '1/1']:
        return 2.0
    
    if fractional in ['SP', 'N/A', '-', '']:
        return None
    
    match = re.match(r'(\d+)/(\d+)', fractional)
    if match:
        num = float(match.group(1))
        den = float(match.group(2))
        return round((num / den) + 1.0, 2)
    
    try:
        return round(float(fractional), 2)
    except:
        return None


def fetch_all_odds(racecards_path: Path, region: str = 'uk') -> pd.DataFrame:
    """
    Fetch odds for all races in racecards JSON from Oddschecker.
    
    Note: This is a basic implementation. Oddschecker's actual HTML structure
    may vary and may require more sophisticated parsing or browser automation.
    """
    if not racecards_path.exists():
        raise FileNotFoundError(f"Racecards not found: {racecards_path}")
    
    with open(racecards_path, 'r', encoding='utf-8') as f:
        racecards = json.load(f)
    
    all_odds = []
    
    for region_name in racecards:
        for course in racecards[region_name]:
            for off_time, race in racecards[region_name][course].items():
                race_id = race.get('race_id')
                runners = race.get('runners', [])
                
                if not race_id or not runners:
                    continue
                
                # Try to construct Oddschecker URL
                date = racecards_path.stem.split('_')[-1]  # Extract date from filename
                url = get_oddschecker_race_url(course, off_time, date, region)
                
                if not url:
                    continue
                
                print(f"Fetching odds for {course} {off_time}...", end=' ')
                
                odds = fetch_oddschecker_odds(url, race_id, runners)
                
                if odds:
                    print(f"OK ({len(odds)} horses)")
                    all_odds.extend(odds)
                else:
                    print("No odds found")
                
                time.sleep(2)  # Be polite
    
    if not all_odds:
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    return pd.DataFrame(all_odds)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch horse racing odds from Oddschecker (US-accessible).'
    )
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--region', default='uk', help='Region code (uk, us, au)')
    parser.add_argument('--output', help='Output CSV path')
    
    args = parser.parse_args()
    
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/oddschecker_odds_{args.date}.csv")
    
    print(f"\nFetching Oddschecker odds for {args.date}...")
    print(f"Reading racecards from {racecards_path}")
    
    df_odds = fetch_all_odds(racecards_path, args.region)
    
    if len(df_odds) == 0:
        print("\n⚠️  No odds found!")
        print("\nOddschecker may have changed their HTML structure or is blocking requests.")
        print("\nRecommended alternatives for US users:")
        print("1. Manual CSV entry from bookmaker websites")
        print("2. Use a commercial horse racing API (e.g., Sportradar)")
        print("3. TwinSpires API (US horse racing)")
        print("4. Browser automation with Selenium")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_odds.to_csv(output_path, index=False)
    
    print(f"\n✓ Fetched {len(df_odds)} odds entries")
    print(f"✓ Saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Unique races: {df_odds['race_id'].nunique()}")
    print(f"  Unique horses: {df_odds['horse_id'].nunique()}")
    
    print(f"\nNext step:")
    print(f"  python scripts/merge_odds_from_csv.py --date {args.date} --odds-csv {output_path}")


if __name__ == '__main__':
    main()
