#!/usr/bin/env python
"""
Extract odds from racecards JSON to CSV format.

If your external scraper includes bookmaker odds embedded in the runners,
this script extracts them to CSV compatible with merge_odds_from_csv.py.

Usage:
  python scripts/extract_odds_from_racecards.py --date 2025-12-28
  python scripts/extract_odds_from_racecards.py --date 2025-12-28 --output custom_odds.csv
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple


def extract_odds_from_runner(runner: Dict, race_id: int, race_date: str) -> List[Tuple]:
    """
    Extract odds entries from a single runner.
    
    Returns list of tuples: (race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp)
    """
    rows = []
    horse_id = runner.get('horse_id')
    horse_name = runner.get('name', '')
    
    if not horse_id:
        return rows
    
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Check for direct odds fields
    for odds_key in ('bookmaker_odds', 'price', 'odds', 'best_price', 'best_odds'):
        if runner.get(odds_key):
            odds_value = runner.get(odds_key)
            try:
                odds_decimal = float(odds_value)
                rows.append((
                    race_id,
                    horse_id,
                    horse_name,
                    odds_decimal,
                    'Embedded',
                    timestamp
                ))
                break  # Only take first found odds
            except (ValueError, TypeError):
                continue
    
    # Check for nested bookmakers structure
    if runner.get('bookmakers'):
        bookmakers_data = runner.get('bookmakers')
        
        # Case 1: bookmakers is a dict with bookmaker names as keys
        if isinstance(bookmakers_data, dict):
            for bookmaker_name, odds_info in bookmakers_data.items():
                if isinstance(odds_info, list) and odds_info:
                    # Take first entry if it's a list
                    price = odds_info[0].get('price') or odds_info[0].get('odds')
                    if price:
                        try:
                            rows.append((
                                race_id,
                                horse_id,
                                horse_name,
                                float(price),
                                bookmaker_name,
                                timestamp
                            ))
                        except (ValueError, TypeError):
                            continue
                elif isinstance(odds_info, dict):
                    price = odds_info.get('price') or odds_info.get('odds')
                    if price:
                        try:
                            rows.append((
                                race_id,
                                horse_id,
                                horse_name,
                                float(price),
                                bookmaker_name,
                                timestamp
                            ))
                        except (ValueError, TypeError):
                            continue
        
        # Case 2: bookmakers is a list of bookmaker objects
        elif isinstance(bookmakers_data, list):
            for bookmaker_obj in bookmakers_data:
                if isinstance(bookmaker_obj, dict):
                    price = bookmaker_obj.get('price') or bookmaker_obj.get('odds')
                    bookmaker_name = bookmaker_obj.get('name') or bookmaker_obj.get('bookmaker') or 'Unknown'
                    if price:
                        try:
                            rows.append((
                                race_id,
                                horse_id,
                                horse_name,
                                float(price),
                                bookmaker_name,
                                timestamp
                            ))
                        except (ValueError, TypeError):
                            continue
    
    return rows


def extract_all_odds(racecards_path: Path) -> List[Tuple]:
    """
    Extract all odds from racecards JSON.
    
    Returns list of tuples: (race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp)
    """
    if not racecards_path.exists():
        raise FileNotFoundError(f"Racecards not found: {racecards_path}")
    
    with open(racecards_path, 'r', encoding='utf-8') as f:
        racecards = json.load(f)
    
    all_odds = []
    
    # Iterate through nested structure: region -> course -> time -> race
    for region in racecards.values():
        for course in region.values():
            for off_time, race in course.items():
                race_id = race.get('race_id')
                race_date = race.get('date', '')
                runners = race.get('runners', [])
                
                if not race_id or not runners:
                    continue
                
                # Extract odds from each runner
                for runner in runners:
                    odds_entries = extract_odds_from_runner(runner, race_id, race_date)
                    all_odds.extend(odds_entries)
    
    return all_odds


def main():
    parser = argparse.ArgumentParser(
        description='Extract bookmaker odds from racecards JSON to CSV.'
    )
    parser.add_argument(
        '--date',
        required=True,
        help='Date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--output',
        help='Output CSV path (default: data/raw/odds_from_racecards_YYYY-MM-DD.csv)'
    )
    
    args = parser.parse_args()
    
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/odds_from_racecards_{args.date}.csv")
    
    print(f"\nExtracting odds from {racecards_path}...")
    
    try:
        odds_data = extract_all_odds(racecards_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ Error reading racecards: {e}")
        return
    
    if not odds_data:
        print("\n⚠️  No odds found in racecards JSON!")
        print("\nPossible reasons:")
        print("1. Runners don't have odds fields (bookmaker_odds, price, odds)")
        print("2. Runners don't have nested 'bookmakers' structure")
        print("3. Odds need to be fetched separately (use fetch_odds_*.py scripts)")
        return
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
        writer.writerows(odds_data)
    
    print(f"\n✓ Extracted {len(odds_data)} odds entries")
    print(f"✓ Saved to {output_path}")
    
    # Summary
    unique_races = len(set(row[0] for row in odds_data))
    unique_horses = len(set(row[1] for row in odds_data))
    unique_bookmakers = set(row[4] for row in odds_data)
    
    print(f"\nSummary:")
    print(f"  Total odds: {len(odds_data)}")
    print(f"  Unique races: {unique_races}")
    print(f"  Unique horses: {unique_horses}")
    print(f"  Bookmakers: {', '.join(unique_bookmakers)}")
    
    print(f"\nNext step:")
    print(f"  python scripts/merge_odds_from_csv.py --date {args.date} --odds-csv {output_path}")


if __name__ == '__main__':
    main()
