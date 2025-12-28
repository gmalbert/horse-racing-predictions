#!/usr/bin/env python
"""
Interactive tool to manually enter odds for races.

Since automated scraping is challenging (blocking, geo-restrictions, etc.),
this tool provides a user-friendly way to manually enter odds from any source.

Usage:
  python scripts/manual_odds_entry.py --date 2025-12-22
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_racecards(path: Path) -> dict:
    """Load racecards JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def manual_entry_session(racecards: dict, date: str) -> pd.DataFrame:
    """Interactive session to enter odds."""
    print("\n" + "="*60)
    print("MANUAL ODDS ENTRY")
    print("="*60)
    print("\nInstructions:")
    print("- Enter odds as decimal (e.g., 5.50) or fractional (e.g., 9/2)")
    print("- Press Enter to skip a horse")
    print("- Type 'quit' to finish early")
    print("- Type 'skip' to skip entire race")
    print("\n")
    
    all_odds = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    race_count = 0
    total_races = sum(
        len(times)
        for region in racecards.values()
        for course in region.values()
        for times in [course]
    )
    
    for region in racecards:
        for course in racecards[region]:
            for off_time, race in racecards[region][course].items():
                race_count += 1
                race_id = race.get('race_id')
                runners = race.get('runners', [])
                
                if not race_id or not runners:
                    continue
                
                print(f"\n{'─'*60}")
                print(f"Race {race_count}/{total_races}")
                print(f"Course: {course}")
                print(f"Time: {off_time}")
                print(f"Runners: {len(runners)}")
                print(f"{'─'*60}\n")
                
                # Ask if user wants to enter odds for this race
                response = input(f"Enter odds for this race? (y/n/quit): ").strip().lower()
                
                if response == 'quit':
                    print("\nQuitting early...")
                    break
                
                if response != 'y':
                    print("Skipping race...\n")
                    continue
                
                # Ask for bookmaker name once per race
                bookmaker = input("Bookmaker name (e.g., Bet365, William Hill): ").strip()
                if not bookmaker:
                    bookmaker = "Manual Entry"
                
                # Enter odds for each runner
                for i, runner in enumerate(runners, 1):
                    horse_name = runner.get('name', '')
                    horse_id = runner.get('horse_id')
                    
                    if not horse_id:
                        continue
                    
                    print(f"\n  [{i}/{len(runners)}] {horse_name}")
                    odds_input = input("      Odds (decimal or fractional): ").strip()
                    
                    if odds_input.lower() == 'quit':
                        print("\nQuitting early...")
                        break
                    
                    if odds_input.lower() == 'skip':
                        print("Skipping rest of race...\n")
                        break
                    
                    if not odds_input:
                        continue
                    
                    # Parse odds
                    decimal_odds = parse_odds(odds_input)
                    
                    if decimal_odds:
                        all_odds.append({
                            'race_id': race_id,
                            'horse_id': horse_id,
                            'horse_name': horse_name,
                            'bookmaker_odds': decimal_odds,
                            'bookmaker': bookmaker,
                            'odds_timestamp': timestamp,
                        })
                        print(f"      ✓ Recorded: {decimal_odds}")
                    else:
                        print(f"      ✗ Invalid odds format")
    
    return pd.DataFrame(all_odds) if all_odds else pd.DataFrame(
        columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp']
    )


def parse_odds(odds_str: str) -> float:
    """Parse odds from various formats."""
    odds_str = odds_str.strip().upper()
    
    # Handle special cases
    if odds_str in ['EVENS', 'EVS', '1/1']:
        return 2.0
    
    # Try fractional (e.g., "9/2")
    if '/' in odds_str:
        try:
            parts = odds_str.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                return round((num / den) + 1.0, 2)
        except:
            pass
    
    # Try decimal
    try:
        return round(float(odds_str), 2)
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Manually enter odds for horse races.'
    )
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--output', help='Output CSV path')
    
    args = parser.parse_args()
    
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/manual_odds_{args.date}.csv")
    
    if not racecards_path.exists():
        print(f"Error: Racecards not found at {racecards_path}")
        return
    
    print(f"\nLoading racecards from {racecards_path}")
    racecards = load_racecards(racecards_path)
    
    # Count total races
    total_races = sum(
        1
        for region in racecards.values()
        for course in region.values()
        for _ in course
    )
    
    print(f"Found {total_races} races")
    
    # Start manual entry
    df_odds = manual_entry_session(racecards, args.date)
    
    if len(df_odds) == 0:
        print("\n⚠️  No odds entered")
        return
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_odds.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved {len(df_odds)} odds entries to {output_path}")
    print(f"\nSummary:")
    print(f"  Unique races: {df_odds['race_id'].nunique()}")
    print(f"  Unique horses: {df_odds['horse_id'].nunique()}")
    print(f"  Bookmakers: {', '.join(df_odds['bookmaker'].unique())}")
    print(f"\nNext step:")
    print(f"  python scripts/merge_odds_from_csv.py --date {args.date} --odds-csv {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
