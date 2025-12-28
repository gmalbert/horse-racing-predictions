#!/usr/bin/env python
"""
Scrape bookmaker odds from Racing Post race pages.

Reads racecards JSON and scrapes odds for each race, outputting CSV compatible
with merge_odds_from_csv.py.

Usage:
  python scripts/scrape_odds.py --date 2025-12-22
  python scripts/scrape_odds.py --date 2025-12-22 --bookmakers "Bet365,William Hill"
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from lxml import html
import pandas as pd


def get_request(url: str, session: requests.Session = None) -> Tuple[int, requests.Response]:
    """Make HTTP GET request with anti-bot headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-GB,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
    
    if session is None:
        session = requests.Session()
    
    for attempt in range(3):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.status_code, resp
            elif resp.status_code == 406:
                print(f"Warning: Racing Post blocking request (406). Consider using browser automation.")
                return resp.status_code, resp
            time.sleep(1 + attempt)
        except requests.RequestException as e:
            if attempt == 2:
                print(f"Failed to fetch {url}: {e}")
                return 0, None
            time.sleep(2 ** attempt)
    
    return 0, None


def normalize_name(name: str) -> str:
    """Normalize horse name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)     # Collapse whitespace
    return name


def scrape_race_odds(race_url: str, race_id: int, runners: List[Dict], session: requests.Session) -> List[Dict]:
    """
    Scrape odds for a single race.
    
    Returns list of dicts with: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    print(f"Scraping odds for race {race_id}...", end=' ')
    
    status, resp = get_request(race_url, session)
    
    if status != 200 or not resp:
        print(f"FAILED (status: {status})")
        return []
    
    try:
        doc = html.fromstring(resp.content)
    except Exception as e:
        print(f"FAILED (parse error: {e})")
        return []
    
    odds_data = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Try to find odds on the page
    # Racing Post shows odds in various formats - we'll try multiple approaches
    
    # Approach 1: Look for odds in runner cards
    for runner in runners:
        horse_name = runner.get('name', '')
        horse_id = runner.get('horse_id')
        
        if not horse_id:
            continue
        
        # Try to find odds for this horse
        # Racing Post typically shows odds in format like "5/1", "7/2", "EVENS", etc.
        
        # Look for elements containing horse name and odds
        horse_name_norm = normalize_name(horse_name)
        
        # Try different XPath patterns to find odds
        odds_patterns = [
            f".//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{horse_name_norm}')]//span[contains(@class, 'odds')]",
            f".//div[contains(@data-test-selector, 'runner')]//span[contains(@class, 'price')]",
            f".//span[contains(@class, 'RC-runnerInfo__odds')]",
        ]
        
        best_odds = None
        bookmaker = "Racing Post"  # Default bookmaker
        
        for pattern in odds_patterns:
            try:
                odds_elements = doc.xpath(pattern)
                if odds_elements:
                    odds_text = odds_elements[0].text_content().strip()
                    # Convert fractional to decimal
                    decimal_odds = convert_fractional_to_decimal(odds_text)
                    if decimal_odds:
                        best_odds = decimal_odds
                        break
            except Exception:
                continue
        
        if best_odds:
            odds_data.append({
                'race_id': race_id,
                'horse_id': horse_id,
                'horse_name': horse_name,
                'bookmaker_odds': best_odds,
                'bookmaker': bookmaker,
                'odds_timestamp': timestamp
            })
    
    print(f"OK ({len(odds_data)} horses with odds)")
    return odds_data


def convert_fractional_to_decimal(fractional: str) -> float:
    """Convert fractional odds (e.g., '5/1', '7/2', 'EVENS') to decimal."""
    fractional = fractional.strip().upper()
    
    # Handle special cases
    if fractional in ['EVENS', 'EVS', '1/1']:
        return 2.0
    
    if fractional in ['SP', 'N/A', '-']:
        return None
    
    # Parse fraction
    match = re.match(r'(\d+)/(\d+)', fractional)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        return round((numerator / denominator) + 1.0, 2)
    
    # Try parsing as decimal directly
    try:
        return round(float(fractional), 2)
    except ValueError:
        return None


def scrape_all_odds(racecards_path: Path, bookmakers: List[str] = None) -> pd.DataFrame:
    """
    Scrape odds for all races in racecards JSON.
    
    Args:
        racecards_path: Path to racecards JSON file
        bookmakers: List of bookmaker names to filter (None = all)
    
    Returns:
        DataFrame with columns: race_id, horse_id, horse_name, bookmaker_odds, bookmaker, odds_timestamp
    """
    if not racecards_path.exists():
        raise FileNotFoundError(f"Racecards not found: {racecards_path}")
    
    with open(racecards_path, 'r', encoding='utf-8') as f:
        racecards = json.load(f)
    
    all_odds = []
    session = requests.Session()
    
    # Iterate through nested structure: region -> course -> time -> race
    for region in racecards:
        for course in racecards[region]:
            for off_time, race in racecards[region][course].items():
                race_id = race.get('race_id')
                race_url = race.get('href')
                runners = race.get('runners', [])
                
                if not race_id or not race_url or not runners:
                    continue
                
                # Scrape odds for this race
                odds = scrape_race_odds(race_url, race_id, runners, session)
                all_odds.extend(odds)
                
                # Be polite - rate limit requests
                time.sleep(1.5)
    
    if not all_odds:
        print("\nWarning: No odds found. Racing Post may be blocking requests.")
        print("Consider using browser automation (Selenium) or a different odds source.")
        return pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'bookmaker_odds', 'bookmaker', 'odds_timestamp'])
    
    df = pd.DataFrame(all_odds)
    
    # Filter by bookmakers if specified
    if bookmakers:
        df = df[df['bookmaker'].isin(bookmakers)]
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Scrape bookmaker odds from Racing Post race pages.'
    )
    parser.add_argument(
        '--date',
        required=True,
        help='Date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--bookmakers',
        help='Comma-separated list of bookmaker names to include (default: all)'
    )
    parser.add_argument(
        '--output',
        help='Output CSV path (default: data/raw/odds_YYYY-MM-DD.csv)'
    )
    
    args = parser.parse_args()
    
    # Parse bookmakers
    bookmakers = None
    if args.bookmakers:
        bookmakers = [b.strip() for b in args.bookmakers.split(',')]
    
    # Paths
    racecards_path = Path(f"data/raw/racecards_{args.date}.json")
    output_path = Path(args.output) if args.output else Path(f"data/raw/odds_{args.date}.csv")
    
    print(f"\nScraping odds for {args.date}...")
    print(f"Reading racecards from {racecards_path}")
    
    # Scrape odds
    df_odds = scrape_all_odds(racecards_path, bookmakers)
    
    if len(df_odds) == 0:
        print("\n⚠️  No odds found!")
        print("\nRacing Post is likely blocking automated requests.")
        print("Recommended alternatives:")
        print("1. Use Selenium with Chrome/Firefox for browser automation")
        print("2. Use Betfair API for exchange odds")
        print("3. Use a commercial odds feed service")
        print("4. Manual entry from bookmaker websites")
        return
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_odds.to_csv(output_path, index=False)
    
    print(f"\n✓ Scraped {len(df_odds)} odds entries")
    print(f"✓ Saved to {output_path}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total odds: {len(df_odds)}")
    print(f"  Unique races: {df_odds['race_id'].nunique()}")
    print(f"  Unique horses: {df_odds['horse_id'].nunique()}")
    print(f"  Bookmakers: {', '.join(df_odds['bookmaker'].unique())}")
    
    print(f"\nNext step:")
    print(f"  python scripts/merge_odds_from_csv.py --date {args.date} --odds-csv {output_path}")


if __name__ == '__main__':
    main()
