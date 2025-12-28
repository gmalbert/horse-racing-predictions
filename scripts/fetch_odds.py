#!/usr/bin/env python3
"""Fetch bookmaker odds from The Odds API and merge into predictions CSV.

Usage:
  python scripts/fetch_odds.py --date YYYY-MM-DD

This script expects an `ODDS_API_KEY` in the environment (use .env).
It will update `data/processed/predictions_{date}.csv` adding/overwriting
columns: `bookmaker_odds`, `bookmaker`, `odds_timestamp`.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import time

import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def get_api_key():
    load_dotenv(ROOT / '.env')
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        raise SystemExit('ODDS_API_KEY not found in environment. Add it to .env')
    return key


def fetch_odds_for_date(api_key, date_str):
    """Call The Odds API to get horse racing odds for the date.

    Returns parsed JSON list of events or raises on error.
    """
    # The Odds API v4 sports key for UK horse racing isn't documented here; try general horse_racing
    base = 'https://api.the-odds-api.com/v4/sports/{sport}/odds'
    sport_candidates = ['horse_racing', 'horse_racing_gb', 'horseracing', 'racing_horse']

    params = {
        'regions': 'uk',
        'markets': 'h2h',
        'dateFormat': 'iso',
        'oddsFormat': 'decimal',
        'date': date_str,
        'apiKey': api_key,
    }

    last_err = None
    for sport in sport_candidates:
        url = base.format(sport=sport)
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        # If unknown sport, try next candidate
        last_err = (resp.status_code, resp.text)
        # small delay between tries
        time.sleep(0.5)

    # If we get here, all attempts failed
    code, text = last_err
    raise SystemExit(f'Odds API error {code}: {text}')


def match_and_merge(predictions: pd.DataFrame, odds_events: list) -> pd.DataFrame:
    """Try to find a matching odds outcome for each prediction row.

    Matching strategy (heuristic):
      - For each event from odds API, inspect bookmakers -> markets -> outcomes
      - If outcome['name'] contains the horse name (case-insensitive containment), accept
      - Prefer the first matching bookmaker/odds found
    """
    # Prepare columns
    if 'bookmaker_odds' not in predictions.columns:
        predictions['bookmaker_odds'] = pd.NA
    if 'bookmaker' not in predictions.columns:
        predictions['bookmaker'] = pd.NA
    if 'odds_timestamp' not in predictions.columns:
        predictions['odds_timestamp'] = pd.NA

    # Build flattened list of outcomes for quick lookup
    outcomes = []
    for ev in odds_events:
        commence = ev.get('commence_time')
        site = ev.get('site') or ev.get('sport_nice')
        for bm in ev.get('bookmakers', []):
            bkey = bm.get('key') or bm.get('title')
            for m in bm.get('markets', []):
                for o in m.get('outcomes', []):
                    outcomes.append({
                        'event': ev,
                        'commence_time': commence,
                        'bookmaker': bkey,
                        'market_key': m.get('key'),
                        'name': o.get('name'),
                        'price': o.get('price'),
                    })

    # Lowercase name for search
    for idx, row in predictions.iterrows():
        horse = str(row.get('horse', '')).lower()
        course = str(row.get('course', '')).lower()
        race_time = str(row.get('race_time_gmt', '')).lower()  # short hh:mm GMT stored previously

        # simple linear search through outcomes (can optimize if needed)
        found = False
        for o in outcomes:
            if not o['name']:
                continue
            name = str(o['name']).lower()
            # match horse name contained in outcome name
            if horse in name or name in horse:
                # optionally check course or time similarity in event metadata
                # Accept the match
                predictions.at[idx, 'bookmaker_odds'] = o['price']
                predictions.at[idx, 'bookmaker'] = o['bookmaker']
                predictions.at[idx, 'odds_timestamp'] = o['commence_time'] or pd.NA
                found = True
                break

        if not found:
            # leave NaN
            continue

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Date YYYY-MM-DD (default: today)', default=datetime.utcnow().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    api_key = get_api_key()
    date_str = args.date

    predictions_file = DATA_DIR / f'predictions_{date_str}.csv'
    if not predictions_file.exists():
        print(f'Predictions file not found: {predictions_file}')
        sys.exit(1)

    print(f'Loading predictions: {predictions_file}')
    preds = pd.read_csv(predictions_file)

    print('Fetching odds from The Odds API...')
    try:
        odds_events = fetch_odds_for_date(api_key, date_str)
    except Exception as e:
        print('Failed to fetch odds:', e)
        sys.exit(1)

    if not odds_events:
        print('No odds events returned for date', date_str)
        sys.exit(0)

    print(f'Got {len(odds_events)} events from odds API; merging...')
    updated = match_and_merge(preds, odds_events)

    # Save updated CSV (create backup)
    backup = predictions_file.with_suffix('.csv.bak')
    if predictions_file.exists():
        predictions_file.replace(backup)
    updated.to_csv(predictions_file, index=False)

    print(f'Updated predictions saved to {predictions_file} (backup at {backup})')


if __name__ == '__main__':
    main()
