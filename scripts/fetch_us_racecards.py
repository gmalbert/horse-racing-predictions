"""
Fetch racecards from The Racing API for US tracks.
Saves raw JSON responses to data/raw/us_racecards_YYYY-MM-DD.json

Usage:
    python scripts/fetch_us_racecards.py --date 2025-12-31
    python scripts/fetch_us_racecards.py  # Fetches today
    python scripts/fetch_us_racecards.py --days 2  # Today and tomorrow
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv('RACING_API_USERNAME')
PASSWORD = os.getenv('RACING_API_PASSWORD')
BASE_URL = "https://api.theracingapi.com/v1"

if not USERNAME or not PASSWORD:
    print("ERROR: RACING_API_USERNAME / RACING_API_PASSWORD not set in .env")
    sys.exit(1)


def fetch_us_racecards(date_str: str) -> dict | None:
    """
    Fetch US racecards for *date_str* (YYYY-MM-DD) from The Racing API.

    Tries /racecards first (works on base plan), then /racecards/pro as
    a fallback for accounts with a higher-tier subscription.
    """
    endpoints_to_try = [
        # (endpoint, params)
        (f"{BASE_URL}/racecards",     {'region': 'US', 'date': date_str}),
        (f"{BASE_URL}/racecards/pro", {'region_codes': 'us', 'date': date_str}),
    ]

    for endpoint, params in endpoints_to_try:
        try:
            response = requests.get(
                endpoint,
                auth=(USERNAME, PASSWORD),
                params=params,
                timeout=30,
            )

            if response.status_code == 403:
                # This endpoint requires a higher plan — try next
                continue
            if response.status_code == 404:
                print(f"  No US racecards returned for {date_str} (404)")
                return None

            response.raise_for_status()
            data = response.json()

            # Normalise key — API may return 'racecards' or 'races'
            racecards = data.get('racecards') or data.get('races') or []
            print(f"  Fetched {len(racecards)} US races for {date_str} via {endpoint.split('/')[-1]}")
            return data

        except requests.exceptions.HTTPError as exc:
            print(f"  HTTP error ({endpoint.split('/')[-1]}) for {date_str}: {exc}")
            continue
        except requests.exceptions.RequestException as exc:
            print(f"  Request failed for {date_str}: {exc}")
            return None

    print(f"  All endpoints exhausted for {date_str} — check your API plan.")
    return None


def save_racecards(data: dict, date_str: str) -> Path:
    """Persist racecards JSON to data/raw/us_racecards_YYYY-MM-DD.json."""
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'us_racecards_{date_str}.json'

    with open(output_file, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)

    print(f"  Saved → {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Fetch US racecards from The Racing API')
    parser.add_argument('--date', type=str, help='Date (YYYY-MM-DD); defaults to today')
    parser.add_argument('--days', type=int, default=1,
                        help='Number of days to fetch starting from --date (default: 1)')
    args = parser.parse_args()

    start_date = (
        datetime.strptime(args.date, '%Y-%m-%d') if args.date
        else datetime.now()
    )

    dates = [
        (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(args.days)
    ]

    print(f"\n{'='*60}")
    print(f"Fetching US Racecards for {', '.join(dates)}")
    print(f"{'='*60}\n")

    success_count = 0
    for date_str in dates:
        print(f"[{date_str}]")
        data = fetch_us_racecards(date_str)
        if data:
            save_racecards(data, date_str)
            success_count += 1
        else:
            print(f"  FAILED: could not fetch US racecards for {date_str}")

    print(f"\n{'='*60}")
    print(f"Done — {success_count}/{len(dates)} dates fetched successfully")
    print(f"{'='*60}\n")

    if success_count == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
