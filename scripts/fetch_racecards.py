#!/usr/bin/env python3
"""Fetch racecards from The Racing API two days before race dates.

Usage:
    # Fetch racecards for a specific date (default: 2 days from now)
    python scripts/fetch_racecards.py --date 2026-01-20

    # Fetch racecards by course and date
    python scripts/fetch_racecards.py --date 2026-01-20 --course Ascot

    # Fetch a specific racecard by ID
    python scripts/fetch_racecards.py --racecard-id 12345

    # Fetch all racecards for today + 2 days
    python scripts/fetch_racecards.py

Outputs: data/raw/racecards_YYYY-MM-DD.json (one file per date)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "https://api.theracingapi.com/v1"
USERNAME = os.getenv("RACING_API_USERNAME")
PASSWORD = os.getenv("RACING_API_PASSWORD")
OUTPUT_DIR = Path("data/raw")


def fetch_racecards_by_date(date_str=None, region="GB", course=None):
    """
    Fetch racecards from the free API endpoint.
    Note: The free API returns upcoming racecards and doesn't accept date/region filters.
    Filtering is done client-side after fetching.
    
    Args:
        date_str (str): Date in YYYY-MM-DD format (used for client-side filtering)
        region (str): Region code (used for client-side filtering, default: GB)
        course (str): Optional course name filter (client-side)
    
    Returns:
        dict: API response with racecards
    """
    url = f"{BASE_URL}/racecards/free"
    
    print(f"Fetching racecards from free API...")
    
    try:
        response = requests.get(url, auth=(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        data = response.json()
        
        total_racecards = len(data.get('racecards', []))
        print(f"[OK] Retrieved {total_racecards} racecards from API")
        
        # Apply client-side filters if specified
        if data.get('racecards'):
            racecards = data['racecards']
            
            # Filter by date if specified
            if date_str:
                racecards = [rc for rc in racecards if rc.get('date') == date_str]
                print(f"[OK] Filtered to {len(racecards)} racecards for date {date_str}")
            
            # Filter by region if specified
            if region:
                racecards = [rc for rc in racecards if rc.get('region', '').upper() == region.upper()]
                print(f"[OK] Filtered to {len(racecards)} racecards for region {region}")
            
            # Filter by course if specified
            if course:
                racecards = [rc for rc in racecards if course.lower() in rc.get('course', '').lower()]
                print(f"[OK] Filtered to {len(racecards)} racecards for course '{course}'")
            
            data['racecards'] = racecards
            data['total'] = len(racecards)
        
        return data
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error: {e}")
        print(f"Response: {e.response.text if e.response else 'N/A'}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request error: {e}")
        return None


def fetch_racecard_by_id(racecard_id):
    """
    Fetch a specific racecard by ID.
    
    Args:
        racecard_id (str): Racecard ID
    
    Returns:
        dict: API response with racecard details
    """
    url = f"{BASE_URL}/racecards/free/{racecard_id}"
    
    print(f"Fetching racecard ID: {racecard_id}")
    
    try:
        response = requests.get(url, auth=(USERNAME, PASSWORD), timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"[OK] Racecard retrieved")
        return data
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error: {e}")
        print(f"Response: {e.response.text if e.response else 'N/A'}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request error: {e}")

def save_racecards(data, filename):
    """Save racecard data to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch racecards from The Racing API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Race date in YYYY-MM-DD format (default: today + 2 days)"
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=2,
        help="Number of days ahead to fetch (default: 2)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="GB",
        help="Region code (default: GB)"
    )
    parser.add_argument(
        "--course",
        type=str,
        help="Filter by course name (e.g., Ascot, Newmarket)"
    )
    parser.add_argument(
        "--racecard-id",
        type=str,
        help="Fetch a specific racecard by ID"
    )
    
    args = parser.parse_args()
    
    # Check credentials
    if not USERNAME or not PASSWORD:
        print("✗ Error: RACING_API_USERNAME and RACING_API_PASSWORD must be set in .env file")
        sys.exit(1)
    
    # Fetch by racecard ID
    if args.racecard_id:
        data = fetch_racecard_by_id(args.racecard_id)
        if data:
            filename = f"racecard_{args.racecard_id}.json"
            save_racecards(data, filename)
        else:
            sys.exit(1)
        return
    
    # Fetch by date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target_date = datetime.now() + timedelta(days=args.days_ahead)
    
    date_str = target_date.strftime("%Y-%m-%d")
    
    data = fetch_racecards_by_date(date_str, region=args.region, course=args.course)
    
    if data:
        filename = f"racecards_{date_str}"
        if args.course:
            filename += f"_{args.course.lower().replace(' ', '_')}"
        filename += ".json"
        save_racecards(data, filename)
    else:
        print("[ERROR] No data retrieved")
        sys.exit(1)


if __name__ == "__main__":
    main()
