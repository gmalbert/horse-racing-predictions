#!/usr/bin/env python3
"""
Race Result Capture — Auto-fetch final results and SP from The Racing API

Fetches race results for a given date and updates:
- Predictions CSV with actual results
- Betting history with outcomes
- Calculates CLV (Closing Line Value) metrics

Usage:
  python scripts/capture_race_results.py
  python scripts/capture_race_results.py --date 2026-06-08
  python scripts/capture_race_results.py --date 2026-06-08 --update-history

Requires:
  RACING_API_USERNAME and RACING_API_PASSWORD in .env
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

load_dotenv(ROOT / ".env")


def fetch_race_results(target_date: date) -> dict | None:
    """Fetch race results from The Racing API."""
    
    username = os.getenv("RACING_API_USERNAME")
    password = os.getenv("RACING_API_PASSWORD")
    
    if not username or not password:
        print("[!] Missing Racing API credentials in .env")
        print("    Required: RACING_API_USERNAME, RACING_API_PASSWORD")
        return None
    
    # API endpoint
    url = f"https://api.theracingapi.com/v1/results/{target_date}"
    
    try:
        print(f"[*] Fetching results from {url}")
        response = requests.get(url, auth=(username, password), timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Save raw response
        results_file = RAW_DIR / f"results_{target_date}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[✓] Saved raw results to {results_file.name}")
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"[!] API request failed: {e}")
        return None


def parse_results_to_dataframe(results_data: dict) -> pd.DataFrame:
    """Parse API results JSON to flat dataframe."""
    
    records = []
    
    races = results_data.get('races', [])
    
    for race in races:
        race_id = race.get('race_id', '')
        race_time = race.get('off_time', '')
        course = race.get('course', '')
        distance = race.get('distance', '')
        race_class = race.get('class', '')
        going = race.get('going', '')
        
        runners = race.get('runners', [])
        
        for runner in runners:
            horse_name = runner.get('name', '')
            finish_position = runner.get('position', 999)
            sp_decimal = runner.get('sp_decimal', None)
            
            records.append({
                'race_id': race_id,
                'race_time': race_time,
                'course': course,
                'distance': distance,
                'race_class': race_class,
                'going': going,
                'horse_name': horse_name,
                'finish_position': finish_position,
                'sp_decimal': sp_decimal
            })
    
    return pd.DataFrame(records)


def update_predictions_with_results(target_date: date, results_df: pd.DataFrame):
    """Merge results into predictions CSV."""
    
    pred_file = DATA_DIR / f"predictions_{target_date}.csv"
    
    if not pred_file.exists():
        print(f"[!] No predictions file found: {pred_file.name}")
        return
    
    # Load predictions
    predictions = pd.read_csv(pred_file)
    
    # Merge results
    predictions = predictions.merge(
        results_df[['race_id', 'horse_name', 'finish_position', 'sp_decimal']],
        on=['race_id', 'horse_name'],
        how='left',
        suffixes=('', '_result')
    )
    
    # Create actual_result column (1 = win, 0 = loss)
    predictions['actual_result'] = (predictions['finish_position'] == 1).astype(int)
    
    # Calculate CLV if SP available
    if 'sp_decimal' in predictions.columns and predictions['sp_decimal'].notna().any():
        predictions['clv'] = (
            predictions['sp_decimal'] / predictions['win_probability'].apply(lambda p: 1.0 / p if p > 0 else 999.0) - 1
        ) * 100
    
    # Save updated predictions
    predictions.to_csv(pred_file, index=False)
    print(f"[✓] Updated {pred_file.name} with results")
    
    # Summary
    won = predictions['actual_result'].sum()
    total = len(predictions[predictions['actual_result'].notna()])
    if total > 0:
        print(f"    Predicted winners: {won} / {total} ({won/total*100:.1f}%)")


def update_betting_history(target_date: date, results_df: pd.DataFrame):
    """Update betting history with race results."""
    
    history_file = DATA_DIR / "betting_history.csv"
    
    if not history_file.exists():
        print(f"[i] No betting history file found")
        return
    
    # Load history
    history = pd.read_csv(history_file)
    history['bet_date'] = pd.to_datetime(history['bet_date']).dt.date
    
    # Filter bets for target date
    todays_bets = history[history['bet_date'] == target_date].copy()
    
    if len(todays_bets) == 0:
        print(f"[i] No bets logged for {target_date}")
        return
    
    # Merge results
    todays_bets = todays_bets.merge(
        results_df[['race_id', 'horse_name', 'finish_position']],
        on=['race_id', 'horse_name'],
        how='left'
    )
    
    # Update results
    updated_count = 0
    
    for idx, row in todays_bets.iterrows():
        if pd.notna(row['finish_position']) and history.loc[idx, 'result'] == 'Pending':
            if row['finish_position'] == 1:
                history.loc[idx, 'result'] = 'Won'
                history.loc[idx, 'profit'] = row['stake'] * (row['market_odds'] - 1)
            else:
                history.loc[idx, 'result'] = 'Lost'
                history.loc[idx, 'profit'] = -row['stake']
            
            updated_count += 1
    
    if updated_count > 0:
        history.to_csv(history_file, index=False)
        print(f"[✓] Updated {updated_count} bets in betting history")
    else:
        print(f"[i] No pending bets to update")


def main():
    parser = argparse.ArgumentParser(description="Capture race results from The Racing API")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD, default: yesterday)")
    parser.add_argument("--update-history", action="store_true", help="Update betting history with results")
    
    args = parser.parse_args()
    
    # Determine target date (default: yesterday)
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[!] Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today() - timedelta(days=1)
    
    print(f"[*] Capturing race results for {target_date}")
    
    # Fetch results
    results_data = fetch_race_results(target_date)
    
    if not results_data:
        print(f"[!] Failed to fetch results for {target_date}")
        sys.exit(1)
    
    # Parse results
    results_df = parse_results_to_dataframe(results_data)
    
    if len(results_df) == 0:
        print(f"[!] No results found for {target_date}")
        sys.exit(1)
    
    print(f"[✓] Parsed {len(results_df)} runners from {results_df['race_id'].nunique()} races")
    
    # Update predictions
    update_predictions_with_results(target_date, results_df)
    
    # Update betting history if requested
    if args.update_history:
        update_betting_history(target_date, results_df)
    
    print(f"[✓] Result capture complete for {target_date}")


if __name__ == "__main__":
    main()
