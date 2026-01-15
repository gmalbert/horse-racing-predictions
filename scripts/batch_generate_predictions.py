#!/usr/bin/env python3
"""
Batch Generate Predictions for All Racecards

Automatically scans data/raw/ for racecard files, identifies which dates
need predictions, and runs the prediction model for all missing dates.

Usage:
  python scripts/batch_generate_predictions.py              # Process all racecards
  python scripts/batch_generate_predictions.py --force      # Regenerate all (skip existing)
  python scripts/batch_generate_predictions.py --dry-run    # Show what would be processed
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "predict_todays_races.py"


def find_racecards():
    """Find all racecard files in data/raw/ and extract dates"""
    pattern = re.compile(r"racecards_(\d{4}-\d{2}-\d{2})\.json")
    racecards = []
    
    for file in DATA_RAW.glob("racecards_*.json"):
        match = pattern.match(file.name)
        if match:
            date_str = match.group(1)
            racecards.append((date_str, file))
    
    # Sort by date
    racecards.sort(key=lambda x: x[0])
    return racecards


def check_predictions_exist(date_str):
    """Check if predictions already exist for a given date"""
    pred_file = DATA_PROCESSED / f"predictions_{date_str}.csv"
    return pred_file.exists()


def run_prediction(date_str, dry_run=False):
    """Run prediction script for a specific date"""
    if dry_run:
        print(f"  [DRY-RUN] Would run: python scripts/predict_todays_races.py --date {date_str}")
        return True
    
    print(f"\n{'='*70}")
    print(f"  Running predictions for {date_str}...")
    print(f"{'='*70}")
    
    try:
        # Run the prediction script
        result = subprocess.run(
            [sys.executable, str(PREDICT_SCRIPT), "--date", date_str],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Success: Predictions generated for {date_str}")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: Prediction took too long for {date_str}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate predictions for all racecards"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate predictions even if they already exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running predictions"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Only process racecards from this date onwards (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Only process racecards up to this date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  BATCH PREDICTION GENERATOR")
    print("="*70)
    
    # Find all racecards
    racecards = find_racecards()
    
    if not racecards:
        print("\n[!] No racecards found in data/raw/")
        print("    Expected format: racecards_YYYY-MM-DD.json")
        return
    
    print(f"\nFound {len(racecards)} racecard file(s)")
    
    # Filter by date range if specified
    if args.start_date:
        racecards = [(d, f) for d, f in racecards if d >= args.start_date]
    if args.end_date:
        racecards = [(d, f) for d, f in racecards if d <= args.end_date]
    
    # Identify which ones need processing
    to_process = []
    already_exists = []
    
    for date_str, file_path in racecards:
        if check_predictions_exist(date_str):
            if args.force:
                to_process.append(date_str)
            else:
                already_exists.append(date_str)
        else:
            to_process.append(date_str)
    
    # Summary
    print("\n" + "-"*70)
    print("SUMMARY:")
    print("-"*70)
    print(f"  Total racecards:           {len(racecards)}")
    print(f"  Already have predictions:  {len(already_exists)}")
    print(f"  Need predictions:          {len(to_process)}")
    
    if args.force and already_exists:
        print(f"  Will regenerate (--force): {len(already_exists)}")
    
    if not to_process:
        print("\n✓ All racecards already have predictions!")
        if not args.force:
            print("  Use --force to regenerate existing predictions")
        return
    
    # Show what will be processed
    print(f"\n{'Date':<15} {'Status':<20} {'Racecard File'}")
    print("-"*70)
    for date_str, _ in racecards:
        if date_str in to_process:
            status = "REGENERATE" if check_predictions_exist(date_str) else "NEW"
            print(f"{date_str:<15} {status:<20} racecards_{date_str}.json")
        elif date_str in already_exists:
            print(f"{date_str:<15} {'SKIP (exists)':<20} racecards_{date_str}.json")
    
    if args.dry_run:
        print("\n[DRY-RUN] Exiting without running predictions")
        return
    
    # Confirm
    if len(to_process) > 5 and not args.force:
        response = input(f"\n⚠ Process {len(to_process)} dates? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Process each date
    print("\n" + "="*70)
    print("  PROCESSING PREDICTIONS")
    print("="*70)
    
    success_count = 0
    failed_dates = []
    
    for i, date_str in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] Processing {date_str}...")
        
        if run_prediction(date_str, dry_run=args.dry_run):
            success_count += 1
        else:
            failed_dates.append(date_str)
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  ✓ Successful: {success_count}/{len(to_process)}")
    if failed_dates:
        print(f"  ✗ Failed:     {len(failed_dates)}")
        print(f"\nFailed dates:")
        for date_str in failed_dates:
            print(f"  - {date_str}")
    else:
        print("\n🎉 All predictions generated successfully!")
    
    print()


if __name__ == "__main__":
    main()
