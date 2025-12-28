#!/usr/bin/env python3
"""
One-click prediction workflow.

After copying racecards JSON to data/raw/, run this script to:
1. Generate win/place/show predictions using ML models
2. Save predictions CSV
3. Optionally launch Streamlit UI

Usage:
    python run_predictions.py                    # Use latest racecard
    python run_predictions.py --date 2025-12-28  # Use specific date
    python run_predictions.py --ui               # Auto-launch Streamlit after predictions
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import glob

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
PREDICTIONS_DIR = BASE_DIR / "data" / "processed"
PYTHON = sys.executable


def find_latest_racecard():
    """Find the most recent racecards JSON file"""
    pattern = str(DATA_DIR / "racecards_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Extract dates and find newest
    dated_files = []
    for f in files:
        filename = Path(f).name
        try:
            date_str = filename.replace('racecards_', '').replace('.json', '')
            date = datetime.strptime(date_str, '%Y-%m-%d')
            dated_files.append((date, date_str, f))
        except:
            continue
    
    if not dated_files:
        return None
    
    dated_files.sort(reverse=True)
    return dated_files[0][1]  # Return date string


def run_predictions(date_str):
    """Run prediction script for given date"""
    print("="*60)
    print(f"RUNNING PREDICTIONS FOR {date_str}")
    print("="*60)
    
    racecard_file = DATA_DIR / f"racecards_{date_str}.json"
    
    if not racecard_file.exists():
        print(f"\n❌ ERROR: Racecard file not found: {racecard_file}")
        print("\nExpected location:")
        print(f"  {racecard_file}")
        print("\nMake sure you've copied the racecards JSON file to data/raw/")
        return False
    
    print(f"\n✓ Found racecard: {racecard_file}")
    
    # Run prediction script
    predict_script = BASE_DIR / "scripts" / "predict_todays_races.py"
    
    print(f"\nRunning predictions...")
    result = subprocess.run(
        [PYTHON, str(predict_script), "--date", date_str],
        cwd=BASE_DIR,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ Prediction script failed with exit code {result.returncode}")
        return False
    
    # Check output file was created
    predictions_file = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
    if not predictions_file.exists():
        print(f"\n❌ Predictions file was not created: {predictions_file}")
        return False
    
    print(f"\n✅ SUCCESS!")
    print(f"\nPredictions saved to:")
    print(f"  {predictions_file}")
    
    return True


def launch_streamlit():
    """Launch Streamlit UI"""
    print("\n" + "="*60)
    print("LAUNCHING STREAMLIT UI")
    print("="*60)
    
    ui_script = BASE_DIR / "predictions.py"
    
    print(f"\nStarting Streamlit...")
    print("(Press Ctrl+C to stop)")
    
    subprocess.run(
        ["streamlit", "run", str(ui_script)],
        cwd=BASE_DIR
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run horse racing predictions workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_predictions.py                    # Use latest racecard
  python run_predictions.py --date 2025-12-28  # Use specific date
  python run_predictions.py --ui               # Launch Streamlit after
  python run_predictions.py --date 2025-12-28 --ui
        """
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date of racecards (YYYY-MM-DD). If not provided, uses latest racecard file.'
    )
    
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Launch Streamlit UI after generating predictions'
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date_str = args.date
    else:
        print("Looking for latest racecard file...")
        date_str = find_latest_racecard()
        if not date_str:
            print("\n❌ No racecard files found in data/raw/")
            print("\nExpected format: racecards_YYYY-MM-DD.json")
            print(f"Location: {DATA_DIR}")
            print("\nPlease copy a racecard file or specify --date")
            return 1
        print(f"✓ Found latest racecard: {date_str}")
    
    # Validate date format
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"\n❌ Invalid date format: {date_str}")
        print("Expected format: YYYY-MM-DD (e.g., 2025-12-28)")
        return 1
    
    # Run predictions
    success = run_predictions(date_str)
    
    if not success:
        return 1
    
    # Launch UI if requested
    if args.ui:
        launch_streamlit()
    else:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\nTo view predictions in the UI:")
        print(f"  streamlit run predictions.py")
        print("\nOr run with --ui flag next time:")
        print(f"  python run_predictions.py --date {date_str} --ui")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
