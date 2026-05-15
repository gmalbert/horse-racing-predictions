"""
Batch-generate US predictions for every us_racecards_*.json found in data/raw/
that does NOT already have a corresponding us_predictions_*.csv in data/processed/.

Usage:
    python scripts/batch_generate_us_predictions.py
    python scripts/batch_generate_us_predictions.py --force   # regenerate all
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / 'data' / 'raw'
PROC_DIR = BASE_DIR / 'data' / 'processed'


def main():
    parser = argparse.ArgumentParser(description='Batch-generate US race predictions')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate predictions even if output already exists')
    args = parser.parse_args()

    racecard_files = sorted(RAW_DIR.glob('us_racecards_*.json'))
    if not racecard_files:
        print("No US racecard files found in data/raw/")
        sys.exit(0)

    print(f"Found {len(racecard_files)} US racecard file(s)\n")

    generated = 0
    skipped   = 0
    failed    = 0

    for rc_file in racecard_files:
        # Extract date from filename: us_racecards_YYYY-MM-DD.json
        date_str = rc_file.stem.replace('us_racecards_', '')
        output   = PROC_DIR / f'us_predictions_{date_str}.csv'

        if output.exists() and not args.force:
            print(f"[SKIP]  {date_str}  (predictions already exist)")
            skipped += 1
            continue

        print(f"[GEN]   {date_str} …", end=' ', flush=True)
        result = subprocess.run(
            [sys.executable, 'scripts/predict_us_races.py', '--date', date_str],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("OK")
            generated += 1
        else:
            print("FAILED")
            print(result.stderr[-500:])
            failed += 1

    print(f"\n{'='*50}")
    print(f"Generated: {generated}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"{'='*50}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
