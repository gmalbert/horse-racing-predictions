#!/usr/bin/env python3
"""
Build race_scores_engineered.parquet

Runs the full feature engineering pipeline (same functions used by phase3)
on the base parquet and saves the result to data/processed/.

This lets ensemble_model.py and backtest_walk_forward.py use all 75 engineered
features without having to re-train the model.

Usage:
  python scripts/build_engineered_dataset.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Import phase3 functions (safe: guarded by __name__ == '__main__')
from phase3_build_horse_model import load_data, engineer_all_features

OUTPUT = ROOT / "data" / "processed" / "race_scores_engineered.parquet"


def main():
    print("=" * 60)
    print("BUILDING ENGINEERED DATASET")
    print("=" * 60)

    df = load_data()
    df = engineer_all_features(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"\n[SAVED] {OUTPUT}")
    print(f"        {len(df):,} rows  |  {len(df.columns)} columns")
    print("\nDone. Re-run ensemble_model.py and backtest_walk_forward.py to use full 75-feature set.")


if __name__ == "__main__":
    main()
