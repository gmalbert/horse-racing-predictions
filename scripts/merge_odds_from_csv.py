#!/usr/bin/env python
"""
Merge bookmaker odds from CSV into predictions CSV.

Uses race_id + horse_id as primary join keys. Falls back to fuzzy name matching
if necessary.

Expected odds CSV format:
  race_id,horse_id,bookmaker_odds,bookmaker,odds_timestamp
  908701,8556534,3.50,Bet365,2025-12-22T14:30:00Z

Usage:
  python scripts/merge_odds_from_csv.py --date 2025-12-22 --odds-csv data/raw/odds_2025-12-22.csv
"""

import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


def load_racecards(date: str) -> dict:
    """Load racecards JSON for reference (contains race_id mapping)."""
    racecard_path = Path(f"data/raw/racecards_{date}.json")
    if not racecard_path.exists():
        print(f"Warning: Racecards JSON not found at {racecard_path}")
        return {}
    
    with open(racecard_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    """Normalize horse/jockey name for fuzzy matching."""
    import re
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)     # Collapse whitespace
    return name


def merge_odds_into_predictions(
    predictions_path: Path,
    odds_csv_path: Path,
    racecards: dict,
    output_path: Path = None
):
    """
    Merge odds CSV into predictions CSV using race_id+horse_id.
    
    Args:
        predictions_path: Path to predictions_YYYY-MM-DD.csv
        odds_csv_path: Path to odds CSV with columns: race_id, horse_id, bookmaker_odds, bookmaker, odds_timestamp
        racecards: Racecards JSON dict (for fallback matching if needed)
        output_path: Optional output path (defaults to predictions_path)
    """
    if output_path is None:
        output_path = predictions_path
    
    # Load predictions
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    df_pred = pd.read_csv(predictions_path)
    print(f"Loaded {len(df_pred)} predictions from {predictions_path}")
    
    # Load odds CSV
    if not odds_csv_path.exists():
        raise FileNotFoundError(f"Odds CSV not found: {odds_csv_path}")
    
    df_odds = pd.read_csv(odds_csv_path)
    print(f"Loaded {len(df_odds)} odds records from {odds_csv_path}")
    
    # Ensure required columns in odds CSV
    required_cols = ['race_id', 'horse_id', 'bookmaker_odds', 'bookmaker']
    missing = [c for c in required_cols if c not in df_odds.columns]
    if missing:
        raise ValueError(f"Odds CSV missing required columns: {missing}")
    
    # Optional: odds_timestamp column
    if 'odds_timestamp' not in df_odds.columns:
        df_odds['odds_timestamp'] = datetime.utcnow().isoformat() + 'Z'
    
    # Convert types for merge
    df_pred['race_id'] = df_pred['race_id'].astype(int)
    df_odds['race_id'] = df_odds['race_id'].astype(int)
    df_pred['horse_id'] = df_pred['horse_id'].astype(int)
    df_odds['horse_id'] = df_odds['horse_id'].astype(int)
    
    # Merge on race_id + horse_id
    df_merged = df_pred.merge(
        df_odds[['race_id', 'horse_id', 'bookmaker_odds', 'bookmaker', 'odds_timestamp']],
        on=['race_id', 'horse_id'],
        how='left',
        suffixes=('', '_new')
    )
    
    # Update existing odds columns or create them
    if 'bookmaker_odds' in df_merged.columns and 'bookmaker_odds_new' in df_merged.columns:
        # Prefer new odds if present, else keep existing
        df_merged['bookmaker_odds'] = df_merged['bookmaker_odds_new'].combine_first(df_merged['bookmaker_odds'])
        df_merged.drop(columns=['bookmaker_odds_new'], inplace=True)
    elif 'bookmaker_odds_new' in df_merged.columns:
        df_merged.rename(columns={'bookmaker_odds_new': 'bookmaker_odds'}, inplace=True)
    
    if 'bookmaker' in df_merged.columns and 'bookmaker_new' in df_merged.columns:
        df_merged['bookmaker'] = df_merged['bookmaker_new'].combine_first(df_merged['bookmaker'])
        df_merged.drop(columns=['bookmaker_new'], inplace=True)
    elif 'bookmaker_new' in df_merged.columns:
        df_merged.rename(columns={'bookmaker_new': 'bookmaker'}, inplace=True)
    
    if 'odds_timestamp' in df_merged.columns and 'odds_timestamp_new' in df_merged.columns:
        df_merged['odds_timestamp'] = df_merged['odds_timestamp_new'].combine_first(df_merged['odds_timestamp'])
        df_merged.drop(columns=['odds_timestamp_new'], inplace=True)
    elif 'odds_timestamp_new' in df_merged.columns:
        df_merged.rename(columns={'odds_timestamp_new': 'odds_timestamp'}, inplace=True)
    
    # Count merges
    matched = df_merged['bookmaker_odds'].notna().sum()
    total = len(df_merged)
    print(f"\nMerge summary:")
    print(f"  Total predictions: {total}")
    print(f"  Matched with odds: {matched} ({100*matched/total:.1f}%)")
    print(f"  Unmatched: {total - matched}")
    
    # Save updated predictions
    df_merged.to_csv(output_path, index=False)
    print(f"\nSaved updated predictions to {output_path}")
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge bookmaker odds from CSV into predictions CSV.'
    )
    parser.add_argument(
        '--date',
        required=True,
        help='Date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--odds-csv',
        required=True,
        help='Path to odds CSV file (columns: race_id, horse_id, bookmaker_odds, bookmaker, odds_timestamp)'
    )
    parser.add_argument(
        '--output',
        help='Optional output path for updated predictions CSV (defaults to overwriting predictions file)'
    )
    
    args = parser.parse_args()
    
    # Paths
    predictions_path = Path(f"data/processed/predictions_{args.date}.csv")
    odds_csv_path = Path(args.odds_csv)
    output_path = Path(args.output) if args.output else predictions_path
    
    # Load racecards for reference
    racecards = load_racecards(args.date)
    
    # Merge
    merge_odds_into_predictions(
        predictions_path=predictions_path,
        odds_csv_path=odds_csv_path,
        racecards=racecards,
        output_path=output_path
    )


if __name__ == '__main__':
    main()
