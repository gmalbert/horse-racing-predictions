#!/usr/bin/env python3
"""
Add Enhanced Form Features (V2)

More sophisticated form analysis including:
- Weighted position average (recent races count more)
- Position relative to field size
- Form consistency (std dev of positions)
- Form trend (improving/declining)
- Class-adjusted form

Usage:
  python scripts/add_enhanced_form_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path('data/processed')
INPUT_FILE = DATA_DIR / 'race_scores_with_all_features_no_leakage.parquet'
OUTPUT_FILE = DATA_DIR / 'race_scores_enhanced_form.parquet'


def weighted_pos_avg(positions, weights=[0.5, 0.3, 0.2]):
    """
    Weight recent races more heavily.
    Most recent = 0.5, second = 0.3, third = 0.2
    """
    if len(positions) == 0:
        return np.nan
    positions = positions[:len(weights)]  # Take only as many as we have weights
    if len(positions) < len(weights):
        weights = weights[:len(positions)]
        weights = np.array(weights) / sum(weights)  # Renormalize
    return np.average(positions, weights=weights)


def engineer_enhanced_form_features(df):
    """
    More sophisticated form analysis.
    """
    print("\n" + "="*70)
    print("ENHANCED FORM FEATURES")
    print("="*70)
    
    # Ensure sorted by horse and date
    df = df.sort_values(['horse', 'date_dt']).copy()
    
    # Ensure we have pos_clean
    if 'pos_clean' not in df.columns:
        if 'pos' in df.columns:
            df['pos_clean'] = pd.to_numeric(df['pos'], errors='coerce')
        else:
            print("ERROR: No position column found")
            return df
    
    print("\n1. Weighted Position Average (recent = more weight)")
    # Weighted position average - use .shift(1) to prevent leakage
    df['weighted_pos_avg'] = df.groupby('horse')['pos_clean'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).apply(weighted_pos_avg, raw=False)
    )
    
    print("2. Position Percentage (relative to field size)")
    # First calculate avg_last_3_pos if it doesn't exist
    if 'avg_last_3_pos' not in df.columns:
        df['avg_last_3_pos'] = df.groupby('horse')['pos_clean'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
    
    # Position relative to field size (1st of 5 vs 1st of 15)
    # Need avg field size from last 3 races
    df['avg_field_size_last_3'] = df.groupby('horse')['field_size'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    df['pos_pct_last_3'] = df['avg_last_3_pos'] / df['avg_field_size_last_3'].clip(lower=1)
    df['pos_pct_last_3'] = df['pos_pct_last_3'].fillna(0.5)  # Default to mid-pack
    
    print("3. Form Consistency (std dev of last 5 positions)")
    # Consistency (std dev of last 5 positions) - lower = more consistent
    df['form_consistency'] = df.groupby('horse')['pos_clean'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )
    df['form_consistency'] = df['form_consistency'].fillna(0)
    
    print("4. Form Trend (improving/declining)")
    # Improvement trend (negative slope = improving positions)
    def calc_trend(positions):
        """Calculate linear trend of positions. Negative = improving."""
        if len(positions) < 2:
            return 0
        try:
            slope = np.polyfit(range(len(positions)), positions, 1)[0]
            return -slope  # Negative slope means improving (smaller positions)
        except:
            return 0
    
    df['form_trend'] = df.groupby('horse')['pos_clean'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False)
    )
    df['form_trend'] = df['form_trend'].fillna(0)
    
    print("5. Class-Adjusted Form")
    # Performance at this class level specifically
    df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # For each horse-class combination, calculate historical win rate
    # Use expanding window to avoid leakage
    df['form_at_class'] = df.groupby(['horse', 'class_num'])['won'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df['form_at_class'] = df['form_at_class'].fillna(0)
    
    # Count of runs at this class
    df['runs_at_class'] = df.groupby(['horse', 'class_num']).cumcount()
    
    print("\n✓ Enhanced form features created:")
    print("  - weighted_pos_avg: Recent positions weighted more heavily")
    print("  - pos_pct_last_3: Position as % of field size")
    print("  - form_consistency: Std dev of last 5 positions")
    print("  - form_trend: Slope of recent positions (positive = improving)")
    print("  - form_at_class: Win rate at this specific class")
    print("  - runs_at_class: Experience at this class level")
    
    return df


def main():
    print("="*70)
    print("ADD ENHANCED FORM FEATURES")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Loaded: {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    
    # Ensure date_dt exists
    if 'date_dt' not in df.columns and 'date' in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'])
    
    # Add features
    df = engineer_enhanced_form_features(df)
    
    # Save
    print(f"\n{'='*70}")
    print(f"SAVING")
    print(f"{'='*70}")
    print(f"Output: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved: {len(df):,} records x {len(df.columns)} columns")
    
    # Summary
    new_features = ['weighted_pos_avg', 'pos_pct_last_3', 'form_consistency', 
                    'form_trend', 'form_at_class', 'runs_at_class']
    
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS")
    print(f"{'='*70}")
    for feat in new_features:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            coverage = non_null / len(df) * 100
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            print(f"\n{feat}:")
            print(f"  Coverage: {coverage:.1f}% ({non_null:,} records)")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std:  {std_val:.4f}")
            print(f"  Min:  {df[feat].min():.4f}")
            print(f"  Max:  {df[feat].max():.4f}")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
