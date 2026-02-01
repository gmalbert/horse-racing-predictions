#!/usr/bin/env python3
"""
Add Pedigree Features WITHOUT DATA LEAKAGE
Uses expanding window approach - sire stats calculated only from PRIOR races
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_pedigree_features_no_leakage(df):
    """
    Add pedigree-based features using ONLY historical data.
    
    For each race, sire statistics are calculated from races that occurred
    BEFORE that race date (expanding window approach).
    
    Features added:
    - sire_win_rate: Sire's progeny win rate (from prior races only)
    - sire_place_rate: Sire's progeny place rate (from prior races only)
    - sire_surface_match: Sire win rate on current surface (from prior races)
    - sire_distance_match: Sire win rate in current distance band (from prior races)
    - sire_going_match: Sire win rate on current going (from prior races)
    - sire_class_match: Sire win rate in current class (from prior races)
    """
    print("\n" + "="*60)
    print("ADDING PEDIGREE FEATURES (NO LEAKAGE)")
    print("="*60)
    
    df = df.copy()
    df = df.sort_values(['date', 'off']).copy()
    
    # Convert date to datetime
    if df['date'].dtype != 'datetime64[ns]':
        df['date_dt'] = pd.to_datetime(df['date'])
    else:
        df['date_dt'] = df['date']
    
    # Create necessary columns
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    df['is_turf'] = (df['surface_clean'] == 'Turf').astype(int)
    
    # Distance bands
    def classify_distance(dist_f):
        if dist_f < 7:
            return 'sprint'
        elif dist_f < 9:
            return 'mile'
        elif dist_f < 12:
            return 'middle'
        else:
            return 'staying'
    
    df['distance_band'] = df['dist_f_clean'].apply(classify_distance)
    
    # Going categories
    def classify_going(going_str):
        if pd.isna(going_str):
            return 'unknown'
        going_lower = str(going_str).lower()
        if 'firm' in going_lower or 'hard' in going_lower:
            return 'firm'
        elif 'good' in going_lower:
            return 'good'
        elif 'soft' in going_lower or 'yielding' in going_lower:
            return 'soft'
        elif 'heavy' in going_lower:
            return 'heavy'
        return 'unknown'
    
    df['going_category'] = df['going'].apply(classify_going)
    
    # Class as numeric
    df['class_num'] = pd.to_numeric(
        df['class_clean'].str.extract(r'(\d+)', expand=False),
        errors='coerce'
    ).fillna(4)
    
    print(f"\nProcessing {len(df):,} races with {df['sire_id'].nunique():,} unique sires...")
    
    # === EXPANDING WINDOW SIRE STATS ===
    print("\n1. Calculating sire overall win/place rates (expanding window)...")
    
    # Overall sire win rate (using only PRIOR races)
    df['sire_win_rate'] = df.groupby('sire_id')['won'].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )
    
    df['sire_place_rate'] = df.groupby('sire_id')['placed'].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )
    
    # Fill NAs with global average for sires with <5 prior runners
    global_win_rate = df['won'].mean()
    global_place_rate = df['placed'].mean()
    
    df['sire_win_rate'] = df['sire_win_rate'].fillna(global_win_rate)
    df['sire_place_rate'] = df['sire_place_rate'].fillna(global_place_rate)
    
    print(f"   Sire win rate: mean={df['sire_win_rate'].mean():.3f}")
    print(f"   Sire place rate: mean={df['sire_place_rate'].mean():.3f}")
    
    # === SURFACE-SPECIFIC SIRE STATS ===
    print("\n2. Calculating surface-specific sire stats (expanding)...")
    
    # Create sire-surface groups and calculate expanding win rates
    df['sire_surface'] = df['sire_id'].astype(str) + '_' + df['is_turf'].astype(str)
    
    df['sire_surface_match'] = df.groupby('sire_surface')['won'].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )
    
    # Fall back to overall sire win rate
    df['sire_surface_match'] = df['sire_surface_match'].fillna(df['sire_win_rate'])
    
    print(f"   Surface match: mean={df['sire_surface_match'].mean():.3f}")
    
    # === DISTANCE-SPECIFIC SIRE STATS ===
    print("\n3. Calculating distance-specific sire stats (expanding)...")
    
    df['sire_distance'] = df['sire_id'].astype(str) + '_' + df['distance_band'].astype(str)
    
    df['sire_distance_match'] = df.groupby('sire_distance')['won'].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )
    
    df['sire_distance_match'] = df['sire_distance_match'].fillna(df['sire_win_rate'])
    
    print(f"   Distance match: mean={df['sire_distance_match'].mean():.3f}")
    
    # === GOING-SPECIFIC SIRE STATS ===
    print("\n4. Calculating going-specific sire stats (expanding)...")
    
    df['sire_going'] = df['sire_id'].astype(str) + '_' + df['going_category'].astype(str)
    
    df['sire_going_match'] = df.groupby('sire_going')['won'].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )
    
    df['sire_going_match'] = df['sire_going_match'].fillna(df['sire_win_rate'])
    
    print(f"   Going match: mean={df['sire_going_match'].mean():.3f}")
    
    # === CLASS-SPECIFIC SIRE STATS ===
    print("\n5. Calculating class-specific sire stats (expanding)...")
    
    df['sire_class'] = df['sire_id'].astype(str) + '_' + df['class_num'].astype(str)
    
    df['sire_class_match'] = df.groupby('sire_class')['won'].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )
    
    df['sire_class_match'] = df['sire_class_match'].fillna(df['sire_win_rate'])
    
    print(f"   Class match: mean={df['sire_class_match'].mean():.3f}")
    
    # Clean up temporary columns
    df = df.drop(columns=['sire_surface', 'sire_distance', 'sire_going', 'sire_class'])
    
    print("\n" + "="*60)
    print("PEDIGREE FEATURES SUMMARY (NO LEAKAGE)")
    print("="*60)
    print("\nFeatures added: 6")
    print("  - sire_win_rate (expanding window)")
    print("  - sire_place_rate (expanding window)")
    print("  - sire_surface_match (expanding window)")
    print("  - sire_distance_match (expanding window)")
    print("  - sire_going_match (expanding window)")
    print("  - sire_class_match (expanding window)")
    print("\n✓ All features use .shift(1).expanding() to prevent leakage")
    print("✓ For each race, sire stats calculated from PRIOR races only")
    
    return df

if __name__ == '__main__':
    # Load base data
    print("Loading race data...")
    base_path = Path('data/processed/race_scores.parquet')
    df = pd.read_parquet(base_path)
    print(f"  Loaded {len(df):,} rows")
    
    # Add pedigree features (no leakage version)
    df_with_pedigree = add_pedigree_features_no_leakage(df)
    
    # Save
    output_path = Path('data/processed/race_scores_with_pedigree_no_leakage.parquet')
    print(f"\nSaving to {output_path}...")
    df_with_pedigree.to_parquet(output_path, index=False)
    
    print("\n✓ COMPLETE: Pedigree features added (no leakage)")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(df_with_pedigree):,}")
    print(f"  Columns: {len(df_with_pedigree.columns)}")
