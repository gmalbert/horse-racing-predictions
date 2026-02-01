#!/usr/bin/env python3
"""
Add Pedigree Features to Race Data
Implements CRITICAL_DATA_GAPS.md Section 1: Pedigree/Breeding Data
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_sire_lookup():
    """Load the sire lookup table."""
    lookup_path = Path('data/processed/lookups/sire_stats.csv')
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Sire lookup not found at {lookup_path}. "
            "Run scripts/build_sire_lookup.py first."
        )
    return pd.read_csv(lookup_path)

def add_pedigree_features(df, sire_lookup):
    """
    Add pedigree-based features for cold start horses.
    
    Features added:
    - sire_win_rate: Sire's progeny overall win rate
    - sire_place_rate: Sire's progeny place rate
    - sire_turf_rate: Sire win rate on turf
    - sire_aw_rate: Sire win rate on AW
    - sire_surface_match: Match current surface to sire preference
    - sire_distance_match: Match current distance band to sire strength
    - sire_going_match: Match current going to sire preference
    - sire_class_match: Match current class to sire typical class
    """
    print("\n" + "="*60)
    print("ADDING PEDIGREE FEATURES")
    print("="*60)
    
    df = df.copy()
    
    # Ensure distance band exists
    if 'distance_band' not in df.columns:
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
    
    # Standardize distance band to lowercase
    df['distance_band'] = df['distance_band'].str.lower()
    
    # Merge sire statistics
    print(f"\n1. Merging sire statistics for {df['sire_id'].nunique():,} unique sires...")
    
    sire_cols = [
        'sire_id', 'win_rate', 'place_rate', 
        'turf_win_rate', 'aw_win_rate', 'surface_preference',
        'sprint_win_rate', 'mile_win_rate', 'middle_win_rate', 'staying_win_rate',
        'firm_win_rate', 'good_win_rate', 'soft_win_rate', 'heavy_win_rate',
        'class1_win_rate', 'class2_win_rate', 'class3_win_rate', 'class4_win_rate',
        'avg_dist', 'avg_class', 'avg_or'
    ]
    
    df = df.merge(
        sire_lookup[sire_cols],
        on='sire_id',
        how='left',
        suffixes=('', '_sire')
    )
    
    # Rename to have clear 'sire_' prefix
    df = df.rename(columns={
        'win_rate': 'sire_win_rate',
        'place_rate': 'sire_place_rate'
    })
    
    # === SURFACE MATCH ===
    print("\n2. Calculating surface match features...")
    df['is_turf'] = (df['surface_clean'] == 'Turf').astype(int)
    
    df['sire_surface_match'] = np.where(
        df['is_turf'] == 1,
        df['turf_win_rate'].fillna(df['sire_win_rate']),
        df['aw_win_rate'].fillna(df['sire_win_rate'])
    )
    
    # === DISTANCE MATCH ===
    print("\n3. Calculating distance match features...")
    
    def get_distance_match(row):
        """Get sire win rate for current distance band."""
        dist_band = row['distance_band']
        col_name = f'{dist_band}_win_rate'
        return row.get(col_name, row.get('sire_win_rate', 0.10))
    
    df['sire_distance_match'] = df.apply(get_distance_match, axis=1)
    
    # === GOING MATCH ===
    print("\n4. Calculating going match features...")
    
    going_categories = {
        'firm': ['firm', 'good to firm', 'fast'],
        'good': ['good', 'standard'],
        'soft': ['soft', 'good to soft', 'yielding'],
        'heavy': ['heavy', 'soft to heavy', 'slow']
    }
    
    def categorize_going(going_str):
        if pd.isna(going_str):
            return 'good'
        going_lower = going_str.lower()
        for category, keywords in going_categories.items():
            if any(kw in going_lower for kw in keywords):
                return category
        return 'good'
    
    df['going_category'] = df['going'].apply(categorize_going)
    
    def get_going_match(row):
        """Get sire win rate for current going category."""
        going = row['going_category']
        col_name = f'{going}_win_rate'
        return row.get(col_name, row.get('sire_win_rate', 0.10))
    
    df['sire_going_match'] = df.apply(get_going_match, axis=1)
    
    # === CLASS MATCH ===
    print("\n5. Calculating class match features...")
    
    df['class_num'] = pd.to_numeric(
        df['class_clean'].str.extract(r'(\d+)', expand=False),
        errors='coerce'
    )
    
    def get_class_match(row):
        """Get sire win rate for current class."""
        class_num = row['class_num']
        if pd.notna(class_num) and class_num in [1, 2, 3, 4]:
            col_name = f'class{int(class_num)}_win_rate'
            return row.get(col_name, row.get('sire_win_rate', 0.10))
        return row.get('sire_win_rate', 0.10)
    
    df['sire_class_match'] = df.apply(get_class_match, axis=1)
    
    # === COLD START BOOST ===
    print("\n6. Creating cold start adjusted features...")
    
    # For horses with limited form, boost sire features
    if 'career_runs' in df.columns:
        df['is_cold_start'] = (df['career_runs'] < 3).astype(int)
    else:
        df['is_cold_start'] = 1  # Assume cold start if no career data
    
    # Adjusted career stats using sire data for cold start horses
    if 'career_win_rate' in df.columns:
        df['career_win_rate_adj'] = np.where(
            df['is_cold_start'] == 1,
            df['sire_win_rate'].fillna(0.10) * 0.7 + df['career_win_rate'].fillna(0) * 0.3,
            df['career_win_rate'].fillna(0.10)
        )
    
    if 'career_place_rate' in df.columns:
        df['career_place_rate_adj'] = np.where(
            df['is_cold_start'] == 1,
            df['sire_place_rate'].fillna(0.30) * 0.7 + df['career_place_rate'].fillna(0) * 0.3,
            df['career_place_rate'].fillna(0.30)
        )
    
    # Fill remaining NAs with reasonable defaults
    pedigree_features = [
        'sire_win_rate', 'sire_place_rate',
        'sire_surface_match', 'sire_distance_match', 
        'sire_going_match', 'sire_class_match'
    ]
    
    for feat in pedigree_features:
        if feat in df.columns:
            df[feat] = df[feat].fillna(0.10)  # Default to 10% if no sire data
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("PEDIGREE FEATURES SUMMARY")
    print("="*60)
    
    cold_start_count = df['is_cold_start'].sum()
    cold_start_pct = (cold_start_count / len(df)) * 100
    
    print(f"Cold start horses: {cold_start_count:,} / {len(df):,} ({cold_start_pct:.1f}%)")
    print(f"Horses with sire data: {df['sire_win_rate'].notna().sum():,}")
    print(f"\nNew features added: {len(pedigree_features) + 2}")
    print(f"  - sire_win_rate")
    print(f"  - sire_place_rate")
    print(f"  - sire_surface_match")
    print(f"  - sire_distance_match")
    print(f"  - sire_going_match")
    print(f"  - sire_class_match")
    print(f"  - career_win_rate_adj (cold start adjusted)")
    print(f"  - career_place_rate_adj (cold start adjusted)")
    
    return df

if __name__ == '__main__':
    # Load data
    print("Loading race data...")
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Load sire lookup
    print("Loading sire lookup...")
    sire_lookup = load_sire_lookup()
    
    # Add pedigree features
    df_with_pedigree = add_pedigree_features(df, sire_lookup)
    
    # Save
    output_path = Path('data/processed/race_scores_with_pedigree.parquet')
    print(f"\nSaving to {output_path}...")
    df_with_pedigree.to_parquet(output_path, index=False)
    
    print("\n✓ COMPLETE: Pedigree features added successfully")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(df_with_pedigree):,}")
    print(f"  Columns: {len(df_with_pedigree.columns)} (added {len(df_with_pedigree.columns) - len(df.columns)})")
