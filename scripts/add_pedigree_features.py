#!/usr/bin/env python3
"""
Add Pedigree Features to Race Dataset

This script adds breeding-based features that are critical for predicting
performance of lightly-raced horses (< 3 runs) where form data is sparse.

NEW FEATURES (12 total):
1. sire_win_rate_v2 - Sire's progeny win rate (temporal)
2. sire_place_rate_v2 - Sire's progeny place rate (temporal)
3. sire_turf_win_rate - Sire win rate on turf specifically
4. sire_aw_win_rate - Sire win rate on all-weather specifically
5. sire_surface_pref - Surface preference: turf_wr - aw_wr
6. sire_avg_win_dist - Average winning distance of sire's winners
7. sire_sprint_pct - % of sire wins at sprint distances (<7f)
8. sire_stayer_pct - % of sire wins at staying distances (>10f)
9. sire_class_avg - Average class of sire's winners
10. dam_offspring_count - Number of previous offspring from dam
11. dam_offspring_win_rate - Win rate of dam's previous offspring
12. damsire_stamina_score - Damsire average winning distance / 10

TEMPORAL INTEGRITY:
- All stats calculated only from races BEFORE current race date
- Uses expanding windows grouped by sire/dam
- Prevents lookahead bias

Expected Impact: +0.02-0.03 AUC (particularly for lightly-raced horses)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_data():
    """Load race data with connections features"""
    data_dir = Path('data/processed')
    input_path = data_dir / 'race_scores_connections_v2.parquet'
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Loaded {len(df):,} horse-race records")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def calculate_sire_features(df):
    """Calculate sire-based features with temporal integrity"""
    print("\n" + "="*60)
    print("CALCULATING SIRE FEATURES")
    print("="*60)
    
    # Sort by sire and date for temporal calculations, then reset index
    df = df.sort_values(['sire', 'date']).reset_index(drop=True).copy()
    
    # Basic performance indicators
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    
    # Surface indicators
    df['is_turf'] = (df['surface'] == 'Turf').astype(int)
    df['is_aw'] = (df['surface'] == 'AW').astype(int)
    
    # Distance categories (use dist_f_clean for numeric comparisons)
    distance_col = 'dist_f_clean' if 'dist_f_clean' in df.columns else 'dist_f'
    df['is_sprint'] = (df[distance_col] < 7).astype(int)
    df['is_stayer'] = (df[distance_col] > 10).astype(int)
    
    print("\n1️⃣  Sire Win/Place Rates (temporal expanding window)")
    # Shift to exclude current race, then calculate expanding stats
    df['sire_progeny_count'] = df.groupby('sire').cumcount()
    df['sire_win_rate_v2'] = (
        df.groupby('sire')['won']
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )
    df['sire_place_rate_v2'] = (
        df.groupby('sire')['placed']
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )
    
    print(f"   ✓ Sire win rate: mean={df['sire_win_rate_v2'].mean():.3f}, std={df['sire_win_rate_v2'].std():.3f}")
    print(f"   ✓ Sire place rate: mean={df['sire_place_rate_v2'].mean():.3f}, std={df['sire_place_rate_v2'].std():.3f}")
    
    print("\n2️⃣  Sire Surface Preferences (Turf vs AW)")
    # Calculate surface-specific win rates
    df['sire_turf_wins'] = df['won'] * df['is_turf']
    df['sire_aw_wins'] = df['won'] * df['is_aw']
    
    df['sire_turf_win_rate'] = (
        df.groupby('sire')['sire_turf_wins']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('sire')['is_turf']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(0.0)
    )
    
    df['sire_aw_win_rate'] = (
        df.groupby('sire')['sire_aw_wins']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('sire')['is_aw']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(0.0)
    )
    
    # Preference: positive = turf specialist, negative = AW specialist
    df['sire_surface_pref'] = df['sire_turf_win_rate'] - df['sire_aw_win_rate']
    
    print(f"   ✓ Turf win rate: mean={df['sire_turf_win_rate'].mean():.3f}")
    print(f"   ✓ AW win rate: mean={df['sire_aw_win_rate'].mean():.3f}")
    print(f"   ✓ Surface preference: mean={df['sire_surface_pref'].mean():.3f}")
    
    print("\n3️⃣  Sire Distance Characteristics")
    # Average winning distance for sire's winners
    distance_col = 'dist_f_clean' if 'dist_f_clean' in df.columns else 'dist_f'
    df['sire_win_distance'] = df[distance_col] * df['won']
    df['sire_avg_win_dist'] = (
        df.groupby('sire')['sire_win_distance']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('sire')['won']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(6.0)  # Default to 6f (typical sprint)
    )
    
    # Sprint specialist percentage
    df['sire_sprint_wins'] = df['won'] * df['is_sprint']
    df['sire_sprint_pct'] = (
        df.groupby('sire')['sire_sprint_wins']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('sire')['won']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(0.5)  # Default 50% if no wins yet
    )
    
    # Staying specialist percentage
    df['sire_stayer_wins'] = df['won'] * df['is_stayer']
    df['sire_stayer_pct'] = (
        df.groupby('sire')['sire_stayer_wins']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('sire')['won']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(0.2)  # Default 20% if no wins yet
    )
    
    print(f"   ✓ Avg win distance: mean={df['sire_avg_win_dist'].mean():.1f}f")
    print(f"   ✓ Sprint %: mean={df['sire_sprint_pct'].mean():.3f}")
    print(f"   ✓ Stayer %: mean={df['sire_stayer_pct'].mean():.3f}")
    
    print("\n4️⃣  Sire Class Profile")
    # Average class of sire's winners
    class_col = 'class_num' if 'class_num' in df.columns else 'class_numeric'
    df['sire_win_class'] = df[class_col] * df['won']
    df['sire_class_avg'] = (
        df.groupby('sire')['sire_win_class']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('sire')['won']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(4.0)  # Default to Class 4 if no wins
    )
    
    print(f"   ✓ Sire class avg: mean={df['sire_class_avg'].mean():.2f}")
    
    # Cleanup temporary columns
    temp_cols = [
        'won', 'placed', 'is_turf', 'is_aw', 'is_sprint', 'is_stayer',
        'sire_turf_wins', 'sire_aw_wins', 'sire_win_distance',
        'sire_sprint_wins', 'sire_stayer_wins', 'sire_win_class',
        'sire_progeny_count'
    ]
    df.drop(columns=temp_cols, inplace=True, errors='ignore')
    
    return df

def calculate_dam_features(df):
    """Calculate dam-based features with temporal integrity"""
    print("\n" + "="*60)
    print("CALCULATING DAM FEATURES")
    print("="*60)
    
    # Sort by dam and date
    df = df.sort_values(['dam', 'date']).reset_index(drop=True).copy()
    
    # Performance indicators
    df['won'] = (df['pos_clean'] == 1).astype(int)
    
    print("\n5️⃣  Dam Offspring Performance")
    # Count previous offspring
    df['dam_offspring_count'] = df.groupby('dam').cumcount()
    
    # Dam offspring win rate (from previous offspring only)
    df['dam_offspring_win_rate'] = (
        df.groupby('dam')['won']
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )
    
    print(f"   ✓ Dam offspring count: mean={df['dam_offspring_count'].mean():.1f}")
    print(f"   ✓ Dam offspring win rate: mean={df['dam_offspring_win_rate'].mean():.3f}")
    
    # Cleanup
    df.drop(columns=['won'], inplace=True)
    
    return df

def calculate_damsire_features(df):
    """Calculate damsire-based features (stamina influence)"""
    print("\n" + "="*60)
    print("CALCULATING DAMSIRE FEATURES")
    print("="*60)
    
    # Sort by damsire and date
    df = df.sort_values(['damsire', 'date']).reset_index(drop=True).copy()
    
    df['won'] = (df['pos_clean'] == 1).astype(int)
    
    print("\n6️⃣  Damsire Stamina Index")
    # Average winning distance of damsire's progeny (stamina indicator)
    distance_col = 'dist_f_clean' if 'dist_f_clean' in df.columns else 'dist_f'
    df['damsire_win_distance'] = df[distance_col] * df['won']
    df['damsire_avg_win_dist'] = (
        df.groupby('damsire')['damsire_win_distance']
        .transform(lambda x: x.shift(1).expanding().sum())
        .div(
            df.groupby('damsire')['won']
            .transform(lambda x: x.shift(1).expanding().sum())
            .replace(0, np.nan)
        )
        .fillna(8.0)  # Default to 8f (middle distance)
    )
    
    # Stamina score: avg winning distance / 10 (normalized)
    df['damsire_stamina_score'] = df['damsire_avg_win_dist'] / 10.0
    
    print(f"   ✓ Damsire avg win dist: mean={df['damsire_avg_win_dist'].mean():.1f}f")
    print(f"   ✓ Damsire stamina score: mean={df['damsire_stamina_score'].mean():.3f}")
    
    # Cleanup
    df.drop(columns=['won', 'damsire_win_distance', 'damsire_avg_win_dist'], inplace=True)
    
    return df

def remove_old_pedigree_features(df):
    """Remove old placeholder pedigree features"""
    old_features = [
        'sire_win_rate', 'sire_place_rate', 'sire_surface_match',
        'sire_distance_match', 'sire_going_match', 'sire_class_match'
    ]
    
    existing_old = [col for col in old_features if col in df.columns]
    if existing_old:
        print(f"\n🗑️  Removing {len(existing_old)} old placeholder features:")
        print(f"   {', '.join(existing_old)}")
        df.drop(columns=existing_old, inplace=True)
    
    return df

def save_data(df):
    """Save enhanced dataset with pedigree features"""
    output_path = Path('data/processed/race_scores_pedigree.parquet')
    
    print("\n" + "="*60)
    print("SAVING ENHANCED DATASET")
    print("="*60)
    
    # Create backup of existing file if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.parquet.backup')
        print(f"📦 Creating backup: {backup_path}")
        output_path.rename(backup_path)
    
    print(f"💾 Saving to: {output_path}")
    df.to_parquet(output_path, index=False, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved {len(df):,} records ({file_size_mb:.1f} MB)")
    print(f"   ✓ Total columns: {len(df.columns)}")
    
    # List new pedigree features
    pedigree_features = [
        'sire_win_rate_v2', 'sire_place_rate_v2', 'sire_turf_win_rate',
        'sire_aw_win_rate', 'sire_surface_pref', 'sire_avg_win_dist',
        'sire_sprint_pct', 'sire_stayer_pct', 'sire_class_avg',
        'dam_offspring_count', 'dam_offspring_win_rate', 'damsire_stamina_score'
    ]
    
    print(f"\n✅ Added {len(pedigree_features)} pedigree features:")
    for feat in pedigree_features:
        if feat in df.columns:
            print(f"   ✓ {feat}")
    
    return output_path

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("PEDIGREE FEATURES GENERATION")
    print("="*60)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    # Remove old placeholder features
    df = remove_old_pedigree_features(df)
    
    # Calculate new pedigree features
    df = calculate_sire_features(df)
    df = calculate_dam_features(df)
    df = calculate_damsire_features(df)
    
    # Save enhanced dataset
    output_path = save_data(df)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("COMPLETION SUMMARY")
    print("="*60)
    print(f"✅ Pedigree features added successfully")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"📊 Output: {output_path}")
    print(f"📈 Next step: Run add_going_preference_features.py")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

