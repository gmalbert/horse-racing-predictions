#!/usr/bin/env python3
"""
Build Sire Performance Lookup Tables
Implements pedigree features from CRITICAL_DATA_GAPS.md Section 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def build_sire_lookup(df, min_runners=20):
    """
    Build comprehensive sire statistics lookup table.
    
    Args:
        df: Historical race data with pedigree columns
        min_runners: Minimum progeny runners to include sire in lookup
    
    Returns:
        DataFrame: Sire statistics lookup table
    """
    print("\n" + "="*60)
    print("BUILDING SIRE PERFORMANCE LOOKUP")
    print("="*60)
    
    # Ensure outcome columns exist
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    df['is_turf'] = (df['surface_clean'] == 'Turf').astype(int)
    
    # Convert OR to numeric
    df['or_numeric'] = pd.to_numeric(df['or'], errors='coerce')
    
    print(f"\nAnalyzing {len(df):,} races with {df['sire'].nunique():,} unique sires")
    
    # === OVERALL SIRE STATS ===
    print("\n1. Calculating overall sire statistics...")
    sire_stats = df.groupby('sire_id').agg({
        'sire': 'first',
        'won': ['sum', 'count'],
        'placed': 'sum',
        'is_turf': 'mean',
        'dist_f_clean': 'mean',
        'class_clean': lambda x: pd.to_numeric(x.str.extract(r'(\d+)', expand=False), errors='coerce').mean(),
        'or_numeric': 'mean'
    }).reset_index()
    
    sire_stats.columns = [
        'sire_id', 'sire', 'wins', 'runs', 'places',
        'turf_pct', 'avg_dist', 'avg_class', 'avg_or'
    ]
    
    sire_stats['win_rate'] = sire_stats['wins'] / sire_stats['runs']
    sire_stats['place_rate'] = sire_stats['places'] / sire_stats['runs']
    
    # Filter by minimum runners
    sire_stats = sire_stats[sire_stats['runs'] >= min_runners].copy()
    print(f"   Found {len(sire_stats):,} sires with {min_runners}+ runners")
    
    # === SURFACE SPECIALIZATION ===
    print("\n2. Calculating surface preferences...")
    
    # Turf performance
    turf_df = df[df['is_turf'] == 1]
    sire_turf = turf_df.groupby('sire_id').agg({
        'won': ['sum', 'count']
    }).reset_index()
    sire_turf.columns = ['sire_id', 'turf_wins', 'turf_runs']
    sire_turf['turf_win_rate'] = sire_turf['turf_wins'] / sire_turf['turf_runs']
    
    # AW performance
    aw_df = df[df['is_turf'] == 0]
    sire_aw = aw_df.groupby('sire_id').agg({
        'won': ['sum', 'count']
    }).reset_index()
    sire_aw.columns = ['sire_id', 'aw_wins', 'aw_runs']
    sire_aw['aw_win_rate'] = sire_aw['aw_wins'] / sire_aw['aw_runs']
    
    sire_stats = sire_stats.merge(
        sire_turf[['sire_id', 'turf_win_rate']], 
        on='sire_id', how='left'
    )
    sire_stats = sire_stats.merge(
        sire_aw[['sire_id', 'aw_win_rate']], 
        on='sire_id', how='left'
    )
    
    sire_stats['surface_preference'] = (
        sire_stats['turf_win_rate'].fillna(0) - 
        sire_stats['aw_win_rate'].fillna(0)
    )
    
    # === DISTANCE BANDS ===
    print("\n3. Calculating distance band performance...")
    
    distance_bands = {
        'sprint': (0, 7),
        'mile': (7, 9), 
        'middle': (9, 12),
        'staying': (12, 99)
    }
    
    for band_name, (min_f, max_f) in distance_bands.items():
        band_df = df[df['dist_f_clean'].between(min_f, max_f)]
        
        sire_band = band_df.groupby('sire_id').agg({
            'won': ['sum', 'count']
        }).reset_index()
        sire_band.columns = ['sire_id', f'{band_name}_wins', f'{band_name}_runs']
        sire_band[f'{band_name}_win_rate'] = (
            sire_band[f'{band_name}_wins'] / sire_band[f'{band_name}_runs']
        )
        
        sire_stats = sire_stats.merge(
            sire_band[['sire_id', f'{band_name}_win_rate']], 
            on='sire_id', how='left'
        )
    
    # Best distance band for each sire
    dist_cols = [f'{b}_win_rate' for b in distance_bands.keys()]
    sire_stats['best_distance_band'] = sire_stats[dist_cols].fillna(0).idxmax(axis=1)
    sire_stats['best_distance_band'] = sire_stats['best_distance_band'].str.replace('_win_rate', '')
    
    # === GOING PREFERENCES ===
    print("\n4. Calculating going preferences...")
    
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
    
    for going_type in going_categories.keys():
        going_df = df[df['going_category'] == going_type]
        
        sire_going = going_df.groupby('sire_id').agg({
            'won': ['sum', 'count']
        }).reset_index()
        sire_going.columns = ['sire_id', f'{going_type}_wins', f'{going_type}_runs']
        sire_going[f'{going_type}_win_rate'] = (
            sire_going[f'{going_type}_wins'] / sire_going[f'{going_type}_runs']
        )
        
        sire_stats = sire_stats.merge(
            sire_going[['sire_id', f'{going_type}_win_rate']], 
            on='sire_id', how='left'
        )
    
    # Fill NAs with overall win rate
    for col in sire_stats.columns:
        if col.endswith('_win_rate') and col != 'win_rate':
            sire_stats[col] = sire_stats[col].fillna(sire_stats['win_rate'])
    
    # === CLASS PERFORMANCE ===
    print("\n5. Calculating class-level performance...")
    
    df['class_num'] = pd.to_numeric(
        df['class_clean'].str.extract(r'(\d+)', expand=False), 
        errors='coerce'
    )
    
    for class_level in [1, 2, 3, 4]:
        class_df = df[df['class_num'] == class_level]
        
        sire_class = class_df.groupby('sire_id').agg({
            'won': ['sum', 'count']
        }).reset_index()
        sire_class.columns = ['sire_id', f'class{class_level}_wins', f'class{class_level}_runs']
        sire_class[f'class{class_level}_win_rate'] = (
            sire_class[f'class{class_level}_wins'] / sire_class[f'class{class_level}_runs']
        )
        
        sire_stats = sire_stats.merge(
            sire_class[['sire_id', f'class{class_level}_win_rate']], 
            on='sire_id', how='left'
        )
    
    # Fill class NAs
    for class_level in [1, 2, 3, 4]:
        sire_stats[f'class{class_level}_win_rate'] = (
            sire_stats[f'class{class_level}_win_rate'].fillna(sire_stats['win_rate'])
        )
    
    # === SUMMARY STATS ===
    print("\n" + "="*60)
    print("SIRE LOOKUP SUMMARY")
    print("="*60)
    print(f"Total sires in lookup: {len(sire_stats):,}")
    print(f"Average win rate: {sire_stats['win_rate'].mean():.3f}")
    print(f"Average runs per sire: {sire_stats['runs'].mean():.0f}")
    print(f"\nTop 10 sires by win rate (min {min_runners} runs):")
    print(sire_stats.nlargest(10, 'win_rate')[['sire', 'runs', 'win_rate', 'avg_dist', 'best_distance_band']])
    
    return sire_stats

def save_lookup(sire_stats, output_path='data/processed/lookups/sire_stats.csv'):
    """Save sire lookup table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sire_stats.to_csv(output_path, index=False)
    print(f"\n✓ Saved sire lookup to {output_path}")
    print(f"  Size: {len(sire_stats):,} rows, {len(sire_stats.columns)} columns")

if __name__ == '__main__':
    # Load historical race data
    print("Loading race data...")
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Build sire lookup
    sire_lookup = build_sire_lookup(df, min_runners=20)
    
    # Save
    save_lookup(sire_lookup)
    
    print("\n✓ COMPLETE: Sire lookup table built successfully")
