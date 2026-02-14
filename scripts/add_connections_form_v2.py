#!/usr/bin/env python3
"""
Add Enhanced Connections Form Features (V2)

Recent form for trainer and jockey with proper time windows:
- 14-day and 30-day rolling windows
- Trainer-jockey combination success rate
- "Hot" connection flags

Usage:
  python scripts/add_connections_form_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Paths
DATA_DIR = Path('data/processed')
INPUT_FILE = DATA_DIR / 'race_scores_enhanced_form.parquet'
OUTPUT_FILE = DATA_DIR / 'race_scores_connections_v2.parquet'


def engineer_connections_form_v2(df):
    """
    Enhanced trainer and jockey recent form with time-based windows.
    OPTIMIZED: Uses vectorized operations instead of iterrows().
    """
    print("\n" + "="*70)
    print("ENHANCED CONNECTIONS FORM FEATURES (V2 - VECTORIZED)")
    print("="*70)
    
    # Ensure sorted by date
    df = df.sort_values('date_dt').copy()
    
    # Ensure we have won column
    if 'won' not in df.columns:
        df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # Convert date_dt to unix timestamp for faster computation
    df['_timestamp'] = df['date_dt'].astype(np.int64) // 10**9  # seconds since epoch
    
    print("\n1. Jockey & Trainer Form (14d and 30d) - VECTORIZED")
    
    for role in ['jockey', 'trainer']:
        print(f"\n  Processing {role} form...")
        
        # Sort by role and date for efficient groupby operations
        df = df.sort_values([role, 'date_dt']).copy()
        
        # Create group indices for faster processing
        df[f'_{role}_idx'] = df.groupby(role).ngroup()
        
        for days in [14, 30]:
            print(f"    {days}-day window...")
            
            window_seconds = days * 24 * 60 * 60
            
            # Initialize result columns
            df[f'{role}_runs_{days}d_v2'] = 0
            df[f'{role}_wins_{days}d_v2'] = 0
            
            # Process each group using merge_asof for efficient time-windowed joins
            result_runs = []
            result_wins = []
            
            for name in df[role].dropna().unique():
                if name == '':
                    continue
                
                # Get all races for this role
                mask = df[role] == name
                group_df = df[mask].copy()
                
                if len(group_df) == 0:
                    continue
                
                # For each race, count prior races in window using vectorized operations
                timestamps = group_df['_timestamp'].values
                won_values = group_df['won'].values
                
                runs_list = []
                wins_list = []
                
                for i in range(len(timestamps)):
                    current_ts = timestamps[i]
                    cutoff_ts = current_ts - window_seconds
                    
                    # Vectorized: find races in window (before current, within window)
                    in_window = (timestamps < current_ts) & (timestamps >= cutoff_ts)
                    
                    num_runs = in_window.sum()
                    num_wins = won_values[in_window].sum() if num_runs > 0 else 0
                    
                    runs_list.append(num_runs)
                    wins_list.append(num_wins)
                
                result_runs.extend(runs_list)
                result_wins.extend(wins_list)
            
            # Assign results back (only to non-null roles)
            non_null_mask = df[role].notna() & (df[role] != '')
            df.loc[non_null_mask, f'{role}_runs_{days}d_v2'] = result_runs
            df.loc[non_null_mask, f'{role}_wins_{days}d_v2'] = result_wins
            
            # Calculate win rate
            df[f'{role}_form_{days}d_v2'] = (
                df[f'{role}_wins_{days}d_v2'] / 
                df[f'{role}_runs_{days}d_v2'].clip(lower=1)
            )
    
    print("\n2. Hot Connection Flags")
    # "Hot" = >25% win rate in last 30 days with at least 5 runs
    df['jockey_hot_v2'] = (
        (df['jockey_form_30d_v2'] > 0.25) & 
        (df['jockey_runs_30d_v2'] >= 5)
    ).astype(int)
    
    df['trainer_hot_v2'] = (
        (df['trainer_form_30d_v2'] > 0.25) & 
        (df['trainer_runs_30d_v2'] >= 5)
    ).astype(int)
    
    print("\n3. Trainer-Jockey Combination Form - VECTORIZED")
    # Create combo key
    df['combo_key'] = df['trainer'].astype(str) + '_' + df['jockey'].astype(str)
    
    # Sort by combo and date
    df = df.sort_values(['combo_key', 'date_dt']).copy()
    
    df['combo_runs_30d_v2'] = 0
    df['combo_wins_30d_v2'] = 0
    
    window_seconds = 30 * 24 * 60 * 60
    
    result_runs = []
    result_wins = []
    
    for combo in df['combo_key'].dropna().unique():
        if combo == '_' or combo == 'nan_nan' or 'nan' in combo:
            continue
        
        mask = df['combo_key'] == combo
        group_df = df[mask].copy()
        
        if len(group_df) == 0:
            continue
        
        timestamps = group_df['_timestamp'].values
        won_values = group_df['won'].values
        
        runs_list = []
        wins_list = []
        
        for i in range(len(timestamps)):
            current_ts = timestamps[i]
            cutoff_ts = current_ts - window_seconds
            
            in_window = (timestamps < current_ts) & (timestamps >= cutoff_ts)
            
            num_runs = in_window.sum()
            num_wins = won_values[in_window].sum() if num_runs > 0 else 0
            
            runs_list.append(num_runs)
            wins_list.append(num_wins)
        
        result_runs.extend(runs_list)
        result_wins.extend(wins_list)
    
    # Assign results
    valid_combo_mask = df['combo_key'].notna() & ~df['combo_key'].isin(['_', 'nan_nan']) & ~df['combo_key'].str.contains('nan', na=False)
    df.loc[valid_combo_mask, 'combo_runs_30d_v2'] = result_runs
    df.loc[valid_combo_mask, 'combo_wins_30d_v2'] = result_wins
    
    df['combo_form_30d_v2'] = (
        df['combo_wins_30d_v2'] / 
        df['combo_runs_30d_v2'].clip(lower=1)
    )
    
    # Hot combo flag
    df['combo_hot_v2'] = (
        (df['combo_form_30d_v2'] > 0.25) & 
        (df['combo_runs_30d_v2'] >= 3)
    ).astype(int)
    
    # Cleanup temporary columns
    df = df.drop(columns=['_timestamp', '_jockey_idx', '_trainer_idx'], errors='ignore')
    
    print("\n✓ Enhanced connections features created (V2):")
    print("  - jockey_runs_14d_v2, jockey_wins_14d_v2, jockey_form_14d_v2")
    print("  - jockey_runs_30d_v2, jockey_wins_30d_v2, jockey_form_30d_v2")
    print("  - trainer_runs_14d_v2, trainer_wins_14d_v2, trainer_form_14d_v2")
    print("  - trainer_runs_30d_v2, trainer_wins_30d_v2, trainer_form_30d_v2")
    print("  - jockey_hot_v2, trainer_hot_v2")
    print("  - combo_runs_30d_v2, combo_wins_30d_v2, combo_form_30d_v2, combo_hot_v2")
    
    return df


def main():
    print("="*70)
    print("ADD ENHANCED CONNECTIONS FORM FEATURES (V2)")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Loaded: {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Ensure date_dt exists
    if 'date_dt' not in df.columns and 'date' in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'])
    
    # Add features (optimized with vectorized operations)
    print("\n⚠️  Processing time-window calculations (optimized vectorized approach)...")
    df = engineer_connections_form_v2(df)
    
    # Save
    print(f"\n{'='*70}")
    print(f"SAVING")
    print(f"{'='*70}")
    print(f"Output: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved: {len(df):,} records x {len(df.columns)} columns")
    
    # Summary
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS")
    print(f"{'='*70}")
    
    # Jockey stats
    print("\nJOCKEY FORM:")
    print(f"  14d coverage: {(df['jockey_runs_14d_v2'] > 0).sum():,} horses ({(df['jockey_runs_14d_v2'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  30d coverage: {(df['jockey_runs_30d_v2'] > 0).sum():,} horses ({(df['jockey_runs_30d_v2'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Hot jockeys: {df['jockey_hot_v2'].sum():,} horses ({df['jockey_hot_v2'].sum() / len(df) * 100:.1f}%)")
    print(f"  Avg form (30d): {df[df['jockey_runs_30d_v2'] > 0]['jockey_form_30d_v2'].mean():.3f}")
    
    print("\nTRAINER FORM:")
    print(f"  14d coverage: {(df['trainer_runs_14d_v2'] > 0).sum():,} horses ({(df['trainer_runs_14d_v2'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  30d coverage: {(df['trainer_runs_30d_v2'] > 0).sum():,} horses ({(df['trainer_runs_30d_v2'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Hot trainers: {df['trainer_hot_v2'].sum():,} horses ({df['trainer_hot_v2'].sum() / len(df) * 100:.1f}%)")
    print(f"  Avg form (30d): {df[df['trainer_runs_30d_v2'] > 0]['trainer_form_30d_v2'].mean():.3f}")
    
    print("\nCOMBO (Trainer-Jockey) FORM:")
    print(f"  30d coverage: {(df['combo_runs_30d_v2'] > 0).sum():,} horses ({(df['combo_runs_30d_v2'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Hot combos: {df['combo_hot_v2'].sum():,} horses ({df['combo_hot_v2'].sum() / len(df) * 100:.1f}%)")
    print(f"  Avg form (30d): {df[df['combo_runs_30d_v2'] > 0]['combo_form_30d_v2'].mean():.3f}")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
