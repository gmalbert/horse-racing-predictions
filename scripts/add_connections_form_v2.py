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
    """
    print("\n" + "="*70)
    print("ENHANCED CONNECTIONS FORM FEATURES (V2)")
    print("="*70)
    
    # Ensure sorted by date
    df = df.sort_values('date_dt').copy()
    
    # Ensure we have won column
    if 'won' not in df.columns:
        df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # We'll calculate features using expanding windows with date filtering
    # to avoid leakage
    
    print("\n1. Jockey Form (14d and 30d)")
    # For jockey form, we need to use date-based rolling windows
    # Group by jockey and calculate rolling stats
    
    for role in ['jockey', 'trainer']:
        print(f"\n  Processing {role} form...")
        
        for days in [14, 30]:
            print(f"    {days}-day window...")
            
            # Create a timedelta for the window
            window_str = f'{days}D'
            
            # We need to sort by role and date
            df_sorted = df.sort_values([role, 'date_dt']).copy()
            
            # For each row, count runs and wins in the prior X days
            # Using a manual approach to ensure no leakage
            
            # Initialize columns
            df[f'{role}_runs_{days}d_v2'] = 0
            df[f'{role}_wins_{days}d_v2'] = 0
            
            # Group by role
            for name, group in df_sorted.groupby(role):
                if pd.isna(name) or name == '':
                    continue
                
                group = group.sort_values('date_dt').copy()
                
                runs = []
                wins = []
                
                for idx, row in group.iterrows():
                    race_date = row['date_dt']
                    
                    # Look back X days (excluding today)
                    cutoff_date = race_date - timedelta(days=days)
                    
                    # Filter to prior races in window
                    prior_races = group[
                        (group['date_dt'] < race_date) & 
                        (group['date_dt'] >= cutoff_date)
                    ]
                    
                    num_runs = len(prior_races)
                    num_wins = prior_races['won'].sum() if num_runs > 0 else 0
                    
                    runs.append(num_runs)
                    wins.append(num_wins)
                
                # Assign back to dataframe
                df.loc[group.index, f'{role}_runs_{days}d_v2'] = runs
                df.loc[group.index, f'{role}_wins_{days}d_v2'] = wins
            
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
    
    print("\n3. Trainer-Jockey Combination Form")
    # Create combo key
    df['combo_key'] = df['trainer'].astype(str) + '_' + df['jockey'].astype(str)
    
    # Sort by combo and date
    df_sorted = df.sort_values(['combo_key', 'date_dt']).copy()
    
    df['combo_runs_30d_v2'] = 0
    df['combo_wins_30d_v2'] = 0
    
    for combo, group in df_sorted.groupby('combo_key'):
        if pd.isna(combo) or combo == '_' or combo == 'nan_nan':
            continue
        
        group = group.sort_values('date_dt').copy()
        
        runs = []
        wins = []
        
        for idx, row in group.iterrows():
            race_date = row['date_dt']
            cutoff_date = race_date - timedelta(days=30)
            
            prior_races = group[
                (group['date_dt'] < race_date) & 
                (group['date_dt'] >= cutoff_date)
            ]
            
            num_runs = len(prior_races)
            num_wins = prior_races['won'].sum() if num_runs > 0 else 0
            
            runs.append(num_runs)
            wins.append(num_wins)
        
        df.loc[group.index, 'combo_runs_30d_v2'] = runs
        df.loc[group.index, 'combo_wins_30d_v2'] = wins
    
    df['combo_form_30d_v2'] = (
        df['combo_wins_30d_v2'] / 
        df['combo_runs_30d_v2'].clip(lower=1)
    )
    
    # Hot combo flag
    df['combo_hot_v2'] = (
        (df['combo_form_30d_v2'] > 0.25) & 
        (df['combo_runs_30d_v2'] >= 3)
    ).astype(int)
    
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
    
    # Add features (this will take a while due to iterative calculation)
    print("\n⚠️  WARNING: This will take several minutes due to time-window calculations...")
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
