#!/usr/bin/env python3
"""
Check what data is available for pedigree, pace, and form features.
"""

import pandas as pd
from pathlib import Path

def check_data_availability():
    """Check available columns and data completeness."""
    
    # Load race scores
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    print("="*60)
    print("DATA AVAILABILITY CHECK")
    print("="*60)
    
    print(f"\nTotal rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check for pedigree columns
    print("\n--- PEDIGREE DATA ---")
    pedigree_cols = ['sire', 'sire_id', 'dam', 'dam_id', 'damsire', 'damsire_id']
    for col in pedigree_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            print(f"✓ {col}: {null_pct:.1f}% null")
        else:
            print(f"✗ {col}: NOT FOUND")
    
    # Check for pace/running style data
    print("\n--- PACE/RUNNING STYLE DATA ---")
    pace_cols = ['comments', 'in_running', 'comment', 'running_comment']
    for col in pace_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            print(f"✓ {col}: {null_pct:.1f}% null")
            if null_pct < 90:
                print(f"  Sample: {df[col].dropna().iloc[0][:100]}")
        else:
            print(f"✗ {col}: NOT FOUND")
    
    # Check for jockey/trainer
    print("\n--- JOCKEY/TRAINER DATA ---")
    for col in ['jockey', 'trainer', 'date']:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            print(f"✓ {col}: {null_pct:.1f}% null")
        else:
            print(f"✗ {col}: NOT FOUND")
    
    # Show all columns
    print("\n--- ALL COLUMNS ---")
    print(df.columns.tolist())
    
    # Sample row
    print("\n--- SAMPLE ROW ---")
    print(df.iloc[0].to_dict())

if __name__ == '__main__':
    check_data_availability()
