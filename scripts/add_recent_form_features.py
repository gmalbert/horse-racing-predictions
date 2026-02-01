#!/usr/bin/env python3
"""
Add Jockey/Trainer Recent Form Features
Implements CRITICAL_DATA_GAPS.md Section 3: Jockey/Trainer Form
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

def add_recent_form_features(df):
    """
    Add 14-day and 30-day form for jockeys and trainers.
    VECTORIZED VERSION for performance.
    
    Features added:
    - jockey_form_14d: Jockey win rate last 14 days
    - jockey_form_30d: Jockey win rate last 30 days
    - trainer_form_14d: Trainer win rate last 14 days
    - trainer_form_30d: Trainer win rate last 30 days
    - jockey_course_form_30d: Jockey recent form at this course
    - trainer_course_form_30d: Trainer recent form at this course
    - jockey_trainer_form_30d: This jockey-trainer combo recent form
    """
    print("\n" + "="*60)
    print("ADDING RECENT FORM FEATURES (VECTORIZED)")
    print("="*60)
    
    df = df.copy()
    df = df.sort_values(['date', 'off']).copy()
    
    # Convert date to datetime if needed
    if df['date'].dtype != 'datetime64[ns]':
        df['date_dt'] = pd.to_datetime(df['date'])
    else:
        df['date_dt'] = df['date']
    
    # Set as index for rolling operations
    df = df.set_index('date_dt').sort_index()
    
    # Create win flag
    df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # === JOCKEY FORM ===
    print("\n1. Calculating jockey recent form...")
    
    # 14-day jockey form (vectorized)
    print("   14-day jockey form...")
    df['jockey_form_14d'] = df.groupby('jockey')['won'].transform(
        lambda x: x.shift(1).rolling('14D', min_periods=3).mean()
    )
    
    # 30-day jockey form
    print("   30-day jockey form...")
    df['jockey_form_30d'] = df.groupby('jockey')['won'].transform(
        lambda x: x.shift(1).rolling('30D', min_periods=5).mean()
    )
    
    # === TRAINER FORM ===
    print("\n2. Calculating trainer recent form...")
    
    # 14-day trainer form
    print("   14-day trainer form...")
    df['trainer_form_14d'] = df.groupby('trainer')['won'].transform(
        lambda x: x.shift(1).rolling('14D', min_periods=3).mean()
    )
    
    # 30-day trainer form
    print("   30-day trainer form...")
    df['trainer_form_30d'] = df.groupby('trainer')['won'].transform(
        lambda x: x.shift(1).rolling('30D', min_periods=5).mean()
    )
    
    # === COURSE-SPECIFIC FORM ===
    print("\n3. Calculating course-specific recent form...")
    
    # Jockey at course in last 30 days
    print("   Jockey-course form...")
    df['jockey_course_key'] = df['jockey'] + '_' + df['course_clean']
    df['jockey_course_form_30d'] = df.groupby('jockey_course_key')['won'].transform(
        lambda x: x.shift(1).rolling('30D', min_periods=2).mean()
    )
    
    # Trainer at course in last 30 days
    print("   Trainer-course form...")
    df['trainer_course_key'] = df['trainer'] + '_' + df['course_clean']
    df['trainer_course_form_30d'] = df.groupby('trainer_course_key')['won'].transform(
        lambda x: x.shift(1).rolling('30D', min_periods=2).mean()
    )
    
    # === JOCKEY-TRAINER COMBO ===
    print("\n4. Calculating jockey-trainer combination form...")
    
    df['jockey_trainer_key'] = df['jockey'] + '_' + df['trainer']
    df['jockey_trainer_form_30d'] = df.groupby('jockey_trainer_key')['won'].transform(
        lambda x: x.shift(1).rolling('30D', min_periods=2).mean()
    )
    
    # Reset index
    df = df.reset_index()
    
    # === FILL NAs WITH CAREER RATES ===
    print("\n5. Filling NAs with career statistics...")
    
    # Calculate career rates if not present
    if 'jockey_career_win_rate' not in df.columns:
        df['jockey_career_win_rate'] = df.groupby('jockey')['won'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    
    if 'trainer_career_win_rate' not in df.columns:
        df['trainer_career_win_rate'] = df.groupby('trainer')['won'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    
    # Fill recent form NAs with career rates
    df['jockey_form_14d'] = df['jockey_form_14d'].fillna(df['jockey_career_win_rate']).fillna(0.10)
    df['jockey_form_30d'] = df['jockey_form_30d'].fillna(df['jockey_career_win_rate']).fillna(0.10)
    df['trainer_form_14d'] = df['trainer_form_14d'].fillna(df['trainer_career_win_rate']).fillna(0.10)
    df['trainer_form_30d'] = df['trainer_form_30d'].fillna(df['trainer_career_win_rate']).fillna(0.10)
    
    # Course-specific form falls back to overall recent form
    df['jockey_course_form_30d'] = df['jockey_course_form_30d'].fillna(df['jockey_form_30d'])
    df['trainer_course_form_30d'] = df['trainer_course_form_30d'].fillna(df['trainer_form_30d'])
    df['jockey_trainer_form_30d'] = df['jockey_trainer_form_30d'].fillna(df['jockey_form_30d'])
    
    # === FORM INDICATORS ===
    print("\n6. Creating form indicator flags...")
    
    # Jockey in form (>20% in last 14 days)
    df['jockey_in_form'] = (df['jockey_form_14d'] > 0.20).astype(int)
    
    # Trainer in form (>15% in last 14 days)
    df['trainer_in_form'] = (df['trainer_form_14d'] > 0.15).astype(int)
    
    # Both in form
    df['connections_in_form'] = (df['jockey_in_form'] & df['trainer_in_form']).astype(int)
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("RECENT FORM FEATURES SUMMARY")
    print("="*60)
    
    print(f"\nJockey form coverage:")
    print(f"  14-day: {(df['jockey_form_14d'] != df['jockey_career_win_rate']).sum():,} horses ({(df['jockey_form_14d'] != df['jockey_career_win_rate']).mean()*100:.1f}%)")
    print(f"  30-day: {(df['jockey_form_30d'] != df['jockey_career_win_rate']).sum():,} horses ({(df['jockey_form_30d'] != df['jockey_career_win_rate']).mean()*100:.1f}%)")
    
    print(f"\nTrainer form coverage:")
    print(f"  14-day: {(df['trainer_form_14d'] != df['trainer_career_win_rate']).sum():,} horses ({(df['trainer_form_14d'] != df['trainer_career_win_rate']).mean()*100:.1f}%)")
    print(f"  30-day: {(df['trainer_form_30d'] != df['trainer_career_win_rate']).sum():,} horses ({(df['trainer_form_30d'] != df['trainer_career_win_rate']).mean()*100:.1f}%)")
    
    print(f"\nJockeys in form (>20% last 14d): {df['jockey_in_form'].sum():,}")
    print(f"Trainers in form (>15% last 14d): {df['trainer_in_form'].sum():,}")
    print(f"Both in form: {df['connections_in_form'].sum():,}")
    
    print(f"\nNew features added: 10")
    print(f"  - jockey_form_14d, jockey_form_30d")
    print(f"  - trainer_form_14d, trainer_form_30d")
    print(f"  - jockey_course_form_30d, trainer_course_form_30d")
    print(f"  - jockey_trainer_form_30d")
    print(f"  - jockey_in_form, trainer_in_form, connections_in_form")
    
    return df

if __name__ == '__main__':
    # Load data (prefer no-leakage pace > pace > no-leakage pedigree > pedigree > base)
    print("Loading race data...")
    
    pace_no_leak = Path('data/processed/race_scores_with_all_features_no_leakage.parquet')
    pace_path = Path('data/processed/race_scores_with_all_features.parquet')
    pedigree_no_leak = Path('data/processed/race_scores_with_pedigree_no_leakage.parquet')
    pedigree_path = Path('data/processed/race_scores_with_pedigree.parquet')
    base_path = Path('data/processed/race_scores.parquet')
    
    if pace_no_leak.exists():
        print(f"  Loading from {pace_no_leak} (no leakage, with pace)")
        df = pd.read_parquet(pace_no_leak)
    elif pace_path.exists():
        print(f"  Loading from {pace_path} (with pace features)")
        df = pd.read_parquet(pace_path)
    elif pedigree_no_leak.exists():
        print(f"  Loading from {pedigree_no_leak} (no leakage, with pedigree)")
        df = pd.read_parquet(pedigree_no_leak)
    elif pedigree_path.exists():
        print(f"  Loading from {pedigree_path} (with pedigree features)")
        df = pd.read_parquet(pedigree_path)
    else:
        print(f"  Loading from {base_path} (base data)")
        df = pd.read_parquet(base_path)
    
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Add recent form features
    print("\nAdding recent form features (this may take a few minutes)...")
    df_with_form = add_recent_form_features(df)
    
    # Save (overwrite the all_features file to add form features)
    output_path = Path('data/processed/race_scores_with_all_features_no_leakage.parquet')
    print(f"\nSaving to {output_path}...")
    df_with_form.to_parquet(output_path, index=False)
    
    print("\n✓ COMPLETE: Recent form features added successfully")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(df_with_form):,}")
    print(f"  Columns: {len(df_with_form.columns)}")
