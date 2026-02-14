#!/usr/bin/env python3
"""
Add Going (Ground) Preference Features

Calculates horse and sire preferences for different going conditions:
- Horse's historical performance by going type (heavy, soft, good, firm)
- Going match score (how close is today's going to horse's preferred going)
- Sire going preferences

Input: data/processed/race_scores_connections_v2.parquet
Output: data/processed/race_scores_going_pref.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

print(f"\n{'='*70}")
print(f"ADD GOING PREFERENCE FEATURES")
print(f"{'='*70}\n")

# Paths
DATA_DIR = Path('data/processed')
INPUT_FILE = DATA_DIR / 'race_scores_with_pedigree_no_leakage.parquet'  # Use no-leakage version with full data
OUTPUT_FILE = DATA_DIR / 'race_scores_going_pref.parquet'

# Load data
print(f"📂 Loading: {INPUT_FILE.name}")
df = pd.read_parquet(INPUT_FILE)
print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}\n")

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Categorize going types
def categorize_going(going_str):
    """Categorize going into groups"""
    if pd.isna(going_str):
        return 'good'
    going_lower = str(going_str).lower()
    
    if any(x in going_lower for x in ['heavy', 'mud']):
        return 'heavy'
    elif any(x in going_lower for x in ['soft', 'yield']):
        return 'soft'
    elif any(x in going_lower for x in ['firm', 'fast']):
        return 'firm'
    else:
        return 'good'

print("🌧️ Categorizing Going Types...")
df['going_category'] = df['going'].apply(categorize_going)

going_dist = df['going_category'].value_counts()
print(f"   Going distribution:")
for going, count in going_dist.items():
    print(f"      {going}: {count:,} ({count/len(df)*100:.1f}%)")

# Create performance indicators
df['won'] = (df['pos_clean'] == 1).astype(int)

print("\n🐴 Calculating Horse Going Preferences...")
print("="*70)

# For each horse, calculate win rate by going type (using expanding window)
for going_type in ['heavy', 'soft', 'good', 'firm']:
    print(f"   📊 {going_type.capitalize()} going...")
    
    # Filter to this going type and SORT BY HORSE AND DATE to prevent leakage
    going_df = df[df['going_category'] == going_type].copy()
    going_df = going_df.sort_values(['horse_id', 'date']).reset_index(drop=True)
    
    # Calculate expanding stats using transform (shifted to avoid lookahead)
    going_df[f'horse_{going_type}_wins'] = (
        going_df.groupby('horse_id')['won']
        .transform(lambda x: x.shift(1).expanding().sum())
        .fillna(0)
    )
    going_df[f'horse_{going_type}_runs'] = (
        going_df.groupby('horse_id')['won']
        .transform(lambda x: x.shift(1).expanding().count())
        .fillna(0)
    )
    
    # Merge back to main dataframe using the original index
    df[f'horse_{going_type}_wins'] = df[f'horse_{going_type}_wins'].fillna(0) if f'horse_{going_type}_wins' in df.columns else 0
    df[f'horse_{going_type}_runs'] = df[f'horse_{going_type}_runs'].fillna(0) if f'horse_{going_type}_runs' in df.columns else 0
    
    # Update values for this going type
    mask = df['going_category'] == going_type
    df.loc[mask, f'horse_{going_type}_wins'] = going_df[f'horse_{going_type}_wins'].values
    df.loc[mask, f'horse_{going_type}_runs'] = going_df[f'horse_{going_type}_runs'].values
    
    # Calculate win rate
    df[f'horse_{going_type}_win_rate'] = (
        df[f'horse_{going_type}_wins'] / df[f'horse_{going_type}_runs']
    ).replace([np.inf, -np.inf], 0).fillna(0)

# Calculate horse's best going (where they have highest win rate with min 2 runs)
print(f"\n   🏆 Calculating best going for each horse...")

def get_best_going(row):
    """Determine horse's best going based on win rates"""
    going_rates = {}
    for going in ['heavy', 'soft', 'good', 'firm']:
        runs = row[f'horse_{going}_runs']
        rate = row[f'horse_{going}_win_rate']
        if runs >= 2 and pd.notna(rate):
            going_rates[going] = rate
    
    if going_rates:
        return max(going_rates, key=going_rates.get)
    return row['going_category']  # Default to current going

df['horse_best_going'] = df.apply(get_best_going, axis=1)

# Calculate going match score (win rate on this going vs overall)
df['going_match_score'] = df.apply(
    lambda row: row[f'horse_{row["going_category"]}_win_rate']
    if pd.notna(row[f'horse_{row["going_category"]}_win_rate'])
    else row['horse_win_rate'],
    axis=1
)

# Going match binary (is this the horse's preferred going?)
df['going_is_preferred'] = (df['going_category'] == df['horse_best_going']).astype(int)

print(f"      ✓ Horses with going preference data: {df['horse_best_going'].notna().sum():,}")

print(f"\n🧬 Calculating Sire Going Preferences...")
print("="*70)

# Calculate sire going preferences
for going_type in ['heavy', 'soft', 'good', 'firm']:
    print(f"   📊 {going_type.capitalize()} going...")
    
    # Filter to this going type and SORT BY SIRE AND DATE to prevent leakage
    going_df = df[df['going_category'] == going_type].copy()
    going_df = going_df.sort_values(['sire_id', 'date']).reset_index(drop=True)
    
    # Calculate expanding stats using transform (shifted to avoid lookahead)
    going_df[f'sire_{going_type}_wins'] = (
        going_df.groupby('sire_id')['won']
        .transform(lambda x: x.shift(1).expanding().sum())
        .fillna(0)
    )
    going_df[f'sire_{going_type}_runs'] = (
        going_df.groupby('sire_id')['won']
        .transform(lambda x: x.shift(1).expanding().count())
        .fillna(0)
    )
    
    # Update values for this going type
    df[f'sire_{going_type}_wins'] = df[f'sire_{going_type}_wins'].fillna(0) if f'sire_{going_type}_wins' in df.columns else 0
    df[f'sire_{going_type}_runs'] = df[f'sire_{going_type}_runs'].fillna(0) if f'sire_{going_type}_runs' in df.columns else 0
    
    mask = df['going_category'] == going_type
    df.loc[mask, f'sire_{going_type}_wins'] = going_df[f'sire_{going_type}_wins'].values
    df.loc[mask, f'sire_{going_type}_runs'] = going_df[f'sire_{going_type}_runs'].values
    
    # Calculate win rate
    df[f'sire_{going_type}_win_rate'] = (
        df[f'sire_{going_type}_wins'] / df[f'sire_{going_type}_runs']
    ).replace([np.inf, -np.inf], 0).fillna(0)

# Sire going match (sire's win rate on this going)
df['sire_going_match_v2'] = df.apply(
    lambda row: row[f'sire_{row["going_category"]}_win_rate']
    if pd.notna(row[f'sire_{row["going_category"]}_win_rate'])
    else row['sire_win_rate'],
    axis=1
)

# Cleanup temp columns (keep only the features we want)
temp_cols = [c for c in df.columns if '_wins' in c or '_runs' in c]
df = df.drop(columns=temp_cols, errors='ignore')

print(f"\n📊 Summary:")
print("="*70)
print(f"New going preference features added: 6")
print(f"   - going_category (heavy/soft/good/firm)")
print(f"   - horse_heavy_win_rate, horse_soft_win_rate, horse_good_win_rate, horse_firm_win_rate")
print(f"   - sire_heavy_win_rate, sire_soft_win_rate, sire_good_win_rate, sire_firm_win_rate")
print(f"   - horse_best_going")
print(f"   - going_match_score")
print(f"   - going_is_preferred")
print(f"   - sire_going_match_v2")

print(f"\nGoing match score stats:")
print(f"   Mean: {df['going_match_score'].mean():.3f}")
print(f"   Std: {df['going_match_score'].std():.3f}")
print(f"   Preferred going matches: {df['going_is_preferred'].sum():,} ({df['going_is_preferred'].mean()*100:.1f}%)")

# Save output
print(f"\n💾 Saving: {OUTPUT_FILE.name}")
df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')

file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
print(f"   ✓ Saved {len(df):,} rows, {len(df.columns)} columns ({file_size_mb:.2f} MB)")

print(f"\n✅ Going preference features complete!")
print(f"{'='*70}\n")
