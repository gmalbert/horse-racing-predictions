#!/usr/bin/env python3
"""
Add Official Rating (OR) Context Features

Adds race-level comparisons for Official Ratings:
- OR vs race maximum (how competitive is this horse?)
- OR vs race average (above/below par?)
- OR vs class typical (well-handicapped?)
- OR percentile (elite vs ordinary)
- OR career high (at peak or improving?)

Input: data/processed/race_scores_going_pref.parquet
Output: data/processed/race_scores_or_context.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

print(f"\n{'='*70}")
print(f"ADD OFFICIAL RATING (OR) CONTEXT FEATURES")
print(f"{'='*70}\n")

# Paths
DATA_DIR = Path('data/processed')
INPUT_FILE = DATA_DIR / 'race_scores_going_pref.parquet'
OUTPUT_FILE = DATA_DIR / 'race_scores_or_context.parquet'

# Load data
print(f"📂 Loading: {INPUT_FILE.name}")
df = pd.read_parquet(INPUT_FILE)
print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}\n")

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Convert 'or' to numeric if it exists (handle both 'or' and 'or_numeric' column names)
if 'or' in df.columns and df['or'].dtype == 'object':
    df['or_numeric'] = pd.to_numeric(df['or'], errors='coerce')
elif 'or' in df.columns:
    df['or_numeric'] = df['or']

print("🏆 Calculating Race-Level OR Context...")
print("="*70)

# 1. OR vs Race Maximum
print("   📊 OR vs race maximum...")
race_or_stats = df.groupby('race_id')['or_numeric'].agg(['max', 'mean', 'min', 'std']).reset_index()
race_or_stats.columns = ['race_id', 'race_or_max', 'race_or_mean', 'race_or_min', 'race_or_std']

df = df.merge(race_or_stats, on='race_id', how='left')

df['or_vs_race_max'] = df['or_numeric'] - df['race_or_max']
df['or_vs_race_avg'] = df['or_numeric'] - df['race_or_mean']
df['or_vs_race_min'] = df['or_numeric'] - df['race_or_min']

# Is this horse the highest rated?
df['is_highest_rated'] = (df['or_numeric'] == df['race_or_max']).astype(int)

print(f"      ✓ Mean OR vs max: {df['or_vs_race_max'].mean():.2f}")
print(f"      ✓ Highest rated horses: {df['is_highest_rated'].sum():,} ({df['is_highest_rated'].mean()*100:.1f}%)")

# 2. OR vs Class Typical
print("   🎯 OR vs class typical...")

# Use class_num if available, fall back to class_numeric
class_col = 'class_num' if 'class_num' in df.columns else 'class_numeric'

# Sort by date for temporal integrity
df = df.sort_values([class_col, 'date']).reset_index(drop=True)

# Calculate typical OR for each class using expanding window (no leakage)
df['class_or_typical'] = (
    df.groupby(class_col)['or_numeric']
    .transform(lambda x: x.shift(1).expanding().mean())
    .fillna(df['or_numeric'])
)

df['or_vs_class_typical'] = df['or_numeric'] - df['class_or_typical']

# Well-handicapped flag (OR below class average = potentially well-handicapped)
df['is_well_handicapped'] = (df['or_vs_class_typical'] < -5).astype(int)

print(f"      ✓ Mean OR vs class typical: {df['or_vs_class_typical'].mean():.2f}")
print(f"      ✓ Well-handicapped horses: {df['is_well_handicapped'].sum():,} ({df['is_well_handicapped'].mean()*100:.1f}%)")

# 3. OR Percentile (within all horses)
print("   📈 OR percentile...")

# Calculate OR percentile using expanding window
df['or_rank'] = df.groupby('date')['or_numeric'].rank(pct=True)
df['or_percentile'] = (df['or_rank'] * 100).round(0)

print(f"      ✓ Mean OR percentile: {df['or_percentile'].mean():.1f}")

# 4. OR Career High
print("   🔝 OR career high...")

# Sort by horse and date for temporal integrity
df = df.sort_values(['horse_id', 'date']).reset_index(drop=True)

# Calculate career high OR for each horse (expanding window, no leakage)
df['or_career_high'] = df.groupby('horse_id')['or_numeric'].cummax()
df['or_at_career_high'] = (df['or_numeric'] == df['or_career_high']).astype(int)
df['or_below_career_high'] = df['or_career_high'] - df['or_numeric']

print(f"      ✓ Horses at career high OR: {df['or_at_career_high'].sum():,} ({df['or_at_career_high'].mean()*100:.1f}%)")
print(f"      ✓ Mean OR below career high: {df['or_below_career_high'].mean():.2f}")

# 5. OR Improvement Potential
print("   📊 OR improvement potential...")

# Has horse improved OR in last 3 runs?
df['or_improving_3'] = (df.groupby('horse_id')['or_numeric']
                         .diff(periods=3)
                         .fillna(0) > 5).astype(int)

# OR volatility (std dev of last 5 ORs)
df['or_volatility'] = (df.groupby('horse_id')['or_numeric']
                        .rolling(window=5, min_periods=2)
                        .std()
                        .reset_index(level=0, drop=True)
                        .fillna(0))

print(f"      ✓ Improving horses: {df['or_improving_3'].sum():,} ({df['or_improving_3'].mean()*100:.1f}%)")
print(f"      ✓ Mean OR volatility: {df['or_volatility'].mean():.2f}")

# 6. OR Relative to Field (percentile within race)
print("   🏁 OR percentile within race...")

df['or_race_percentile'] = df.groupby('race_id')['or_numeric'].rank(pct=True) * 100

print(f"      ✓ Mean race percentile: {df['or_race_percentile'].mean():.1f}")

# Cleanup temp columns
temp_cols = ['or_rank']
df = df.drop(columns=temp_cols, errors='ignore')

print(f"\n📊 Summary:")
print("="*70)
print(f"New OR context features added: 13")
print(f"   - race_or_max, race_or_mean, race_or_min, race_or_std")
print(f"   - or_vs_race_max (competitiveness)")
print(f"   - or_vs_race_avg (above/below par)")
print(f"   - or_vs_race_min")
print(f"   - is_highest_rated")
print(f"   - or_vs_class_typical (well-handicapped?)")
print(f"   - is_well_handicapped")
print(f"   - or_percentile (overall)")
print(f"   - or_career_high")
print(f"   - or_at_career_high")
print(f"   - or_below_career_high")
print(f"   - or_improving_3")
print(f"   - or_volatility")
print(f"   - or_race_percentile (within race)")

# Save output
print(f"\n💾 Saving: {OUTPUT_FILE.name}")
df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')

file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
print(f"   ✓ Saved {len(df):,} rows, {len(df.columns)} columns ({file_size_mb:.2f} MB)")

print(f"\n✅ OR context features complete!")
print(f"{'='*70}\n")
