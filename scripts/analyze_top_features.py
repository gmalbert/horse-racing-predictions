"""
Analyze top 3 features to understand if 0.993 AUC is legitimate or indicates leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "race_scores.parquet"

print("="*70)
print("ANALYZING TOP 3 FEATURES FOR DATA LEAKAGE")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_FILE) if DATA_FILE.suffix == '.csv' else pd.read_parquet(DATA_FILE)
print(f"Loaded {len(df):,} records")

# Parse necessary columns
df['date'] = pd.to_datetime(df['date'])
df['pos_clean'] = pd.to_numeric(df['pos'], errors='coerce')
df['won'] = (df['pos_clean'] == 1).astype(int)

# Extract class_num
df['class_num'] = df['class_clean'].str.extract(r'(\d+)').astype(float)

# Extract is_pattern
df['is_pattern'] = df['pattern'].notna().astype(int)

# Extract prize_log
if 'prize_clean' in df.columns:
    df['prize_numeric'] = pd.to_numeric(df['prize_clean'], errors='coerce').fillna(0)
elif 'prize_numeric' in df.columns:
    df['prize_numeric'] = pd.to_numeric(df['prize_numeric'], errors='coerce').fillna(0)
else:
    print("Warning: No prize column found")
    df['prize_numeric'] = 0

df['prize_log'] = np.log1p(df['prize_numeric'])

print("\n" + "="*70)
print("FEATURE 1: prize_log (Log-transformed Prize Money)")
print("="*70)

print(f"\nPrize distribution:")
print(f"  Min:    £{df['prize_numeric'].min():,.0f}")
print(f"  Median: £{df['prize_numeric'].median():,.0f}")
print(f"  Mean:   £{df['prize_numeric'].mean():,.0f}")
print(f"  Max:    £{df['prize_numeric'].max():,.0f}")

# Compare winners vs losers
winners = df[df['won'] == 1]
losers = df[df['won'] == 0]

print(f"\nPrize for races with winners vs losers:")
print(f"  Winners avg prize: £{winners['prize_numeric'].mean():,.0f}")
print(f"  Losers avg prize:  £{losers['prize_numeric'].mean():,.0f}")
print(f"  Difference:        £{winners['prize_numeric'].mean() - losers['prize_numeric'].mean():,.0f}")

print(f"\nPrize_log for winners vs losers:")
print(f"  Winners avg prize_log: {winners['prize_log'].mean():.4f}")
print(f"  Losers avg prize_log:  {losers['prize_log'].mean():.4f}")
print(f"  Difference:            {winners['prize_log'].mean() - losers['prize_log'].mean():.4f}")

# Break down by prize tier
prize_tiers = [
    ("Low (<£5k)", 0, 5000),
    ("Medium (£5k-£20k)", 5000, 20000),
    ("High (£20k-£100k)", 20000, 100000),
    ("Very High (£100k+)", 100000, 1e9)
]

print(f"\nWin rate by prize tier:")
for tier_name, min_prize, max_prize in prize_tiers:
    tier_df = df[(df['prize_numeric'] >= min_prize) & (df['prize_numeric'] < max_prize)]
    if len(tier_df) > 0:
        win_rate = tier_df['won'].mean() * 100
        print(f"  {tier_name:25} {len(tier_df):7,} races, {win_rate:5.2f}% win rate")

print("\n" + "="*70)
print("FEATURE 2: is_pattern (Group/Listed Races)")
print("="*70)

pattern_counts = df.groupby('is_pattern')['won'].agg(['count', 'sum', 'mean'])
pattern_counts.columns = ['Total Races', 'Wins', 'Win Rate']
pattern_counts['Win Rate'] = pattern_counts['Win Rate'] * 100

print(f"\nPattern race distribution:")
print(pattern_counts)

print(f"\nPattern races:")
print(f"  Count:    {(df['is_pattern'] == 1).sum():,} ({(df['is_pattern'] == 1).mean()*100:.2f}%)")
print(f"  Win rate: {df[df['is_pattern'] == 1]['won'].mean()*100:.2f}%")

print(f"\nNon-pattern races:")
print(f"  Count:    {(df['is_pattern'] == 0).sum():,} ({(df['is_pattern'] == 0).mean()*100:.2f}%)")
print(f"  Win rate: {df[df['is_pattern'] == 0]['won'].mean()*100:.2f}%")

print("\n" + "="*70)
print("FEATURE 3: class_num (Race Class)")
print("="*70)

class_stats = df.groupby('class_num')['won'].agg(['count', 'sum', 'mean']).sort_index()
class_stats.columns = ['Total Races', 'Wins', 'Win Rate']
class_stats['Win Rate'] = class_stats['Win Rate'] * 100

print(f"\nWin rate by race class:")
print(class_stats)

print("\n" + "="*70)
print("CRITICAL LEAKAGE CHECK")
print("="*70)

# Check if these features vary WITHIN races (they shouldn't)
print("\nChecking if features vary within individual races...")

# Sample some races
sample_races = df.groupby(['date', 'course_clean', 'off']).head(10)
race_groups = df.groupby(['date', 'course_clean', 'off'])

varying_prize = 0
varying_pattern = 0
varying_class = 0
total_races = 0

for (date, course, off), group in race_groups:
    total_races += 1
    if group['prize_log'].nunique() > 1:
        varying_prize += 1
    if group['is_pattern'].nunique() > 1:
        varying_pattern += 1
    if group['class_num'].nunique() > 1:
        varying_class += 1

print(f"\nOut of {total_races:,} unique races:")
print(f"  Races with varying prize_log:  {varying_prize:,} ({'ERROR - LEAKAGE!' if varying_prize > 0 else 'OK'})")
print(f"  Races with varying is_pattern: {varying_pattern:,} ({'ERROR - LEAKAGE!' if varying_pattern > 0 else 'OK'})")
print(f"  Races with varying class_num:  {varying_class:,} ({'ERROR - LEAKAGE!' if varying_class > 0 else 'OK'})")

print("\n" + "="*70)
print("FIELD SIZE ANALYSIS (potential bias)")
print("="*70)

# Check if field size correlates with features
df['ran'] = pd.to_numeric(df['ran'], errors='coerce')

print(f"\nAverage field size by prize tier:")
for tier_name, min_prize, max_prize in prize_tiers:
    tier_df = df[(df['prize_numeric'] >= min_prize) & (df['prize_numeric'] < max_prize)]
    if len(tier_df) > 0:
        avg_field = tier_df['ran'].mean()
        print(f"  {tier_name:25} {avg_field:5.1f} runners")

print(f"\nAverage field size by class:")
for class_val in sorted(df['class_num'].dropna().unique()):
    class_df = df[df['class_num'] == class_val]
    avg_field = class_df['ran'].mean()
    print(f"  Class {int(class_val):1}: {avg_field:5.1f} runners")

print(f"\nPattern races field size:")
print(f"  Pattern races:     {df[df['is_pattern'] == 1]['ran'].mean():.1f} runners")
print(f"  Non-pattern races: {df[df['is_pattern'] == 0]['ran'].mean():.1f} runners")

print("\n" + "="*70)
print("RACE QUALITY vs WIN PROBABILITY")
print("="*70)

# The key insight: Are these just proxies for "easier to predict" races?
print("\nHypothesis: High-quality races may have more predictable outcomes")
print("(Favorites in Group 1 races may win more often than in low-class races)")

# Check favorite performance by class
# Note: We don't have favorite data here, but we can check field size
print("\nWin rate by field size (smaller fields = more predictable?):")
field_bins = [0, 6, 10, 14, 20, 100]
field_labels = ['2-5', '6-9', '10-13', '14-19', '20+']

df['field_bin'] = pd.cut(df['ran'], bins=field_bins, labels=field_labels)
field_stats = df.groupby('field_bin')['won'].agg(['count', 'mean'])
field_stats.columns = ['Races', 'Win Rate']
field_stats['Win Rate'] = field_stats['Win Rate'] * 100

print(field_stats)

print("\n" + "="*70)
print("TEMPORAL ANALYSIS")
print("="*70)

# Check if prize/class distribution changed over time
df['year'] = df['date'].dt.year

print(f"\nAverage prize by year (checking for temporal drift):")
year_prize = df.groupby('year')['prize_numeric'].mean().sort_index()
for year, avg_prize in year_prize.items():
    print(f"  {year}: £{avg_prize:,.0f}")

print(f"\nPattern race proportion by year:")
year_pattern = df.groupby('year')['is_pattern'].mean().sort_index() * 100
for year, pct in year_pattern.items():
    print(f"  {year}: {pct:.2f}%")

print(f"\nClass 1-2 proportion by year:")
df['is_high_class'] = df['class_num'].isin([1, 2]).astype(int)
year_class = df.groupby('year')['is_high_class'].mean().sort_index() * 100
for year, pct in year_class.items():
    print(f"  {year}: {pct:.2f}%")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("\nIf any of the following are true, the high AUC may be legitimate:")
print("  1. High-quality races (high prize, pattern, low class #) are simply")
print("     more predictable (favorites win more often)")
print("  2. Field sizes vary systematically with race quality")
print("  3. The model is excellent at identifying quality signals")
print("\nIf race-level features vary within races, that's DATA LEAKAGE.")
print("\nCheck the output above for 'ERROR - LEAKAGE!' warnings.")
