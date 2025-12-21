"""
Phase 1, Task 1.3: Data Cleaning and Standardization
Standardizes column values for consistent analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
INPUT_FILE = DATA_DIR / 'processed' / 'all_gb_races.parquet'
OUTPUT_FILE = DATA_DIR / 'processed' / 'all_gb_races_cleaned.parquet'


def clean_distance(df):
    """Convert distance strings (e.g., '6f', '12.5f') to numeric furlongs."""
    print("Cleaning distance values...")
    
    # Extract numeric value from string like "6f" or "12.5f"
    df['dist_f_clean'] = df['dist_f'].str.replace('f', '').astype(float)
    
    # Keep original for reference
    df['dist_f_original'] = df['dist_f']
    
    print(f"  Converted {len(df)} distance values")
    print(f"  Range: {df['dist_f_clean'].min():.1f}f to {df['dist_f_clean'].max():.1f}f")
    
    return df


def clean_course_names(df):
    """Standardize course names and extract surface indicator."""
    print("\nCleaning course names...")
    
    # Create clean course name (remove AW indicator)
    df['course_clean'] = df['course'].str.replace(' (AW)', '', regex=False)
    
    # Extract whether course has AW indicator
    df['is_aw_course'] = df['course'].str.contains('(AW)', regex=False)
    
    # Keep original for reference
    df['course_original'] = df['course']
    
    print(f"  Found {df['course_clean'].nunique()} unique courses")
    print(f"  AW courses: {df['is_aw_course'].sum():,} records")
    print(f"  Turf courses: {(~df['is_aw_course']).sum():,} records")
    
    return df


def clean_position(df):
    """Convert position to numeric, handling non-finishers."""
    print("\nCleaning position values...")
    
    # Store original
    df['pos_original'] = df['pos']
    
    # Convert to numeric (non-finishers become NaN)
    df['pos_clean'] = pd.to_numeric(df['pos'], errors='coerce')
    
    # Create finish status column
    df['finish_status'] = 'completed'
    df.loc[df['pos_clean'].isna(), 'finish_status'] = df.loc[df['pos_clean'].isna(), 'pos']
    
    # Count non-finishers
    non_finishers = df['pos_clean'].isna().sum()
    finishers = (~df['pos_clean'].isna()).sum()
    
    print(f"  Finishers: {finishers:,} ({finishers/len(df)*100:.1f}%)")
    print(f"  Non-finishers: {non_finishers:,} ({non_finishers/len(df)*100:.1f}%)")
    
    if non_finishers > 0:
        print(f"  Non-finisher codes: {df[df['pos_clean'].isna()]['pos'].value_counts().head(10).to_dict()}")
    
    return df


def clean_class(df):
    """Ensure class labels are in consistent format."""
    print("\nCleaning class values...")
    
    # Check current format
    unique_classes = df['class'].unique()
    print(f"  Current class values: {sorted([str(c) for c in unique_classes])}")
    
    # Store original
    df['class_original'] = df['class']
    
    # Standardize to "Class X" format if needed
    # (appears already in correct format based on validation output)
    df['class_clean'] = df['class']
    
    print(f"  All classes standardized to: {sorted(df['class_clean'].unique())}")
    
    return df


def clean_surface(df):
    """Standardize surface values and fill missing."""
    print("\nCleaning surface values...")
    
    # Store original
    df['surface_original'] = df['surface']
    
    # Standardize naming
    df['surface_clean'] = df['surface'].replace({
        'AW': 'All-Weather',
        'Turf': 'Turf'
    })
    
    # Fill missing surface based on course AW indicator
    missing_surface = df['surface_clean'].isna()
    if missing_surface.any():
        print(f"  Found {missing_surface.sum()} missing surface values")
        
        # Use course AW indicator to fill
        df.loc[missing_surface & df['is_aw_course'], 'surface_clean'] = 'All-Weather'
        df.loc[missing_surface & ~df['is_aw_course'], 'surface_clean'] = 'Turf'
        
        print(f"  Filled missing values using course information")
    
    surface_counts = df['surface_clean'].value_counts()
    for surf, count in surface_counts.items():
        print(f"  {surf}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def add_distance_bands(df):
    """Add distance band categorization."""
    print("\nAdding distance bands...")
    
    conditions = [
        df['dist_f_clean'] < 7,
        (df['dist_f_clean'] >= 7) & (df['dist_f_clean'] < 9),
        (df['dist_f_clean'] >= 9) & (df['dist_f_clean'] < 13),
        df['dist_f_clean'] >= 13
    ]
    
    choices = ['Sprint', 'Mile', 'Middle', 'Long']
    
    df['distance_band'] = np.select(conditions, choices, default='Unknown')
    
    band_counts = df['distance_band'].value_counts()
    for band, count in band_counts.items():
        print(f"  {band}: {count:,}")
    
    return df


def clean_prize(df):
    """Convert prize to numeric."""
    print("\nCleaning prize values...")
    
    df['prize_original'] = df['prize']
    df['prize_clean'] = pd.to_numeric(df['prize'], errors='coerce')
    
    missing_prize = df['prize_clean'].isna().sum()
    valid_prize = (~df['prize_clean'].isna()).sum()
    
    print(f"  Valid prize values: {valid_prize:,} ({valid_prize/len(df)*100:.1f}%)")
    print(f"  Missing prize values: {missing_prize:,} ({missing_prize/len(df)*100:.1f}%)")
    
    if valid_prize > 0:
        print(f"  Prize range: £{df['prize_clean'].min():,.0f} to £{df['prize_clean'].max():,.0f}")
        print(f"  Median prize: £{df['prize_clean'].median():,.0f}")
    
    return df


def add_race_quality_tier(df):
    """Add race quality tier based on class and prize."""
    print("\nAdding race quality tiers...")
    
    # Tier based on class
    tier_map = {
        'Class 1': 'Premium',
        'Class 2': 'High',
        'Class 3': 'Standard',
        'Class 4': 'Standard',
        'Class 5': 'Low',
        'Class 6': 'Low',
        'Class 7': 'Low'
    }
    
    df['quality_tier'] = df['class_clean'].map(tier_map)
    
    tier_counts = df['quality_tier'].value_counts()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,}")
    
    return df


def main():
    """Main cleaning workflow."""
    print("=" * 60)
    print("PHASE 1: DATA CLEANING")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Apply cleaning functions
    df = clean_distance(df)
    df = clean_course_names(df)
    df = clean_position(df)
    df = clean_class(df)
    df = clean_surface(df)
    df = add_distance_bands(df)
    df = clean_prize(df)
    df = add_race_quality_tier(df)
    
    # Save cleaned data
    print(f"\n{'=' * 60}")
    print("SAVING CLEANED DATA")
    print(f"{'=' * 60}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n[SAVED] Cleaned data saved to: {OUTPUT_FILE}")
    
    # Summary of new columns
    print(f"\n{'=' * 60}")
    print("NEW COLUMNS ADDED")
    print(f"{'=' * 60}")
    new_cols = [
        'dist_f_clean', 'course_clean', 'is_aw_course', 'pos_clean', 
        'finish_status', 'class_clean', 'surface_clean', 'distance_band',
        'prize_clean', 'quality_tier'
    ]
    for col in new_cols:
        print(f"  - {col}")
    
    print(f"\n{'=' * 60}")
    print("CLEANING COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("1. Create lookup tables (Task 1.4)")
    print("2. Build aggregation scripts (Tasks 1.5-1.6)")
    print("3. Run exploratory analysis (Task 1.7)")


if __name__ == "__main__":
    main()
