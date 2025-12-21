"""
Phase 1, Task 1.1: Data Validation and Cleaning
Validates the historical database and identifies data quality issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PARQUET_FILE = DATA_DIR / 'processed' / 'all_gb_races.parquet'
OUTPUT_DIR = DATA_DIR / 'processed'


def load_data():
    """Load the historical race database."""
    print(f"Loading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df):,} races")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {len(df.columns)}")
    return df


def check_duplicates(df):
    """Check for duplicate race records."""
    print("\n=== DUPLICATE CHECK ===")
    
    # Check for exact duplicates
    exact_dupes = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_dupes:,}")
    
    # Check for duplicate races (same date, course, horse, position)
    race_dupes = df.duplicated(subset=['date', 'course', 'horse', 'pos']).sum()
    print(f"Duplicate race entries (same date/course/horse/pos): {race_dupes:,}")
    
    # Check for duplicate race IDs
    if 'race_id' in df.columns:
        race_id_dupes = df['race_id'].duplicated().sum()
        print(f"Duplicate race_ids: {race_id_dupes:,}")
    
    return exact_dupes, race_dupes


def check_missing_values(df):
    """Analyze missing values in critical columns."""
    print("\n=== MISSING VALUES ===")
    
    critical_cols = ['date', 'course', 'horse', 'pos', 'dist_f', 'class', 'surface']
    
    missing_summary = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_count > 0:
            missing_summary.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'critical': col in critical_cols
            })
    
    missing_df = pd.DataFrame(missing_summary).sort_values('missing_pct', ascending=False)
    
    print("\nColumns with missing values:")
    print(missing_df.to_string(index=False))
    
    # Check critical columns
    critical_missing = missing_df[missing_df['critical']]
    if not critical_missing.empty:
        print(f"\n[WARNING] Critical columns have missing values:")
        print(critical_missing[['column', 'missing_count', 'missing_pct']].to_string(index=False))
    else:
        print("\n[OK] All critical columns have complete data")
    
    return missing_df


def analyze_column_values(df):
    """Analyze unique values and distributions in key columns."""
    print("\n=== COLUMN VALUE ANALYSIS ===")
    
    # Course names
    print(f"\nCourses: {df['course'].nunique()} unique")
    print("Top 10 courses by race count:")
    top_courses = df['course'].value_counts().head(10)
    for course, count in top_courses.items():
        print(f"  {course}: {count:,}")
    
    # Check for inconsistent course names (case, spacing, etc.)
    courses = df['course'].unique()
    print(f"\nSample course names: {list(courses[:10])}")
    
    # Class values
    print(f"\n\nClass: {df['class'].nunique()} unique values")
    print("Class distribution:")
    class_dist = df['class'].value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"  {cls}: {count:,}")
    
    # Surface values
    print(f"\n\nSurface: {df['surface'].nunique()} unique values")
    print("Top 10 surface types:")
    surface_counts = df['surface'].value_counts().head(10)
    for surf, count in surface_counts.items():
        print(f"  {surf}: {count:,}")
    
    # Distance (check for anomalies)
    print(f"\n\nDistance (furlongs):")
    print(f"  Data type: {df['dist_f'].dtype}")
    print(f"  Unique distances: {df['dist_f'].nunique()}")
    print(f"  Sample values: {', '.join(str(x) for x in df['dist_f'].unique()[:10])}")
    
    # Try to convert string format to numeric (remove 'f' suffix)
    try:
        dist_numeric = df['dist_f'].str.replace('f', '').astype(float)
        print(f"  Min: {dist_numeric.min()}")
        print(f"  Max: {dist_numeric.max()}")
        print(f"  Mean: {dist_numeric.mean():.2f}")
        print(f"  Median: {dist_numeric.median()}")
        
        # Check for outliers
        outlier_distances = dist_numeric[dist_numeric < 4]
        if len(outlier_distances) > 0:
            print(f"  [!] {len(outlier_distances)} races with distance < 4f")
        
        outlier_distances = dist_numeric[dist_numeric > 20]
        if len(outlier_distances) > 0:
            print(f"  [!] {len(outlier_distances)} races with distance > 20f")
    except Exception as e:
        print(f"  [!] Could not analyze distance statistics: {e}")
    
    # Prize money
    if 'prize' in df.columns:
        print(f"\n\nPrize Money:")
        df['prize_numeric'] = pd.to_numeric(df['prize'], errors='coerce')
        print(f"  Min: £{df['prize_numeric'].min():,.0f}")
        print(f"  Max: £{df['prize_numeric'].max():,.0f}")
        print(f"  Mean: £{df['prize_numeric'].mean():,.0f}")
        print(f"  Median: £{df['prize_numeric'].median():,.0f}")
    
    # Field size
    if 'ran' in df.columns:
        print(f"\n\nField Size (runners):")
        print(f"  Min: {df['ran'].min()}")
        print(f"  Max: {df['ran'].max()}")
        print(f"  Mean: {df['ran'].mean():.2f}")
        print(f"  Median: {df['ran'].median()}")
        
        # Check for anomalies
        single_runner = df[df['ran'] == 1]
        if len(single_runner) > 0:
            print(f"  [!] {len(single_runner)} races with only 1 runner (walkovers)")
        
        large_field = df[df['ran'] > 30]
        if len(large_field) > 0:
            print(f"  [!] {len(large_field)} races with >30 runners")


def check_data_consistency(df):
    """Check for logical inconsistencies in the data."""
    print("\n=== DATA CONSISTENCY CHECKS ===")
    
    issues = []
    
    # Check: finish position should not exceed field size
    if 'pos' in df.columns and 'ran' in df.columns:
        # Convert pos to numeric (may be string like "1", "2", "PU", "F", etc.)
        df_check = df.copy()
        df_check['pos_numeric'] = pd.to_numeric(df_check['pos'], errors='coerce')
        df_check['ran_numeric'] = pd.to_numeric(df_check['ran'], errors='coerce')
        
        # Only check where both are numeric
        valid_check = df_check.dropna(subset=['pos_numeric', 'ran_numeric'])
        invalid_positions = valid_check[valid_check['pos_numeric'] > valid_check['ran_numeric']]
        if len(invalid_positions) > 0:
            issues.append(f"[!] {len(invalid_positions)} races where finish position > field size")
            print(issues[-1])
        else:
            print("[OK] All finish positions are valid (pos <= field size)")
    
    # Check: dates in the future
    today = pd.Timestamp.now()
    future_races = df[df['date'] > today]
    if len(future_races) > 0:
        issues.append(f"[!] {len(future_races)} races with dates in the future (2025 fixtures)")
        print(issues[-1])
    else:
        print("[OK] No races with future dates")
    
    # Check: negative or zero prize money
    if 'prize' in df.columns:
        df['prize_numeric'] = pd.to_numeric(df['prize'], errors='coerce')
        invalid_prize = df[(df['prize_numeric'] <= 0) & (df['prize_numeric'].notna())]
        if len(invalid_prize) > 0:
            issues.append(f"[!] {len(invalid_prize)} races with prize <= £0")
            print(issues[-1])
        else:
            print("[OK] All prize money values are positive")
    
    # Check: Official Rating anomalies
    if 'or' in df.columns:
        df['or_numeric'] = pd.to_numeric(df['or'], errors='coerce')
        low_or = df[df['or_numeric'] < 30]  # OR typically 40-130
        high_or = df[df['or_numeric'] > 140]
        
        if len(low_or) > 0:
            issues.append(f"[!] {len(low_or)} entries with OR < 30")
            print(issues[-1])
        if len(high_or) > 0:
            issues.append(f"[!] {len(high_or)} entries with OR > 140")
            print(issues[-1])
    
    if not issues:
        print("[OK] No major consistency issues found")
    
    return issues


def generate_summary_report(df):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("DATA VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal Records: {len(df):,}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique Horses: {df['horse'].nunique():,}")
    print(f"Unique Courses: {df['course'].nunique()}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory Usage: {memory_mb:.2f} MB")
    
    # Data types
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())


def main():
    """Run all validation checks."""
    print("="*60)
    print("PHASE 1: DATA VALIDATION")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run checks
    check_duplicates(df)
    missing_df = check_missing_values(df)
    analyze_column_values(df)
    issues = check_data_consistency(df)
    generate_summary_report(df)
    
    # Save missing values report
    if not missing_df.empty:
        report_path = OUTPUT_DIR / 'data_quality_report.csv'
        missing_df.to_csv(report_path, index=False)
        print(f"\n[SAVED] Missing values report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review any warnings above")
    print("2. Run phase1_data_cleaning.py to standardize values")
    print("3. Proceed to aggregation scripts (Task 1.2)")


if __name__ == '__main__':
    main()
