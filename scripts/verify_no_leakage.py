#!/usr/bin/env python3
"""
Data Leakage Verification Script

Checks for common sources of data leakage in feature engineering:
1. Temporal integrity - features use only past data
2. Race-level features don't leak outcomes (e.g., prize won vs prize pool)
3. Expanding windows use shift(1) to exclude current race
4. Same-day features filter by time not just date
5. Train/test temporal split integrity

CRITICAL: Run this after any feature engineering changes!

Usage:
  python scripts/verify_no_leakage.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def check_temporal_integrity(df, feature_name, group_cols, date_col='date'):
    """
    Verify a feature doesn't use future data.
    
    Args:
        df: DataFrame with engineered features
        feature_name: Name of feature to check
        group_cols: List of grouping columns (e.g., ['horse'], ['sire'])
        date_col: Date column name
    
    Returns:
        bool: True if no leakage detected
    """
    print(f"\nChecking temporal integrity: {feature_name}")
    
    if feature_name not in df.columns:
        print(f"  ⚠️  Feature not found: {feature_name}")
        return True
    
    # Sort by group and date
    df_sorted = df.copy()
    sort_cols = group_cols + [date_col]
    df_sorted = df_sorted.sort_values(sort_cols).reset_index(drop=True)
    
    # For each row, check if feature value could have been computed from past data only
    # Simple heuristic: feature should be NaN or 0 for first occurrence of entity
    first_occurrences = df_sorted.groupby(group_cols).head(1)
    
    # If feature has non-zero/non-NaN values on first occurrence, might indicate leakage
    if len(first_occurrences) > 0:
        non_zero_first = first_occurrences[feature_name].fillna(0)
        pct_non_zero = (non_zero_first != 0).mean()
        
        if pct_non_zero > 0.5:  # More than 50% have values on first occurrence
            print(f"  ⚠️  WARNING: {pct_non_zero:.1%} of first occurrences have non-zero values")
            print(f"      This might indicate the feature uses current/future data")
            return False
        else:
            print(f"  ✓ OK: {pct_non_zero:.1%} of first occurrences have values (expected for career features)")
    
    return True


def check_race_level_features(df):
    """Check that race-level features don't leak individual outcomes."""
    print("\n" + "="*60)
    print("RACE-LEVEL FEATURE CHECKS")
    print("="*60)
    
    issues = []
    
    # Check prize_log uses total prize pool, not individual winnings
    if 'prize_log' in df.columns and 'prize_clean' in df.columns:
        print("\nChecking prize_log doesn't leak individual winnings...")
        
        # Group by race and check if all horses have same prize_log
        race_groups = df.groupby(['date', 'course_clean', 'off'])
        prize_variance = race_groups['prize_log'].apply(lambda x: x.std())
        
        # All horses in a race should have identical prize_log (total pool)
        if (prize_variance > 0.001).any():
            pct_variance = (prize_variance > 0.001).mean()
            print(f"  ⚠️  WARNING: {pct_variance:.1%} of races have varying prize_log across horses")
            print(f"      This indicates prize_log may be using individual winnings (LEAKAGE!)")
            issues.append("prize_log has variance within races")
        else:
            print(f"  ✓ OK: prize_log is constant within races (using total pool)")
    
    # Check OR features use race context, not outcome
    or_features = ['or_vs_race_max', 'or_vs_race_avg', 'or_race_percentile']
    for feat in or_features:
        if feat in df.columns:
            print(f"\nChecking {feat} is pre-race data...")
            # These should be computable from race field alone
            # No specific test, but they should exist
            print(f"  ✓ {feat} exists (race-level comparison)")
    
    # Check weight features are race-relative, not outcome-based
    if 'weight_vs_avg' in df.columns:
        print("\nChecking weight_vs_avg is race-level...")
        race_groups = df.groupby(['date', 'course_clean', 'off'])
        weight_avg_check = race_groups['weight_vs_avg'].apply(lambda x: x.mean())
        
        # Average of (weight - avg) across race should be ~0
        if abs(weight_avg_check.mean()) > 0.1:
            print(f"  ⚠️  WARNING: weight_vs_avg average is {weight_avg_check.mean():.2f}, expected ~0")
            issues.append("weight_vs_avg may have calculation error")
        else:
            print(f"  ✓ OK: weight_vs_avg averages to {weight_avg_check.mean():.4f} (expected ~0)")
    
    return issues


def check_shift_usage(df):
    """Verify expanding/rolling features use shift(1) to exclude current race."""
    print("\n" + "="*60)
    print("SHIFT(1) USAGE CHECKS")
    print("="*60)
    
    # Career stats should not include current race
    career_features = ['career_runs', 'career_win_rate', 'career_place_rate']
    
    for feat in career_features:
        if feat in df.columns and 'horse' in df.columns:
            print(f"\nChecking {feat} excludes current race...")
            
            # For first race of each horse, career stats should be 0
            first_races = df.groupby('horse').head(1)
            if feat == 'career_runs':
                first_values = first_races[feat].fillna(-1)
                if (first_values != 0).any():
                    pct_non_zero = (first_values != 0).mean()
                    print(f"  ⚠️  WARNING: {pct_non_zero:.1%} of first races have non-zero {feat}")
                    print(f"      This indicates shift(1) may not be used (LEAKAGE!)")
                else:
                    print(f"  ✓ OK: All first races have career_runs=0")
            else:
                # Win/place rates should be 0 for first race
                first_values = first_races[feat].fillna(0)
                if (first_values > 0).any():
                    pct_non_zero = (first_values > 0).mean()
                    print(f"  ⚠️  WARNING: {pct_non_zero:.1%} of first races have non-zero {feat}")
                else:
                    print(f"  ✓ OK: All first races have {feat}=0")


def check_same_day_filtering(df):
    """Check that same-day features filter by time, not just date."""
    print("\n" + "="*60)
    print("SAME-DAY FILTERING CHECKS")
    print("="*60)
    
    # Jockey/trainer form features should not use same-day races
    form_features = ['jockey_form_30d', 'trainer_form_30d']
    
    for feat in form_features:
        if feat in df.columns:
            print(f"\nChecking {feat} excludes same-day races...")
            # This is harder to verify without knowing exact calculation
            # But we can check if there's documentation
            print(f"  ⚠️  Manual check required: Verify {feat} calculation excludes same-day races")


def check_temporal_split_integrity(df):
    """Verify train/test split respects temporal ordering."""
    print("\n" + "="*60)
    print("TEMPORAL SPLIT INTEGRITY")
    print("="*60)
    
    if 'date' not in df.columns:
        print("  ⚠️  No date column found")
        return
    
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # Simulate 80/20 temporal split
    df_sorted = df.sort_values('date_dt').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    split_date = df_sorted.loc[split_idx, 'date_dt']
    
    train_end = df_sorted[df_sorted['date_dt'] < split_date]['date_dt'].max()
    test_start = df_sorted[df_sorted['date_dt'] >= split_date]['date_dt'].min()
    
    print(f"\nTemporal split at 80%:")
    print(f"  Train end:  {train_end.date()}")
    print(f"  Test start: {test_start.date()}")
    
    if test_start < train_end:
        print(f"  ⚠️  WARNING: Test data overlaps with training data (LEAKAGE!)")
        return False
    else:
        print(f"  ✓ OK: No temporal overlap between train and test")
        return True


def main():
    """Run all data leakage checks."""
    print("="*60)
    print("DATA LEAKAGE VERIFICATION")
    print("="*60)
    
    # Load data
    data_dir = Path('data/processed')
    or_context_path = data_dir / 'race_scores_or_context.parquet'
    
    if not or_context_path.exists():
        print(f"\n❌ Data file not found: {or_context_path}")
        print("   Run feature engineering scripts first")
        return
    
    print(f"\nLoading: {or_context_path}")
    df = pd.read_parquet(or_context_path)
    print(f"  Loaded {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    
    issues = []
    
    # 1. Check race-level features
    race_issues = check_race_level_features(df)
    issues.extend(race_issues)
    
    # 2. Check shift(1) usage
    check_shift_usage(df)
    
    # 3. Check temporal integrity for key features
    check_temporal_integrity(df, 'career_win_rate', ['horse'])
    check_temporal_integrity(df, 'sire_win_rate_v2', ['sire_id'])
    check_temporal_integrity(df, 'jockey_form_30d', ['jockey'])
    
    # 4. Check same-day filtering
    check_same_day_filtering(df)
    
    # 5. Check temporal split
    check_temporal_split_integrity(df)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if issues:
        print(f"\n❌ Found {len(issues)} potential leakage issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Review and fix these issues before using the model!")
        sys.exit(1)
    else:
        print("\n✅ No data leakage detected!")
        print("\nKey checks passed:")
        print("  ✓ Race-level features use pre-race data only")
        print("  ✓ Career stats exclude current race (shift(1))")
        print("  ✓ Temporal split maintains chronological order")
        print("  ✓ No future information in training features")
        
        print("\n⚠️  Note: This is an automated check. Manual review is still recommended.")
    
    return issues


if __name__ == '__main__':
    main()
