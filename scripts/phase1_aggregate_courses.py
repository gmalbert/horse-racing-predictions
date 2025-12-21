"""
Phase 1, Task 1.5: Course-Level Aggregation
Calculates historical statistics for each course to inform race profitability scoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
INPUT_FILE = DATA_DIR / 'processed' / 'all_gb_races_cleaned.parquet'
OUTPUT_FILE = DATA_DIR / 'processed' / 'aggregates' / 'course_statistics.csv'
LOOKUPS_DIR = DATA_DIR / 'processed' / 'lookups'

# Create aggregates directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def calculate_favorite_stats(df):
    """Calculate favorite performance statistics."""
    print("Calculating favorite statistics...")
    
    # Identify favorites (assuming 'fav' column or lowest odds)
    if 'fav' in df.columns:
        # Use existing favorite flag
        favorites = df[df['fav'] == True].copy()
    elif 'or' in df.columns:
        # Use highest official rating as proxy for favorite
        favorites = df.loc[df.groupby('race_id')['or'].idxmax()].copy()
    else:
        print("  [WARNING] No favorite indicator found - skipping favorite stats")
        return None
    
    # Calculate favorite win rate by course
    fav_stats = favorites.groupby('course_clean').agg({
        'race_id': 'count',  # Total favorites
        'pos_clean': lambda x: (x == 1).sum()  # Favorite wins
    }).reset_index()
    
    fav_stats.columns = ['course', 'total_favorites', 'favorite_wins']
    fav_stats['favorite_win_rate'] = (fav_stats['favorite_wins'] / fav_stats['total_favorites'] * 100).round(2)
    
    print(f"  Calculated favorite stats for {len(fav_stats)} courses")
    print(f"  Average favorite win rate: {fav_stats['favorite_win_rate'].mean():.1f}%")
    
    return fav_stats


def aggregate_by_course(df):
    """Aggregate race-level statistics by course."""
    print("\nAggregating by course...")
    
    # Get unique races only (one row per race, not per horse)
    races = df.drop_duplicates(subset='race_id').copy()
    
    print(f"  Working with {len(races):,} unique races")
    
    # Aggregate statistics
    course_agg = races.groupby('course_clean').agg({
        'race_id': 'count',  # Total races
        'prize_clean': ['mean', 'median', 'sum', 'count'],  # Prize statistics
        'ran': ['mean', 'median', 'min', 'max'],  # Field size statistics
        'date': ['min', 'max'],  # Date range
        'surface_clean': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',  # Majority surface
        'distance_band': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'  # Most common distance
    }).reset_index()
    
    # Flatten multi-level columns
    course_agg.columns = [
        'course', 'total_races', 
        'avg_prize', 'median_prize', 'total_prize_pool', 'races_with_prize',
        'avg_field_size', 'median_field_size', 'min_field_size', 'max_field_size',
        'first_race_date', 'last_race_date',
        'primary_surface', 'most_common_distance_band'
    ]
    
    # Calculate data coverage (% races with prize data)
    course_agg['prize_data_coverage'] = (course_agg['races_with_prize'] / course_agg['total_races'] * 100).round(1)
    
    # Count races by class
    class_dist = races.groupby(['course_clean', 'class_clean']).size().unstack(fill_value=0)
    class_dist = class_dist.add_prefix('races_')
    course_agg = course_agg.merge(class_dist, left_on='course', right_index=True, how='left')
    
    # Calculate high-quality race count (Class 1-2)
    course_agg['high_quality_races'] = (
        course_agg.get('races_Class 1', 0) + 
        course_agg.get('races_Class 2', 0)
    )
    
    print(f"  Aggregated statistics for {len(course_agg)} courses")
    
    return course_agg


def add_course_tier(course_stats):
    """Add course tier classification from lookup table."""
    print("\nAdding course tier classification...")
    
    # Load course tiers lookup
    tiers_file = LOOKUPS_DIR / 'course_tiers.csv'
    if tiers_file.exists():
        tiers = pd.read_csv(tiers_file)[['course', 'tier']]
        course_stats = course_stats.merge(tiers, on='course', how='left')
        print(f"  Added tier classification")
    else:
        print(f"  [WARNING] Course tiers lookup not found at {tiers_file}")
        course_stats['tier'] = 'Unknown'
    
    return course_stats


def calculate_predictability_metrics(df, course_stats):
    """Calculate predictability metrics for each course."""
    print("\nCalculating predictability metrics...")
    
    # Favorite performance (if available)
    fav_stats = calculate_favorite_stats(df)
    if fav_stats is not None:
        course_stats = course_stats.merge(
            fav_stats[['course', 'favorite_win_rate']], 
            on='course', 
            how='left'
        )
    
    # Field size consistency (lower std = more predictable)
    races = df.drop_duplicates(subset='race_id')
    field_size_std = races.groupby('course_clean')['ran'].std().reset_index()
    field_size_std.columns = ['course', 'field_size_std']
    course_stats = course_stats.merge(field_size_std, on='course', how='left')
    
    print(f"  Added predictability metrics")
    
    return course_stats


def main():
    """Main aggregation workflow."""
    print("=" * 60)
    print("PHASE 1: COURSE-LEVEL AGGREGATION")
    print("=" * 60)
    
    # Load cleaned data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} records")
    
    # Run aggregations
    course_stats = aggregate_by_course(df)
    course_stats = add_course_tier(course_stats)
    course_stats = calculate_predictability_metrics(df, course_stats)
    
    # Sort by total races (most active courses first)
    course_stats = course_stats.sort_values('total_races', ascending=False)
    
    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING AGGREGATED STATISTICS")
    print(f"{'=' * 60}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Courses analyzed: {len(course_stats)}")
    print(f"Columns: {len(course_stats.columns)}")
    
    course_stats.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[SAVED] Course statistics saved to: {OUTPUT_FILE}")
    
    # Display summary
    print(f"\n{'=' * 60}")
    print("COURSE STATISTICS SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nTop 10 courses by race volume:")
    top_courses = course_stats.head(10)[['course', 'total_races', 'avg_prize', 'median_field_size', 'tier']]
    print(top_courses.to_string(index=False))
    
    print(f"\n{'=' * 60}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("1. Build distance/class aggregation (Task 1.6)")
    print("2. Run exploratory analysis (Task 1.7)")


if __name__ == "__main__":
    main()
