"""
Phase 1, Task 1.6: Distance/Class Aggregation
Calculates statistics by distance band and class combination.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
INPUT_FILE = DATA_DIR / 'processed' / 'all_gb_races_cleaned.parquet'
OUTPUT_FILE = DATA_DIR / 'processed' / 'aggregates' / 'distance_class_statistics.csv'

# Create aggregates directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def aggregate_by_distance_class(df):
    """Aggregate statistics by distance band and class."""
    print("Aggregating by distance band and class...")
    
    # Get unique races only
    races = df.drop_duplicates(subset='race_id').copy()
    
    print(f"  Working with {len(races):,} unique races")
    
    # Aggregate by distance_band and class
    agg_stats = races.groupby(['distance_band', 'class_clean']).agg({
        'race_id': 'count',  # Total races
        'prize_clean': ['mean', 'median', 'count'],  # Prize statistics
        'ran': ['mean', 'median'],  # Field size statistics
        'surface_clean': lambda x: (x == 'Turf').sum() / len(x) * 100  # % Turf races
    }).reset_index()
    
    # Flatten columns
    agg_stats.columns = [
        'distance_band', 'class', 'total_races',
        'avg_prize', 'median_prize', 'races_with_prize',
        'avg_field_size', 'median_field_size',
        'pct_turf_races'
    ]
    
    print(f"  Created {len(agg_stats)} distance/class combinations")
    
    return agg_stats


def calculate_favorite_performance(df):
    """Calculate favorite performance by distance/class."""
    print("\nCalculating favorite performance by distance/class...")
    
    if 'fav' not in df.columns:
        print("  [WARNING] No favorite indicator - skipping favorite stats")
        return None
    
    # Filter to favorites only
    favorites = df[df['fav'] == True].copy()
    
    # Calculate win rate and average position
    fav_stats = favorites.groupby(['distance_band', 'class_clean']).agg({
        'pos_clean': ['count', lambda x: (x == 1).sum(), 'mean', 'median']
    }).reset_index()
    
    fav_stats.columns = [
        'distance_band', 'class', 
        'fav_total', 'fav_wins', 'fav_avg_pos', 'fav_median_pos'
    ]
    
    fav_stats['fav_win_rate'] = (fav_stats['fav_wins'] / fav_stats['fav_total'] * 100).round(2)
    
    print(f"  Calculated favorite stats for {len(fav_stats)} combinations")
    
    return fav_stats


def calculate_competitive_metrics(df):
    """Calculate metrics indicating race competitiveness."""
    print("\nCalculating competitive metrics...")
    
    # Get unique races
    races = df.drop_duplicates(subset='race_id').copy()
    
    # Calculate field size variability (std dev)
    comp_metrics = races.groupby(['distance_band', 'class_clean']).agg({
        'ran': 'std'  # Field size standard deviation
    }).reset_index()
    
    comp_metrics.columns = ['distance_band', 'class', 'field_size_std']
    
    print(f"  Calculated competitive metrics")
    
    return comp_metrics


def add_surface_breakdown(df):
    """Add detailed surface breakdown."""
    print("\nAdding surface breakdown...")
    
    races = df.drop_duplicates(subset='race_id')
    
    surface_breakdown = races.groupby(['distance_band', 'class_clean', 'surface_clean']).size().unstack(fill_value=0)
    surface_breakdown = surface_breakdown.reset_index()
    
    # Rename surface columns
    if 'Turf' in surface_breakdown.columns:
        surface_breakdown = surface_breakdown.rename(columns={'Turf': 'races_turf'})
    if 'All-Weather' in surface_breakdown.columns:
        surface_breakdown = surface_breakdown.rename(columns={'All-Weather': 'races_aw'})
    
    surface_breakdown.columns = ['distance_band', 'class'] + [col for col in surface_breakdown.columns if col not in ['distance_band', 'class_clean']]
    
    print(f"  Added surface breakdown")
    
    return surface_breakdown


def main():
    """Main aggregation workflow."""
    print("=" * 60)
    print("PHASE 1: DISTANCE/CLASS AGGREGATION")
    print("=" * 60)
    
    # Load cleaned data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} records")
    
    # Run aggregations
    stats = aggregate_by_distance_class(df)
    
    # Add favorite performance
    fav_stats = calculate_favorite_performance(df)
    if fav_stats is not None:
        stats = stats.merge(fav_stats, on=['distance_band', 'class'], how='left')
    
    # Add competitive metrics
    comp_metrics = calculate_competitive_metrics(df)
    stats = stats.merge(comp_metrics, on=['distance_band', 'class'], how='left')
    
    # Add surface breakdown
    surface_breakdown = add_surface_breakdown(df)
    stats = stats.merge(surface_breakdown, on=['distance_band', 'class'], how='left')
    
    # Calculate prize data coverage
    stats['prize_data_coverage'] = (stats['races_with_prize'] / stats['total_races'] * 100).round(1)
    
    # Sort by total races
    stats = stats.sort_values('total_races', ascending=False)
    
    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING AGGREGATED STATISTICS")
    print(f"{'=' * 60}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Distance/Class combinations: {len(stats)}")
    print(f"Columns: {len(stats.columns)}")
    
    stats.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[SAVED] Distance/class statistics saved to: {OUTPUT_FILE}")
    
    # Display summary
    print(f"\n{'=' * 60}")
    print("DISTANCE/CLASS STATISTICS SUMMARY")
    print(f"{'=' * 60}")
    
    # Top combinations by race volume
    print(f"\nTop 10 combinations by race volume:")
    display_cols = ['distance_band', 'class', 'total_races', 'avg_field_size']
    if 'fav_win_rate' in stats.columns:
        display_cols.append('fav_win_rate')
    top_combos = stats.head(10)[display_cols]
    print(top_combos.to_string(index=False))
    
    # Favorite win rate by distance band
    if 'fav_win_rate' in stats.columns:
        print(f"\nFavorite win rate by distance band:")
        dist_fav = stats.groupby('distance_band')['fav_win_rate'].mean().round(1)
        for dist, rate in dist_fav.items():
            print(f"  {dist:15s}: {rate:.1f}%")
        
        print(f"\nFavorite win rate by class:")
        class_fav = stats.groupby('class')['fav_win_rate'].mean().round(1)
        for cls, rate in class_fav.items():
            print(f"  {cls:10s}: {rate:.1f}%")
    else:
        print(f"\n[NOTE] Favorite statistics not available in dataset")
    
    print(f"\n{'=' * 60}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("1. Run exploratory analysis (Task 1.7)")
    print("2. Begin Phase 2: Race Profitability Scoring")


if __name__ == "__main__":
    main()
