"""
Phase 2: Race Profitability Scorer
Scores races 0-100 based on predictability and betting value.

Scoring factors (from BETTING_STRATEGY.md):
- Premium courses (+15): Ascot, Newmarket, York, Doncaster, etc.
- Class bonus: Class 1 (+20), Class 2 (+15)
- Distance band (+10): Mile/Middle distances optimal
- Field size (+10): 8-14 runners ideal
- Pattern race (+15): Group/Listed races
- Historical favorite win rate (variable points based on stats)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
CLEANED_DATA = DATA_DIR / 'processed' / 'all_gb_races_cleaned.parquet'
COURSE_STATS = DATA_DIR / 'processed' / 'aggregates' / 'course_statistics.csv'
COURSE_TIERS = DATA_DIR / 'processed' / 'lookups' / 'course_tiers.csv'
OUTPUT_FILE = DATA_DIR / 'processed' / 'race_scores.parquet'


def load_reference_data():
    """Load course statistics and tier information."""
    print("Loading reference data...")
    
    # Load course statistics
    course_stats_df = pd.read_csv(COURSE_STATS)
    print(f"  Loaded stats for {len(course_stats_df)} courses")
    
    # Load course tiers
    course_tiers_df = pd.read_csv(COURSE_TIERS)
    print(f"  Loaded tiers for {len(course_tiers_df)} courses")
    
    return course_stats_df, course_tiers_df


def score_race_profitability(race_row, course_stats_dict, course_tiers_dict):
    """
    Score a single race 0-100 based on predictability factors.
    
    Args:
        race_row: DataFrame row containing race details
        course_stats_dict: Dict mapping course -> stats
        course_tiers_dict: Dict mapping course -> tier
    
    Returns:
        int: Score from 0-100
    """
    score = 0
    
    # Get course name
    course = race_row['course_clean']
    
    # 1. Premium course bonus (+15)
    tier = course_tiers_dict.get(course, 'Unknown')
    if tier == 'Premium':
        score += 15
    elif tier == 'Major':
        score += 5  # Smaller bonus for major courses
    
    # 2. Class bonus (Class 1: +20, Class 2: +15, Class 3: +10)
    race_class = race_row['class_clean']
    if race_class == 'Class 1':
        score += 20
    elif race_class == 'Class 2':
        score += 15
    elif race_class == 'Class 3':
        score += 10
    
    # 3. Distance band bonus (+10 for Mile/Middle)
    distance_band = race_row['distance_band']
    if distance_band in ['Mile', 'Middle']:
        score += 10
    elif distance_band == 'Sprint':
        score += 5  # Sprints somewhat predictable
    # Long distance gets 0
    
    # 4. Field size bonus (+10 for 8-14 runners)
    field_size = race_row['ran']
    if pd.notna(field_size):
        if 8 <= field_size <= 14:
            score += 10
        elif 6 <= field_size <= 16:
            score += 5  # Slightly outside ideal range
        # Very small/large fields get 0
    
    # 5. Pattern race bonus (+15)
    if pd.notna(race_row.get('pattern', np.nan)):
        score += 15
    
    # 6. Historical favorite win rate (up to +35 points)
    course_stats = course_stats_dict.get(course, {})
    fav_win_rate = course_stats.get('favorite_win_rate', None)
    
    if fav_win_rate is not None and pd.notna(fav_win_rate):
        # Scale favorite win rate to 0-35 points
        # 30% win rate = +15 points (baseline)
        # 40%+ win rate = +35 points (excellent)
        # 20% win rate = +0 points (poor)
        if fav_win_rate >= 40:
            score += 35
        elif fav_win_rate >= 35:
            score += 25
        elif fav_win_rate >= 30:
            score += 15
        elif fav_win_rate >= 25:
            score += 5
        # Below 25% gets 0 points
    else:
        # No favorite data - use baseline
        score += 10
    
    # Cap at 100
    return min(score, 100)


def score_all_races(df, course_stats_df, course_tiers_df):
    """Score all unique races in the dataset."""
    print("\nScoring all races...")
    
    # Convert stats to dictionaries for fast lookup
    course_stats_dict = course_stats_df.set_index('course').to_dict('index')
    course_tiers_dict = course_tiers_df.set_index('course')['tier'].to_dict()
    
    # Get unique races only (one score per race, not per horse)
    races = df.drop_duplicates(subset='race_id').copy()
    
    print(f"  Scoring {len(races):,} unique races...")
    
    # Apply scoring function
    races['race_score'] = races.apply(
        lambda row: score_race_profitability(row, course_stats_dict, course_tiers_dict),
        axis=1
    )
    
    # Add score percentile
    races['score_percentile'] = races['race_score'].rank(pct=True) * 100
    
    # Categorize races
    def categorize_race(score):
        if score >= 70:
            return 'Tier 1: Focus'
        elif score >= 50:
            return 'Tier 2: Value'
        else:
            return 'Tier 3: Avoid'
    
    races['race_tier'] = races['race_score'].apply(categorize_race)
    
    print(f"  Scored {len(races):,} races")
    print(f"  Score range: {races['race_score'].min():.0f} to {races['race_score'].max():.0f}")
    print(f"  Mean score: {races['race_score'].mean():.1f}")
    print(f"  Median score: {races['race_score'].median():.1f}")
    
    return races


def analyze_score_distribution(races):
    """Analyze the distribution of scores."""
    print(f"\n{'=' * 60}")
    print("RACE SCORE DISTRIBUTION")
    print(f"{'=' * 60}")
    
    # Tier distribution
    tier_counts = races['race_tier'].value_counts().sort_index()
    print(f"\nRaces by tier:")
    for tier, count in tier_counts.items():
        pct = count / len(races) * 100
        print(f"  {tier:20s}: {count:6,} ({pct:5.1f}%)")
    
    # Score deciles
    print(f"\nScore distribution (deciles):")
    deciles = races['race_score'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for pct, score in deciles.items():
        print(f"  {int(pct*100):3d}th percentile: {score:5.1f}")
    
    # Top score breakdown
    print(f"\nTop scoring races (score >= 70):")
    high_scores = races[races['race_score'] >= 70]
    if len(high_scores) > 0:
        print(f"  Total: {len(high_scores):,} races ({len(high_scores)/len(races)*100:.1f}%)")
        
        # By class
        class_dist = high_scores['class_clean'].value_counts()
        print(f"\n  By class:")
        for cls, count in class_dist.head(5).items():
            print(f"    {cls}: {count:,}")
        
        # By course
        course_dist = high_scores['course_clean'].value_counts()
        print(f"\n  By course:")
        for course, count in course_dist.head(10).items():
            print(f"    {course:20s}: {count:,}")
        
        # By distance band
        dist_dist = high_scores['distance_band'].value_counts()
        print(f"\n  By distance band:")
        for dist, count in dist_dist.items():
            print(f"    {dist:10s}: {count:,}")


def main():
    """Main scoring workflow."""
    print("=" * 60)
    print("PHASE 2: RACE PROFITABILITY SCORING")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading race data from {CLEANED_DATA}...")
    df = pd.read_parquet(CLEANED_DATA)
    print(f"Loaded {len(df):,} records")
    
    # Load reference data
    course_stats_df, course_tiers_df = load_reference_data()
    
    # Score all races
    scored_races = score_all_races(df, course_stats_df, course_tiers_df)
    
    # Analyze distribution
    analyze_score_distribution(scored_races)
    
    # Merge scores back to full dataset (all horse entries)
    print(f"\n{'=' * 60}")
    print("MERGING SCORES TO FULL DATASET")
    print(f"{'=' * 60}")
    
    # Keep only race_id and score columns
    race_scores = scored_races[['race_id', 'race_score', 'score_percentile', 'race_tier']]
    
    # Merge with full dataset
    df_with_scores = df.merge(race_scores, on='race_id', how='left')
    
    print(f"  Merged scores to {len(df_with_scores):,} records")
    
    # Save
    print(f"\nSaving scored data to {OUTPUT_FILE}...")
    df_with_scores.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n[SAVED] Scored races saved to: {OUTPUT_FILE}")
    
    print(f"\n{'=' * 60}")
    print("SCORING COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("1. Backtest scoring system on historical data")
    print("2. Validate that Tier 1 races show better favorite ROI")
    print("3. Build Phase 3: Horse-level prediction model")


if __name__ == "__main__":
    main()
