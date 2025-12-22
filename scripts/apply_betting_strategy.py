#!/usr/bin/env python3
"""Apply betting strategy tier filters to identify high-value races.

Based on BETTING_STRATEGY.md criteria:
- Tier 1: Focus Races (highest ROI potential)
- Tier 2: Value Handicaps (medium risk/reward)
- Tier 3: Avoid (low predictability)

This script evaluates both historical and upcoming races against the strategy criteria.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def apply_tier1_focus_criteria(df, strict_mode=True):
    """Apply Tier 1: Focus Races criteria.
    
    Characteristics:
    - Class 1 or Class 2 (quality holds up)
    - Pattern races (Group 1/2/3, Listed) - elite horses
    - Prize ≥ £20,000 (significant purses attract serious runners)
    - Field size 6-12 (not too small, not too large)
    - Premium courses: Ascot, Newmarket, York, Doncaster
    - Turf surface (more predictable than AW)
    - Distance: Mile to Middle (7-12f specialists)
    
    Args:
        df: DataFrame with race data
        strict_mode: If False, relax some criteria for predicted fixtures
    
    Returns:
        DataFrame with tier1_focus boolean column
    """
    df = df.copy()
    
    # Class filter
    class_match = df['class'].isin(['Class 1', 'Class 2'])
    
    # Pattern race filter (if available) - relaxed in non-strict mode
    if 'pattern' in df.columns:
        pattern_match = df['pattern'].notna() & (df['pattern'] != '')
    else:
        pattern_match = pd.Series(False, index=df.index)
    
    # Prize filter - lower threshold in non-strict mode
    prize_threshold = 20000 if strict_mode else 10000
    if 'prize' in df.columns:
        prize_match = df['prize'] >= prize_threshold
    elif 'prize_clean' in df.columns:
        prize_match = df['prize_clean'] >= prize_threshold
    else:
        prize_match = pd.Series(True, index=df.index)  # Skip if no prize data
    
    # Field size filter - more lenient in non-strict mode
    if 'ran' in df.columns:
        if strict_mode:
            field_match = (df['ran'] >= 6) & (df['ran'] <= 12)
        else:
            field_match = (df['ran'] >= 5) & (df['ran'] <= 14)  # Wider range
    else:
        field_match = pd.Series(True, index=df.index)
    
    # Premium courses filter
    premium_courses = ['Ascot', 'Newmarket', 'York', 'Doncaster']
    if 'course' in df.columns:
        course_col = 'course'
    elif 'Course' in df.columns:
        course_col = 'Course'
    else:
        course_col = 'course_clean' if 'course_clean' in df.columns else None
    
    if course_col:
        course_match = df[course_col].isin(premium_courses)
    else:
        course_match = pd.Series(True, index=df.index)
    
    # Surface filter (Turf only in strict mode, allow some AW in non-strict for winter)
    if 'surface' in df.columns:
        if strict_mode:
            surface_match = df['surface'] == 'Turf'
        else:
            surface_match = df['surface'].isin(['Turf', 'AWT'])  # Allow AW in winter
    elif 'Surface' in df.columns:
        if strict_mode:
            surface_match = df['Surface'] == 'Turf'
        else:
            surface_match = df['Surface'].isin(['Turf', 'AWT'])
    elif 'surface_clean' in df.columns:
        if strict_mode:
            surface_match = df['surface_clean'] == 'Turf'
        else:
            surface_match = df['surface_clean'].isin(['Turf', 'AWT'])
    else:
        surface_match = pd.Series(True, index=df.index)
    
    # Distance filter (7-12f = Mile to Middle)
    if 'dist_f_clean' in df.columns:
        dist_match = (df['dist_f_clean'] >= 7) & (df['dist_f_clean'] <= 12)
    elif 'dist_f' in df.columns:
        # Check if numeric
        try:
            dist_match = (pd.to_numeric(df['dist_f'], errors='coerce') >= 7) & (pd.to_numeric(df['dist_f'], errors='coerce') <= 12)
        except:
            dist_match = pd.Series(True, index=df.index)
    elif 'Distance' in df.columns:
        # Try to parse distance if it's string
        dist_match = pd.Series(True, index=df.index)
    else:
        dist_match = pd.Series(True, index=df.index)
    
    # Combine all criteria (in non-strict mode, don't require pattern races)
    if strict_mode:
        df['tier1_focus'] = (
            class_match &
            prize_match &
            field_match &
            course_match &
            surface_match &
            dist_match
        )
    else:
        # Relaxed criteria for predictions
        df['tier1_focus'] = (
            class_match &
            prize_match &
            course_match &
            dist_match
        )
    
    # Bonus points for pattern races
    df['tier1_pattern_bonus'] = pattern_match & df['tier1_focus']
    
    return df


def apply_tier2_value_criteria(df):
    """Apply Tier 2: Value Handicaps criteria.
    
    Characteristics:
    - Class 3 or Class 4
    - Handicap races
    - Field size 10-16 (large fields = favorites overbet)
    - Prize ≥ £10,000
    - Good courses: Goodwood, Haydock, Sandown (expand from premium)
    - Distance: Sprint to Mile (5-10f)
    
    Returns:
        DataFrame with tier2_value boolean column
    """
    df = df.copy()
    
    # Class filter
    class_match = df['class'].isin(['Class 3', 'Class 4'])
    
    # Handicap filter
    if 'type' in df.columns:
        handicap_match = df['type'].str.contains('Handicap', case=False, na=False)
    elif 'race_name' in df.columns:
        handicap_match = df['race_name'].str.contains('Handicap', case=False, na=False)
    else:
        handicap_match = pd.Series(True, index=df.index)  # Assume True if no type data
    
    # Field size filter (larger fields)
    if 'ran' in df.columns:
        field_match = (df['ran'] >= 10) & (df['ran'] <= 16)
    else:
        field_match = pd.Series(True, index=df.index)
    
    # Prize filter
    if 'prize' in df.columns:
        prize_match = df['prize'] >= 10000
    elif 'prize_clean' in df.columns:
        prize_match = df['prize_clean'] >= 10000
    else:
        prize_match = pd.Series(True, index=df.index)
    
    # Good courses (wider than Tier 1)
    value_courses = ['Goodwood', 'Haydock', 'Sandown', 'Chester', 'Newbury', 
                     'Salisbury', 'Thirsk', 'Ripon']
    
    if 'course' in df.columns:
        course_col = 'course'
    elif 'Course' in df.columns:
        course_col = 'Course'
    else:
        course_col = 'course_clean' if 'course_clean' in df.columns else None
    
    if course_col:
        course_match = df[course_col].isin(value_courses)
    else:
        course_match = pd.Series(True, index=df.index)
    
    # Distance filter (Sprint to Mile: 5-10f)
    if 'dist_f_clean' in df.columns:
        dist_match = (df['dist_f_clean'] >= 5) & (df['dist_f_clean'] <= 10)
    elif 'dist_f' in df.columns:
        try:
            dist_match = (pd.to_numeric(df['dist_f'], errors='coerce') >= 5) & (pd.to_numeric(df['dist_f'], errors='coerce') <= 10)
        except:
            dist_match = pd.Series(True, index=df.index)
    else:
        dist_match = pd.Series(True, index=df.index)
    
    # Combine all criteria
    df['tier2_value'] = (
        class_match &
        handicap_match &
        field_match &
        prize_match &
        course_match &
        dist_match
    )
    
    return df


def apply_tier3_avoid_criteria(df):
    """Apply Tier 3: Avoid criteria.
    
    Characteristics to AVOID:
    - Class 5, 6, 7 (weak form)
    - All-Weather surface (unless winter specialist angle)
    - Apprentice or Amateur races (jockey inexperience)
    - Field size 1-4 (walk-overs or no value)
    - Maiden races for 3yo+ (poor horses past their prime)
    
    Returns:
        DataFrame with tier3_avoid boolean column
    """
    df = df.copy()
    
    # Class filter (Classes to avoid)
    class_avoid = df['class'].isin(['Class 5', 'Class 6', 'Class 7'])
    
    # Surface filter (All-Weather to avoid)
    if 'surface' in df.columns:
        surface_avoid = df['surface'].isin(['AWT', 'All-Weather'])
    elif 'Surface' in df.columns:
        surface_avoid = df['Surface'].isin(['AWT', 'All-Weather'])
    elif 'surface_clean' in df.columns:
        surface_avoid = df['surface_clean'].isin(['AWT', 'All-Weather'])
    else:
        surface_avoid = pd.Series(False, index=df.index)
    
    # Apprentice/Amateur races
    if 'race_name' in df.columns:
        apprentice_avoid = df['race_name'].str.contains('Apprentice|Amateur', case=False, na=False)
    else:
        apprentice_avoid = pd.Series(False, index=df.index)
    
    # Small field size
    if 'ran' in df.columns:
        small_field = df['ran'] <= 4
    else:
        small_field = pd.Series(False, index=df.index)
    
    # Old maiden races (3yo+ maidens are weak)
    if 'race_name' in df.columns and 'age_band' in df.columns:
        old_maiden = (
            df['race_name'].str.contains('Maiden', case=False, na=False) &
            df['age_band'].str.contains('3yo|4yo|5yo', case=False, na=False)
        )
    elif 'race_name' in df.columns:
        old_maiden = df['race_name'].str.contains('Maiden.*[3-5]yo', case=False, na=False)
    else:
        old_maiden = pd.Series(False, index=df.index)
    
    # Combine (any of these criteria = avoid)
    df['tier3_avoid'] = (
        class_avoid |
        surface_avoid |
        apprentice_avoid |
        small_field |
        old_maiden
    )
    
    return df


def classify_race_betting_tier(df, strict_mode=True):
    """Classify races into betting strategy tiers.
    
    Priority:
    1. Tier 1 (Focus) - highest priority
    2. Tier 2 (Value) - medium priority
    3. Tier 3 (Avoid) - lowest priority
    4. Unclassified - doesn't fit any tier
    
    Args:
        df: DataFrame with race data
        strict_mode: Use strict criteria (True for historical, False for predictions)
    
    Returns:
        DataFrame with betting_tier column
    """
    df = df.copy()
    
    # Apply all tier criteria
    df = apply_tier1_focus_criteria(df, strict_mode=strict_mode)
    df = apply_tier2_value_criteria(df)
    df = apply_tier3_avoid_criteria(df)
    
    # Classify (priority order: T1 > T2 > T3 > Other)
    conditions = [
        df['tier1_focus'],
        df['tier2_value'],
        df['tier3_avoid']
    ]
    
    choices = [
        'Tier 1: Focus',
        'Tier 2: Value',
        'Tier 3: Avoid'
    ]
    
    df['betting_tier'] = np.select(conditions, choices, default='Unclassified')
    
    return df


def generate_betting_watchlist(df, include_tier2=True):
    """Generate a betting watchlist from classified races.
    
    Args:
        df: DataFrame with betting_tier column
        include_tier2: Include Tier 2 Value races (default True)
    
    Returns:
        DataFrame of races worth monitoring for betting opportunities
    """
    # Filter to actionable tiers
    if include_tier2:
        watchlist = df[df['betting_tier'].isin(['Tier 1: Focus', 'Tier 2: Value'])].copy()
    else:
        watchlist = df[df['betting_tier'] == 'Tier 1: Focus'].copy()
    
    # Add watchlist priority (1 = highest)
    watchlist['priority'] = watchlist['betting_tier'].map({
        'Tier 1: Focus': 1,
        'Tier 2: Value': 2
    })
    
    # Sort by priority, then date
    if 'date' in watchlist.columns:
        watchlist = watchlist.sort_values(['priority', 'date'])
    elif 'Date' in watchlist.columns:
        watchlist = watchlist.sort_values(['priority', 'Date'])
    else:
        watchlist = watchlist.sort_values('priority')
    
    return watchlist


def main():
    """Apply betting strategy tiers to historical and upcoming races."""
    
    print("=" * 70)
    print("BETTING STRATEGY TIER CLASSIFICATION")
    print("=" * 70)
    
    # Load historical scored races
    historical_file = Path('data/processed/race_scores.parquet')
    if historical_file.exists():
        print(f"\n📊 Loading historical races from {historical_file}...")
        df_hist = pd.read_parquet(historical_file)
        print(f"   Loaded {len(df_hist):,} historical races")
        
        # Apply tier classification
        print("\n🔍 Applying betting strategy tiers to historical data...")
        df_hist_classified = classify_race_betting_tier(df_hist)
        
        # Summary statistics
        print("\n📈 Historical Race Classification:")
        tier_counts = df_hist_classified['betting_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = (count / len(df_hist_classified)) * 100
            print(f"   {tier:20s}: {count:6,} races ({pct:5.1f}%)")
        
        # Tier 1 breakdown
        tier1 = df_hist_classified[df_hist_classified['betting_tier'] == 'Tier 1: Focus']
        if len(tier1) > 0:
            print(f"\n🎯 Tier 1 Focus Races Breakdown:")
            print(f"   Total: {len(tier1):,} races")
            if 'course_clean' in tier1.columns:
                print(f"\n   Top 5 Courses:")
                top_courses = tier1['course_clean'].value_counts().head()
                for course, count in top_courses.items():
                    print(f"   - {course:20s}: {count:4,} races")
            
            if 'class_clean' in tier1.columns or 'class' in tier1.columns:
                class_col = 'class_clean' if 'class_clean' in tier1.columns else 'class'
                print(f"\n   Class Distribution:")
                class_dist = tier1[class_col].value_counts()
                for cls, count in class_dist.items():
                    print(f"   - {cls:10s}: {count:4,} races")
        
        # Save classified historical data
        output_file = Path('data/processed/race_scores_with_betting_tiers.parquet')
        df_hist_classified.to_parquet(output_file)
        print(f"\n💾 Saved classified historical data to {output_file}")
    
    # Load upcoming predicted fixtures
    fixtures_file = Path('data/processed/scored_fixtures_calendar.csv')
    if fixtures_file.exists():
        print(f"\n📅 Loading upcoming fixtures from {fixtures_file}...")
        df_fixtures = pd.read_csv(fixtures_file)
        print(f"   Loaded {len(df_fixtures):,} upcoming fixtures")
        
        # Apply tier classification
        print("\n🔍 Applying betting strategy tiers to upcoming fixtures...")
        df_fixtures_classified = classify_race_betting_tier(df_fixtures, strict_mode=False)
        
        # Summary statistics
        print("\n📈 Upcoming Fixtures Classification:")
        tier_counts = df_fixtures_classified['betting_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = (count / len(df_fixtures_classified)) * 100
            print(f"   {tier:20s}: {count:6,} races ({pct:5.1f}%)")
        
        # Generate watchlist
        watchlist = generate_betting_watchlist(df_fixtures_classified, include_tier2=True)
        print(f"\n📋 Betting Watchlist: {len(watchlist)} races")
        
        # Show top upcoming opportunities
        if len(watchlist) > 0:
            print(f"\n🎯 Top 10 Upcoming Betting Opportunities:")
            display_cols = ['date', 'course', 'class', 'betting_tier', 'prize', 'race_score']
            available_cols = [c for c in display_cols if c in watchlist.columns]
            
            top10 = watchlist.head(10)[available_cols]
            print(top10.to_string(index=False))
        
        # Save classified fixtures
        output_file = Path('data/processed/fixtures_with_betting_tiers.csv')
        df_fixtures_classified.to_csv(output_file, index=False)
        print(f"\n💾 Saved classified fixtures to {output_file}")
        
        # Save watchlist
        watchlist_file = Path('data/processed/betting_watchlist.csv')
        watchlist.to_csv(watchlist_file, index=False)
        print(f"💾 Saved betting watchlist to {watchlist_file}")
    
    print("\n✅ Betting strategy tier classification complete!")


if __name__ == "__main__":
    main()
