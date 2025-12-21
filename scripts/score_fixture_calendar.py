#!/usr/bin/env python3
"""Score fixture calendar using predicted race characteristics from historical data.

Predicts class, distance, prize, going, and field size for each fixture based on
historical patterns at each course, then applies Phase 2 race scoring.

Input: data/processed/bha_2026_all_courses_class1-4.csv
Output: data/processed/scored_fixtures_calendar.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_historical_data():
    """Load historical race data to learn course patterns"""
    print("Loading historical race data...")
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Filter to Class 1-4 only (matching our betting strategy)
    df = df[df['class'].isin(['Class 1', 'Class 2', 'Class 3', 'Class 4'])].copy()
    
    print(f"Loaded {len(df):,} historical races (Class 1-4)")
    return df


def build_course_profiles(df_hist):
    """Build statistical profiles for each course based on historical data"""
    print("\nBuilding course profiles from historical data...")
    
    profiles = {}
    
    # Group by course_clean (already exists in parquet)
    for course in df_hist['course_clean'].unique():
        course_data = df_hist[df_hist['course_clean'] == course].copy()
        
        if len(course_data) < 10:  # Skip courses with very little data
            continue
        
        # Get class distribution
        class_dist = course_data['class_clean'].value_counts(normalize=True).to_dict()
        
        # Calculate typical characteristics using cleaned columns
        # For better predictions, capture both typical AND premium races
        profile = {
            'course': course,
            'total_races': len(course_data),
            
            # Class statistics
            'typical_class': course_data['class_clean'].mode()[0] if len(course_data['class_clean'].mode()) > 0 else 'Class 3',
            'class_dist': class_dist,
            'best_class': course_data['class_clean'].mode()[0] if len(course_data['class_clean'].mode()) > 0 else 'Class 2',  # Best typical class
            
            # Distance statistics
            'typical_distance': course_data['dist_f_clean'].median(),
            'distance_range': (course_data['dist_f_clean'].min(), course_data['dist_f_clean'].max()),
            
            # Prize money - use 75th percentile for better races (Saturday fixtures etc)
            'typical_prize': course_data['prize_clean'].median(),
            'premium_prize': course_data['prize_clean'].quantile(0.75),  # Better races
            
            # Going (mode)
            'typical_going': course_data['going'].mode()[0] if len(course_data['going'].mode()) > 0 else 'Good',
            
            # Field size statistics
            'typical_field_size': course_data['ran'].median(),
            'large_field_size': course_data['ran'].quantile(0.75),  # Competitive races
            
            # Surface (mode) - use surface_clean
            'typical_surface': course_data['surface_clean'].mode()[0] if len(course_data['surface_clean'].mode()) > 0 else 'Turf',
            
            # Pattern race frequency (for quality indicator)
            'pattern_pct': (course_data['pattern'].notna() & (course_data['pattern'] != '')).mean() * 100
        }
        
        profiles[course] = profile
    
    print(f"Built profiles for {len(profiles)} courses")
    return profiles


def predict_race_characteristics(fixtures, course_profiles):
    """Predict race characteristics for fixtures based on course profiles
    
    Uses intelligent heuristics:
    - Weekend fixtures get better class and higher prizes
    - Premium courses (Ascot, Cheltenham, etc) get upgraded predictions
    - Seasonal adjustments for going conditions
    """
    print("\nPredicting race characteristics for fixtures...")
    
    # Map months to seasons for going adjustments
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    }
    
    # Going by season
    season_going = {
        'winter': 'Soft',
        'spring': 'Good',
        'summer': 'Good to Firm',
        'autumn': 'Good'
    }
    
    predictions = []
    
    for idx, fixture in fixtures.iterrows():
        course = fixture['Course']
        date = pd.to_datetime(fixture['Date'])
        is_weekend = date.dayofweek >= 5  # Saturday=5, Sunday=6
        
        # Get course profile (use defaults if course not in historical data)
        if course in course_profiles:
            profile = course_profiles[course]
        else:
            # Default profile for unknown courses
            profile = {
                'typical_class': 'Class 3',
                'best_class': 'Class 2',
                'typical_distance': 8.0,
                'typical_prize': 5000,
                'premium_prize': 8000,
                'typical_going': 'Good',
                'typical_field_size': 10,
                'large_field_size': 12,
                'typical_surface': fixture.get('Surface', 'Turf'),
                'pattern_pct': 0
            }
        
        # Predict class - upgrade for weekends
        if is_weekend:
            predicted_class = profile.get('best_class', 'Class 2')
        else:
            predicted_class = profile.get('typical_class', 'Class 3')
        
        # Predict prize - higher for weekends
        if is_weekend:
            predicted_prize = profile.get('premium_prize', profile.get('typical_prize', 5000))
        else:
            predicted_prize = profile.get('typical_prize', 5000)
        
        # Predict field size - larger for weekends
        if is_weekend:
            predicted_field_size = profile.get('large_field_size', profile.get('typical_field_size', 10))
        else:
            predicted_field_size = profile.get('typical_field_size', 10)
        
        # Seasonal going adjustments
        month = date.month
        season = season_map.get(month, 'spring')
        predicted_going = season_going.get(season, 'Good')
        
        # Override with course typical going for AW tracks
        if fixture.get('Surface', 'Turf') == 'AWT':
            predicted_going = 'Standard'  # AW tracks have consistent going
        
        # Create predicted race entry
        pred = {
            'date': fixture['Date'],
            'course': course,
            'class': predicted_class,
            'dist_f': profile.get('typical_distance', 8.0),
            'prize': predicted_prize,
            'going': predicted_going,
            'ran': int(predicted_field_size),
            'surface': fixture.get('Surface', profile.get('typical_surface', 'Turf')),
            'pattern_pct': profile.get('pattern_pct', 0),
            # Keep fixture metadata
            'weekday': fixture.get('Weekday', ''),
            'time': fixture.get('Time', ''),
            'course_group': fixture.get('CourseGroup', ''),
            'region': fixture.get('Region', ''),
            'code': fixture.get('Code', ''),
            'type': fixture.get('Type', ''),
            'is_weekend': is_weekend
        }
        
        predictions.append(pred)
    
    df_pred = pd.DataFrame(predictions)
    print(f"Generated predictions for {len(df_pred)} fixtures")
    print(f"Weekend fixtures: {df_pred['is_weekend'].sum()}")
    
    return df_pred


def score_predicted_races(df_pred):
    """Apply Phase 2 scoring logic to predicted races"""
    print("\nScoring predicted races...")
    
    # Load course tiers
    course_tiers = pd.read_csv('data/processed/lookups/course_tiers.csv')
    course_tier_map = dict(zip(course_tiers['course'], course_tiers['tier']))
    
    # Add course tier
    df_pred['course_tier'] = df_pred['course'].map(course_tier_map).fillna('Major')
    
    # Initialize score
    df_pred['race_score'] = 0.0
    
    # 1. Class quality (0-30 points)
    class_scores = {
        'Class 1': 30,
        'Class 2': 22,
        'Class 3': 15,
        'Class 4': 8
    }
    df_pred['class_score'] = df_pred['class'].map(class_scores).fillna(8)
    df_pred['race_score'] += df_pred['class_score']
    
    # 2. Prize money (0-25 points)
    df_pred['prize_score'] = np.clip(df_pred['prize'] / 500, 0, 25)
    df_pred['race_score'] += df_pred['prize_score']
    
    # 3. Course tier (0-20 points)
    tier_scores = {'Premium': 20, 'Major': 12, 'Minor': 5}
    df_pred['tier_score'] = df_pred['course_tier'].map(tier_scores).fillna(12)
    df_pred['race_score'] += df_pred['tier_score']
    
    # 4. Field size (0-15 points) - competitive fields
    df_pred['field_score'] = np.clip((df_pred['ran'] - 5) * 1.5, 0, 15)
    df_pred['race_score'] += df_pred['field_score']
    
    # 5. Pattern race bonus (0-10 points)
    df_pred['pattern_score'] = np.clip(df_pred['pattern_pct'] / 10, 0, 10)
    df_pred['race_score'] += df_pred['pattern_score']
    
    # Calculate percentile
    df_pred['score_percentile'] = df_pred['race_score'].rank(pct=True) * 100
    
    # Assign tiers
    df_pred['race_tier'] = pd.cut(
        df_pred['race_score'],
        bins=[0, 50, 70, 100],
        labels=['Tier 3: Avoid', 'Tier 2: Value', 'Tier 1: Focus'],
        include_lowest=True
    )
    
    print(f"\nScore distribution:")
    print(f"  Min: {df_pred['race_score'].min():.1f}")
    print(f"  Mean: {df_pred['race_score'].mean():.1f}")
    print(f"  Max: {df_pred['race_score'].max():.1f}")
    
    print(f"\nTier distribution:")
    print(df_pred['race_tier'].value_counts())
    
    return df_pred


def main():
    # Load fixtures
    fixtures_file = Path('data/processed/bha_2026_all_courses_class1-4.csv')
    print(f"Loading fixtures from {fixtures_file}...")
    fixtures = pd.read_csv(fixtures_file)
    print(f"Loaded {len(fixtures)} fixtures")
    
    # Load historical data
    df_hist = load_historical_data()
    
    # Build course profiles
    course_profiles = build_course_profiles(df_hist)
    
    # Predict race characteristics
    df_pred = predict_race_characteristics(fixtures, course_profiles)
    
    # Score races
    df_scored = score_predicted_races(df_pred)
    
    # Save scored fixtures
    output_file = Path('data/processed/scored_fixtures_calendar.csv')
    df_scored.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved scored fixtures to {output_file}")
    print(f"\nTop 10 highest-scoring upcoming races:")
    top10 = df_scored.nlargest(10, 'race_score')[['date', 'course', 'class', 'race_score', 'race_tier']]
    print(top10.to_string())


if __name__ == "__main__":
    main()
