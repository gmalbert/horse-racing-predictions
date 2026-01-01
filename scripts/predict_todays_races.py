#!/usr/bin/env python3
"""Predict win probabilities for today's races using the trained ML model.

Reads racecards from data/raw/racecards_YYYY-MM-DD.json
Generates horse features from historical data
Runs ML model predictions
Outputs predictions to data/processed/predictions_YYYY-MM-DD.csv

Usage:
  python scripts/predict_todays_races.py              # Use today's date
  python scripts/predict_todays_races.py --date 2025-12-31  # Specific date
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
import pytz

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import odds converter
sys.path.insert(0, str(project_root / "scripts"))
from odds_converter import probability_to_decimal_odds, probability_to_fractional_odds

# Paths
DATA_DIR = project_root / "data"
WIN_MODEL_FILE = project_root / "models" / "horse_win_predictor.pkl"
PLACE_MODEL_FILE = project_root / "models" / "horse_place_predictor.pkl"
SHOW_MODEL_FILE = project_root / "models" / "horse_show_predictor.pkl"
FEATURE_COLS_FILE = project_root / "models" / "feature_columns.txt"
HISTORICAL_DATA = DATA_DIR / "processed" / "race_scores_with_betting_tiers.parquet"


def load_models():
    """Load trained ML models (win, place, show) and feature columns"""
    print("\nLoading ML models...")
    
    with open(WIN_MODEL_FILE, 'rb') as f:
        win_model = pickle.load(f)
    
    with open(PLACE_MODEL_FILE, 'rb') as f:
        place_model = pickle.load(f)
    
    with open(SHOW_MODEL_FILE, 'rb') as f:
        show_model = pickle.load(f)
    
    with open(FEATURE_COLS_FILE, 'r') as f:
        feature_cols = [line.strip() for line in f]
    
    print(f"[OK] Models loaded (win, place, show) with {len(feature_cols)} features")
    return win_model, place_model, show_model, feature_cols


def load_racecards(date_str):
    """Load racecards from JSON file"""
    racecards_file = DATA_DIR / "raw" / f"racecards_{date_str}.json"
    
    if not racecards_file.exists():
        print(f"[ERROR] Racecards file not found: {racecards_file}")
        print(f"Run: python scripts/fetch_racecards.py --date {date_str}")
        return None
    
    print(f"\nLoading racecards from {racecards_file}...")
    with open(racecards_file, 'r') as f:
        data = json.load(f)
    
    # Handle nested structure: {region: {course: {time: race}}}
    racecards = []
    if isinstance(data, dict) and 'racecards' in data:
        # Old format: {'racecards': [...]}
        racecards = data.get('racecards', [])
    elif isinstance(data, dict):
        # New nested format: {region: {course: {time: race}}}
        for region in data.values():
            for course in region.values():
                for race in course.values():
                    racecards.append(race)
    
    print(f"[OK] Loaded {len(racecards)} racecards")
    
    return racecards


def load_historical_data():
    """Load historical race data for feature engineering"""
    print("\nLoading historical data...")
    df = pd.read_parquet(HISTORICAL_DATA)
    
    # Convert date to datetime
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    
    print(f"[OK] Loaded {len(df):,} historical records")
    print(f"Date range: {df['date_dt'].min()} to {df['date_dt'].max()}")
    
    return df


def extract_distance_furlongs(dist_str):
    """Convert distance string to furlongs (e.g., '16.0' -> 16.0)"""
    try:
        return float(dist_str)
    except:
        return 8.0  # Default


def extract_class_num(class_str):
    """Extract class number from string (e.g., 'Class 4' -> 4)"""
    try:
        return int(class_str.replace('Class ', ''))
    except:
        return 4  # Default


def extract_days_since_last_run(last_run_str):
    """Extract days since last run from string (e.g., '25', '(240P)' -> 240)"""
    if not last_run_str or last_run_str == '-':
        return 999
    
    # Remove parentheses and letters
    clean_str = ''.join(c for c in str(last_run_str) if c.isdigit())
    
    try:
        return int(clean_str) if clean_str else 999
    except:
        return 999


def encode_going(going_str):
    """Encode going condition to numeric (1=Firm to 7=Heavy)"""
    going_map = {
        'Hard': 0.5, 'Firm': 1, 'Good To Firm': 2, 'Good': 3,
        'Good To Soft': 4, 'Soft': 5, 'Heavy': 6, 'Standard': 3
    }
    return going_map.get(going_str, 3)


def build_horse_features_from_racecard(runner, race_info, historical_df):
    """Build features for a specific horse from racecard data
    
    Matches the 24 features used in the trained model (18 original + 6 jockey):
    - career_runs, career_win_rate, career_place_rate, career_earnings
    - cd_runs, cd_win_rate
    - class_num, class_step
    - or_numeric, or_change, or_trend_3
    - avg_last_3_pos, wins_last_3
    - days_since_last
    - field_size, is_turf, going_numeric
    - race_score
    - jockey_career_runs, jockey_career_win_rate
    - jockey_course_runs, jockey_course_win_rate
    - jockey_trainer_runs, jockey_trainer_win_rate
    """
    horse_name = runner.get('horse') or runner.get('name', 'Unknown')
    jockey_name = runner.get('jockey', 'Unknown')
    trainer_name = runner.get('trainer', 'Unknown')
    
    # Filter to this horse's history
    horse_history = historical_df[
        historical_df['horse'].str.lower() == horse_name.lower()
    ].copy()
    
    # Calculate jockey stats from ALL historical data (not just this horse)
    jockey_history = historical_df[
        historical_df['jockey'].str.lower() == jockey_name.lower()
    ] if jockey_name != 'Unknown' else pd.DataFrame()
    
    # Jockey-course stats
    course = race_info['course']
    jockey_course_history = jockey_history[
        jockey_history['course'].str.lower() == course.lower()
    ] if len(jockey_history) > 0 else pd.DataFrame()
    
    # Jockey-trainer stats
    jockey_trainer_history = jockey_history[
        jockey_history['trainer'].str.lower() == trainer_name.lower()
    ] if len(jockey_history) > 0 and trainer_name != 'Unknown' else pd.DataFrame()
    
    if horse_history.empty:
        # New horse or no data - use defaults
        features = {
            'career_runs': 0,
            'career_win_rate': 0.0,
            'career_place_rate': 0.0,
            'career_earnings': 0.0,
            'cd_runs': 0,
            'cd_win_rate': 0.0,
            'class_num': extract_class_num(race_info.get('race_class', 'Class 4')),
            'class_step': 0,
            'or_numeric': float(runner.get('ofr') or 0) if runner.get('ofr') and runner.get('ofr') != '-' else 0,
            'or_change': 0,
            'or_trend_3': 0,
            'avg_last_3_pos': 10.0,
            'wins_last_3': 0,
            'days_since_last': extract_days_since_last_run(runner.get('last_run')),
            'field_size': int(race_info.get('field_size', 10)),
            'is_turf': 1 if race_info.get('surface') == 'Turf' else 0,
            'going_numeric': encode_going(race_info.get('going', 'Good')),
            'race_score': 50.0,  # Default
            # Jockey features
            'jockey_career_runs': len(jockey_history),
            'jockey_career_win_rate': (jockey_history['pos'] == 1).sum() / len(jockey_history) if len(jockey_history) > 0 else 0.0,
            'jockey_course_runs': len(jockey_course_history),
            'jockey_course_win_rate': (jockey_course_history['pos'] == 1).sum() / len(jockey_course_history) if len(jockey_course_history) > 0 else 0.0,
            'jockey_trainer_runs': len(jockey_trainer_history),
            'jockey_trainer_win_rate': (jockey_trainer_history['pos'] == 1).sum() / len(jockey_trainer_history) if len(jockey_trainer_history) > 0 else 0.0
        }
        return features
    
    # Sort by date
    horse_history = horse_history.sort_values('date_dt', ascending=False)
    
    # Career stats
    career_runs = len(horse_history)
    # pos is already numeric in this dataset
    career_wins = (horse_history['pos'] == 1).sum()
    career_places = (horse_history['pos'] <= 3).sum()
    # prize is string with currency symbol, need to clean
    career_earnings = horse_history['prize'].replace('[\£,]', '', regex=True).astype(float).sum() if 'prize' in horse_history.columns else 0.0
    
    # Course/distance specific
    course = race_info['course']
    dist_f = extract_distance_furlongs(race_info['distance_f'])
    
    cd_history = horse_history[
        (horse_history['course'].str.lower() == course.lower()) &
        (horse_history['dist_f'].between(dist_f - 1, dist_f + 1))
    ]
    cd_runs = len(cd_history)
    cd_wins = (cd_history['pos'] == 1).sum() if cd_runs > 0 else 0
    
    # Class
    class_num = extract_class_num(race_info['race_class'])
    recent_class = horse_history.head(3)['class'].str.extract(r'(\d+)').astype(float).mean()[0] if len(horse_history) >= 3 else class_num
    class_step = recent_class - class_num  # Positive = stepping up
    
    # Official Rating
    or_numeric = float(runner.get('ofr', 0)) if runner.get('ofr', '-') != '-' else 0
    recent_or = horse_history.head(3)['or'].replace('-', np.nan).astype(float).dropna() if 'or' in horse_history.columns else pd.Series([or_numeric])
    or_change = or_numeric - recent_or.iloc[0] if len(recent_or) > 0 and or_numeric > 0 else 0
    or_trend_3 = recent_or.mean() if len(recent_or) > 0 else or_numeric
    
    # Recent form (last 3 races)
    recent_3 = horse_history.head(3)
    avg_last_3_pos = recent_3['pos'].mean() if len(recent_3) > 0 else 10.0
    wins_last_3 = (recent_3['pos'] == 1).sum()
    
    # Days since last race
    days_since_last = extract_days_since_last_run(runner.get('last_run'))
    
    # Race context
    field_size = int(race_info.get('field_size', 10))
    is_turf = 1 if race_info.get('surface') == 'Turf' else 0
    going_numeric = encode_going(race_info.get('going', 'Good'))
    
    # Race score (if available in race_info, else default)
    race_score = race_info.get('race_score', 50.0)
    
    # Jockey features
    jockey_career_runs = len(jockey_history)
    jockey_career_wins = (jockey_history['pos'] == 1).sum() if len(jockey_history) > 0 else 0
    jockey_course_runs = len(jockey_course_history)
    jockey_course_wins = (jockey_course_history['pos'] == 1).sum() if len(jockey_course_history) > 0 else 0
    jockey_trainer_runs = len(jockey_trainer_history)
    jockey_trainer_wins = (jockey_trainer_history['pos'] == 1).sum() if len(jockey_trainer_history) > 0 else 0
    
    features = {
        'career_runs': career_runs,
        'career_win_rate': career_wins / career_runs if career_runs > 0 else 0.0,
        'career_place_rate': career_places / career_runs if career_runs > 0 else 0.0,
        'career_earnings': career_earnings,
        'cd_runs': cd_runs,
        'cd_win_rate': cd_wins / cd_runs if cd_runs > 0 else 0.0,
        'class_num': class_num,
        'class_step': class_step,
        'or_numeric': or_numeric,
        'or_change': or_change,
        'or_trend_3': or_trend_3,
        'avg_last_3_pos': avg_last_3_pos,
        'wins_last_3': wins_last_3,
        'days_since_last': days_since_last,
        'field_size': field_size,
        'is_turf': is_turf,
        'going_numeric': going_numeric,
        'race_score': race_score,
        'jockey_career_runs': jockey_career_runs,
        'jockey_career_win_rate': jockey_career_wins / jockey_career_runs if jockey_career_runs > 0 else 0.0,
        'jockey_course_runs': jockey_course_runs,
        'jockey_course_win_rate': jockey_course_wins / jockey_course_runs if jockey_course_runs > 0 else 0.0,
        'jockey_trainer_runs': jockey_trainer_runs,
        'jockey_trainer_win_rate': jockey_trainer_wins / jockey_trainer_runs if jockey_trainer_runs > 0 else 0.0
    }
    
    return features


def predict_race(racecard, historical_df, win_model, place_model, show_model, feature_cols):
    """Generate predictions for all horses in a race"""
    
    race_predictions = []
    
    for runner in racecard.get('runners', []):
        # Build features
        features = build_horse_features_from_racecard(runner, racecard, historical_df)
        
        # Create feature vector in correct order
        feature_vector = [features.get(col, 0) for col in feature_cols]
        
        # Predict all three probabilities
        X = np.array(feature_vector).reshape(1, -1)
        win_prob = win_model.predict_proba(X)[0][1]
        place_prob = place_model.predict_proba(X)[0][1]
        show_prob = show_model.predict_proba(X)[0][1]
        
        # Store prediction
        # Convert race time from GMT to US Eastern Time
        gmt = pytz.timezone('GMT')
        eastern = pytz.timezone('US/Eastern')
        
        # Handle both off_dt (old format) and off_time (new format)
        if 'off_dt' in racecard:
            race_dt_gmt = datetime.fromisoformat(racecard['off_dt'].replace('+00:00', '')).replace(tzinfo=gmt)
            race_time_gmt = racecard.get('off_time', race_dt_gmt.strftime('%H:%M'))
        else:
            # New format just has off_time string
            race_time_gmt = racecard.get('off_time', '00:00')
            # Create a datetime for timezone conversion
            date_str = racecard.get('date', datetime.now().strftime('%Y-%m-%d'))
            race_dt_gmt = datetime.strptime(f"{date_str} {race_time_gmt}", '%Y-%m-%d %H:%M').replace(tzinfo=gmt)
        
        race_dt_eastern = race_dt_gmt.astimezone(eastern)
        race_time_eastern = race_dt_eastern.strftime('%I:%M %p ET').lstrip('0')  # Format: 7:20 AM ET
        
        race_predictions.append({
            'course': racecard.get('course', 'Unknown'),
            'race_time': race_time_eastern,
            'race_time_gmt': race_time_gmt,  # Keep original GMT time for reference
            'race_name': racecard.get('race_name', ''),
            'race_class': racecard.get('race_class', ''),
            'distance_f': racecard.get('distance_f', ''),
            'surface': racecard.get('surface', 'Turf'),
            'going': racecard.get('going', ''),
            'field_size': racecard.get('field_size', len(racecard.get('runners', []))),
            'horse': runner.get('horse') or runner.get('name', 'Unknown'),
            'jockey': runner.get('jockey', ''),
            'trainer': runner.get('trainer', ''),
            'age': runner.get('age', ''),
            'weight_lbs': runner.get('lbs', ''),
            'ofr': runner.get('ofr', '-'),
            'last_run': runner.get('last_run', '-'),
            'form': runner.get('form', ''),
            'win_probability': win_prob,
            'place_probability': place_prob,
            'show_probability': show_prob,
            'win_odds_decimal': probability_to_decimal_odds(win_prob),
            'win_odds_fractional': probability_to_fractional_odds(win_prob),
            'place_odds_decimal': probability_to_decimal_odds(place_prob),
            'place_odds_fractional': probability_to_fractional_odds(place_prob),
            'show_odds_decimal': probability_to_decimal_odds(show_prob),
            'show_odds_fractional': probability_to_fractional_odds(show_prob),
            **features  # Include all features for analysis
        })
    
    return race_predictions


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict win probabilities for races using trained ML model'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date to predict (YYYY-MM-DD). Defaults to today.'
    )
    args = parser.parse_args()
    
    # Get target date
    if args.date:
        target_date = args.date
        # Validate date format
        try:
            datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            print(f"[ERROR] Invalid date format: {target_date}. Use YYYY-MM-DD")
            return
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print("="*60)
    print(f"PREDICTING RACES FOR {target_date}")
    print("="*60)
    
    # Load components
    win_model, place_model, show_model, feature_cols = load_models()
    racecards = load_racecards(target_date)
    historical_df = load_historical_data()
    
    if not racecards:
        print("\n[ERROR] No racecards found. Exiting.")
        return
    
    # Generate predictions for all races
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    all_predictions = []
    
    for i, racecard in enumerate(racecards, 1):
        course = racecard['course']
        time = racecard['off_time']
        runners_count = len(racecard.get('runners', []))
        
        print(f"\n[{i}/{len(racecards)}] {time} {course} ({runners_count} runners)")
        
        race_preds = predict_race(racecard, historical_df, win_model, place_model, show_model, feature_cols)
        all_predictions.extend(race_preds)
        
        # Show top 3 predicted horses with all probabilities
        race_df = pd.DataFrame(race_preds).sort_values('win_probability', ascending=False)
        for j, row in enumerate(race_df.head(3).itertuples(), 1):
            print(f"  {j}. {row.horse:25s} Win: {row.win_probability:.1%} | Place: {row.place_probability:.1%} | Show: {row.show_probability:.1%}")
    
    # Save all predictions
    predictions_df = pd.DataFrame(all_predictions)
    output_file = DATA_DIR / "processed" / f"predictions_{target_date}.csv"
    predictions_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal races analyzed: {len(racecards)}")
    print(f"Total horses predicted: {len(predictions_df)}")
    print(f"\n[SAVED] Predictions: {output_file}")
    
    # Show top 10 highest probability horses across all races
    print("\n" + "="*60)
    print("TOP 10 PREDICTIONS (All Races)")
    print("="*60)
    
    top_10 = predictions_df.nlargest(10, 'win_probability')[
        ['race_time', 'course', 'horse', 'jockey', 'win_probability', 'race_class', 'distance_f']
    ]
    
    print("\n" + top_10.to_string(index=False))
    
    # Show races with highest average probabilities (most predictable)
    print("\n" + "="*60)
    print("MOST PREDICTABLE RACES (Highest Top Horse Probability)")
    print("="*60)
    
    race_max_probs = predictions_df.groupby(['race_time', 'course', 'race_name'])['win_probability'].max().reset_index()
    race_max_probs = race_max_probs.sort_values('win_probability', ascending=False).head(10)
    
    print("\n" + race_max_probs.to_string(index=False))
    
    print("\n" + "="*60)
    print("PREDICTIONS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
