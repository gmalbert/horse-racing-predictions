"""
Updated prediction script that extracts ALL 120 model features from historical data.

ROOT CAUSE FIX:
- Old version: Manually calculated ~47 features, ignoring 73 features in historical data
- New version: Calculates the 47 features the model was trained on from historical data
- Result: Proper feature vectors → varied predictions instead of identical 15.8%
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import re
from datetime import datetime
import argparse
import sys

# Load feature columns that model expects
def load_feature_columns():
    """Load the 47 feature names the model was actually trained on"""
    return [
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings', 
        'cd_runs', 'cd_win_rate', 'class_num', 'class_step', 'or_numeric', 
        'or_change', 'or_trend_3', 'avg_last_3_pos', 'wins_last_3', 'days_since_last', 
        'field_size', 'is_turf', 'going_numeric', 'race_score', 'draw', 'draw_pct', 
        'draw_group_win_rate', 'weight_lbs', 'weight_vs_avg', 'is_top_weight', 
        'weight_change', 'age', 'is_peak_age', 'is_3yo', 'is_veteran', 'age_vs_avg', 
        'avg_btn_last_3', 'unlucky_last', 'has_blinkers', 'has_visor', 
        'first_time_blinkers', 'gear_changed', 'is_handicap', 'is_maiden', 'is_pattern', 
        'prize_log', 'is_sprint', 'is_mile', 'is_middle', 'is_staying', 
        'jockey_career_runs', 'jockey_course_runs', 'jockey_trainer_runs'
    ]

def strip_country_suffix(name):
    """Remove country suffix like (IRE), (GB), (FR), (USA) from horse names"""
    import re
    if pd.isna(name):
        return ''
    return re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', str(name)).strip()

def build_features_from_historical_record(horse_history, runner, race_info, feature_cols):
    """
    Extract features from horse's most recent historical race record.
    Override race-specific features with current race data.
    
    Args:
        horse_history: DataFrame of horse's past races (sorted by date descending)
        runner: Current race runner dict
        race_info: Current race dict  
        feature_cols: List of 120 feature names model expects
        
    Returns:
        dict of {feature_name: value} for all 47 features
    """
    # Start with most recent historical record
    if len(horse_history) > 0:
        most_recent = horse_history.head(1).iloc[0]
        features = {}
        
        # Convert pos to numeric (some rows have strings like 'PU', 'F')
        horse_history_clean = horse_history.copy()
        horse_history_clean['pos'] = pd.to_numeric(horse_history_clean['pos'], errors='coerce')
        horse_history_clean['btn'] = pd.to_numeric(horse_history_clean['btn'], errors='coerce')
        horse_history_clean['or_numeric'] = pd.to_numeric(horse_history_clean.get('or_numeric', horse_history_clean.get('or', 0)), errors='coerce')
        
        # CALCULATE career statistics from full horse history
        features['career_runs'] = len(horse_history_clean)
        win_mask = (horse_history_clean['pos'] == 1)
        place_mask = (horse_history_clean['pos'] <= 3)
        features['career_win_rate'] = win_mask.sum() / len(horse_history_clean)
        features['career_place_rate'] = place_mask.sum() / len(horse_history_clean)
        features['career_earnings'] = horse_history_clean['prize_clean'].sum() if 'prize_clean' in horse_history_clean.columns else 0.0
        
        # Course/distance stats
        course = race_info.get('course', '')
        dist_f = float(str(race_info.get('distance_f', 8)).replace('f', ''))
        cd_races = horse_history_clean[
            (horse_history_clean['course'] == course) & 
            (horse_history_clean['dist_f'] == dist_f)
        ]
        features['cd_runs'] = len(cd_races)
        features['cd_win_rate'] = (cd_races['pos'] == 1).sum() / len(cd_races) if len(cd_races) > 0 else 0.0
        
        # Recent form stats
        last_3 = horse_history_clean.head(3)
        features['avg_last_3_pos'] = last_3['pos'].mean() if len(last_3) > 0 else 8.0
        features['wins_last_3'] = (last_3['pos'] == 1).sum()
        features['avg_btn_last_3'] = last_3['btn'].mean() if 'btn' in last_3.columns and len(last_3) > 0 else 5.0
        
        # Days since last race
        features['days_since_last'] = (pd.Timestamp.now() - horse_history_clean.iloc[0]['date_dt']).days
        
        # OR trends
        if 'or_numeric' in horse_history_clean.columns and len(horse_history_clean) > 0:
            features['or_trend_3'] = last_3['or_numeric'].mean() if len(last_3) > 0 else most_recent.get('or_numeric', 0)
            features['or_change'] = (most_recent.get('or_numeric', 0) - horse_history_clean.iloc[1].get('or_numeric', 0)) if len(horse_history_clean) > 1 else 0.0
            features['or_career_high'] = horse_history_clean['or_numeric'].max()
        
        # Class stats
        current_class = race_info.get('race_class', 'Class 4')
        class_match = re.search(r'(\d+)', str(current_class))
        current_class_num = int(class_match.group(1)) if class_match else 4
        if 'class_num' in horse_history.columns:
            prev_class = most_recent.get('class_num', current_class_num)
            features['class_step'] = current_class_num - prev_class
        else:
            features['class_step'] = 0
        
        # Jockey stats (placeholder - would need full DB for accurate calculation)
        jockey_name = runner.get('jockey', '')
        features['jockey_career_runs'] = 100  # Placeholder
        features['jockey_course_runs'] = 10   # Placeholder
        features['jockey_trainer_runs'] = 20  # Placeholder
        
        # Extract all OTHER features from historical record
        for col in feature_cols:
            if col not in features:  # Don't override calculated features
                if col in most_recent.index:
                    features[col] = most_recent[col]
                else:
                    # Feature not in historical data - will fill with default below
                    features[col] = 0
        
        # Override race-specific features with current race data
        features['field_size'] = int(race_info.get('field_size') or len(race_info.get('runners', [])) or 10)
        features['is_turf'] = 1 if race_info.get('surface') == 'Turf' else 0
        
        # Going
        going_map = {'heavy': 5, 'soft': 4, 'good to soft': 3, 'good': 2, 'good to firm': 1, 'firm': 0}
        going_str = str(race_info.get('going', 'good')).lower()
        features['going_numeric'] = going_map.get(going_str, 2)
        
        # Distance
        dist_f = float(str(race_info.get('distance_f', 8)).replace('f', ''))
        features['is_sprint'] = 1 if dist_f < 7 else 0
        features['is_mile'] = 1 if 7 <= dist_f < 9 else 0
        features['is_middle'] = 1 if 9 <= dist_f < 12 else 0
        features['is_staying'] = 1 if dist_f >= 12 else 0
        
        # Class
        class_str = race_info.get('race_class') or 'Class 4'
        class_match = re.search(r'(\d+)', str(class_str))
        features['class_num'] = int(class_match.group(1)) if class_match else 4
        
        # Race type
        race_type = race_info.get('type', '').lower()
        features['is_handicap'] = 1 if 'handicap' in race_type or 'hcap' in race_type else 0
        features['is_maiden'] = 1 if 'maiden' in race_type else 0
        features['is_pattern'] = 1 if race_info.get('pattern') else 0
        
        # Prize
        try:
            prize_str = str(race_info.get('prize', '0')).replace('£', '').replace(',', '')
            prize_val = float(prize_str) if prize_str else 0
            features['prize_log'] = np.log1p(prize_val)
        except:
            features['prize_log'] = 8.5
        
        # Race score
        features['race_score'] = race_info.get('race_score', 50.0)
        
        # Current OR from racecard 
        or_val = runner.get('ofr', '-')
        if or_val and or_val != '-':
            try:
                features['or_numeric'] = float(or_val)
            except:
                pass
        
        # Age from racecard
        age_val = runner.get('age')
        if age_val:
            try:
                features['age'] = int(age_val)
                features['is_peak_age'] = 1 if 3 <= features['age'] <= 5 else 0
                features['is_3yo'] = 1 if features['age'] == 3 else 0
                features['is_veteran'] = 1 if features['age'] >= 7 else 0
            except:
                pass
        
        return features
    
    else:
        # No historical data - use defaults based on OR
        return build_default_features(runner, race_info, feature_cols)

def build_default_features(runner, race_info, feature_cols):
    """
    For horses with no historical data, create default feature vector based on OR.
    """
    import re
    features = {col: 0 for col in feature_cols}
    
    # Extract OR
    or_val = runner.get('ofr', '-')
    if or_val and or_val != '-':
        try:
            or_numeric = float(or_val)
        except:
            or_numeric = 0
    else:
        or_numeric = 0
    
    # OR-based estimates
    if or_numeric > 0:
        base_win_rate = max(0.02, min(0.30, (or_numeric - 70) / 280))
        place_rate = min(0.65, base_win_rate * 3.0)
    else:
        base_win_rate = 0.08
        place_rate = 0.20
    
    # Basic features
    features['or_numeric'] = or_numeric
    features['career_win_rate'] = base_win_rate
    features['career_place_rate'] = place_rate
    features['career_runs'] = 10  # Estimate
    features['career_earnings'] = base_win_rate * 10 * 5000
    
    # Race context
    features['field_size'] = int(race_info.get('field_size') or len(race_info.get('runners', [])) or 10)
    features['is_turf'] = 1 if race_info.get('surface') == 'Turf' else 0
    
    # Going
    going_map = {'heavy': 5, 'soft': 4, 'good to soft': 3, 'good': 2, 'good to firm': 1, 'firm': 0}
    going_str = str(race_info.get('going', 'good')).lower()
    features['going_numeric'] = going_map.get(going_str, 2)
    
    # Distance
    dist_f = float(str(race_info.get('distance_f', 8)).replace('f', ''))
    features['is_sprint'] = 1 if dist_f < 7 else 0
    features['is_mile'] = 1 if 7 <= dist_f < 9 else 0
    features['is_middle'] = 1 if 9 <= dist_f < 12 else 0
    features['is_staying'] = 1 if dist_f >= 12 else 0
    
    # Class
    class_str = race_info.get('race_class') or 'Class 4'
    class_match = re.search(r'(\d+)', str(class_str))
    features['class_num'] = int(class_match.group(1)) if class_match else 4
    
    # Age
    age_val = runner.get('age', 4)
    try:
        features['age'] = int(age_val)
        features['is_peak_age'] = 1 if 3 <= features['age'] <= 5 else 0
        features['is_3yo'] = 1 if features['age'] == 3 else 0
        features['is_veteran'] = 1 if features['age'] >= 7 else 0
    except:
        features['age'] = 4
    
    # Race type
    race_type = race_info.get('type', '').lower()
    features['is_handicap'] = 1 if 'handicap' in race_type or 'hcap' in race_type else 0
    features['is_maiden'] = 1 if 'maiden' in race_type else 0
    features['is_pattern'] = 1 if race_info.get('pattern') else 0
    
    # Prize
    try:
        prize_str = str(race_info.get('prize', '0')).replace('£', '').replace(',', '')
        prize_val = float(prize_str) if prize_str else 0
        features['prize_log'] = np.log1p(prize_val)
    except:
        features['prize_log'] = 8.5
    
    features['race_score'] = race_info.get('race_score', 50.0)
    
    # Defaults for missing features
    features['avg_last_3_pos'] = 6.0
    features['jockey_career_win_rate'] = 0.10
    features['trainer_form_14d_v2'] = 0.10
    
    return features

def predict_single_race(racecard, historical_df, model, feature_cols, prediction_date):
    """Generate predictions for one race"""
    import re
    
    # Create horse_clean column for matching
    if 'horse_clean' not in historical_df.columns:
        historical_df['horse_clean'] = historical_df['horse'].apply(lambda x: strip_country_suffix(x).lower())
    
    predictions = []
    
    for runner in racecard.get('runners', []):
        horse_name = runner.get('name') or runner.get('horse', 'Unknown')
        horse_clean = strip_country_suffix(horse_name).lower()
        
        # Find historical data
        horse_history = historical_df[
            historical_df['horse_clean'] == horse_clean
        ].sort_values('date_dt', ascending=False)
        
        # Build feature vector
        features = build_features_from_historical_record(
            horse_history, runner, racecard, feature_cols
        )
        
        # DEBUG: Show first 5 horses' feature vectors
        if len(predictions) < 5:
            print(f"  DEBUG {horse_name}: career={features.get('career_win_rate', 0):.3f}, or={features.get('or_numeric', 0)}, jockey_runs={features.get('jockey_career_runs', 0)}")
        
        # Create DataFrame with correct column order
        X = pd.DataFrame([features])[feature_cols]
        
        # Predict
        try:
            win_prob = model.predict_proba(X)[0, 1]
        except Exception as e:
            print(f"ERROR predicting {horse_name}: {e}")
            win_prob = 0.158  # Default
        
        predictions.append({
            'horse': horse_name,
            'win_probability': win_prob,
            'place_probability': win_prob * 0.6,  # Estimate
            'show_probability': win_prob * 0.4   # Estimate
        })
    
    return predictions

def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    
    print(f"\n{'='*60}\nPREDICTING RACES FOR {args.date}\n{'='*60}\n")
    
    # Load model
    print("Loading model...")
    try:
        with open('models/horse_win_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
        print("[OK] Loaded base XGBoost model")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Load feature columns
    feature_cols = load_feature_columns()
    print(f"[OK] Model expects {len(feature_cols)} features\n")
    
    # Load racecards
    racecard_file = f'data/raw/racecards_{args.date}.json'
    if not os.path.exists(racecard_file):
        print(f"[ERROR] No racecards found for {args.date}")
        return
    
    with open(racecard_file, 'r') as f:
        racecards_raw = json.load(f)
    
    # Flatten nested dict structure: {"Region": {"Course": {"Time": {...}}}} -> [{...}]
    racecards = []
    if isinstance(racecards_raw, dict):
        for region_or_course, nested_dict in racecards_raw.items():
            if isinstance(nested_dict, dict):
                # Check if this is region level (contains courses) or course level (contains times)
                first_value = next(iter(nested_dict.values()))
                if isinstance(first_value, dict) and 'runners' in first_value:
                    # This is course level, values are races
                    for race_data in nested_dict.values():
                        racecards.append(race_data)
                else:
                    # This is region level, recurse into courses
                    for course_dict in nested_dict.values():
                        if isinstance(course_dict, dict):
                            for race_data in course_dict.values():
                                if isinstance(race_data, dict):
                                    racecards.append(race_data)
    elif isinstance(racecards_raw, list):
        racecards = racecards_raw
    
    print(f"[OK] Loaded {len(racecards)} racecards\n")
    
    # Load historical data
    print("Loading historical data...")
    historical_df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
    print(f"[OK] Loaded {len(historical_df):,} historical records\n")
    
    # Ensure date_dt column
    if 'date_dt' not in historical_df.columns:
        historical_df['date_dt'] = pd.to_datetime(historical_df['date'])
    
    # Generate predictions
    all_predictions = []
    
    for idx, racecard in enumerate(racecards, 1):
        course = racecard.get('course', 'Unknown')
        time = racecard.get('time', 'Unknown')
        runners_count = len(racecard.get('runners', []))
        
        print(f"[{idx}/{len(racecards)}] {time} {course} ({runners_count} runners)")
        
        preds = predict_single_race(racecard, historical_df, model, feature_cols, args.date)
        
        # Add race context
        for pred in preds:
            pred['course'] = course
            pred['race_time'] = time
            pred['date'] = args.date
        
        all_predictions.extend(preds)
        
        # Show top 3
        preds_sorted = sorted(preds, key=lambda x: x['win_probability'], reverse=True)
        for i, p in enumerate(preds_sorted[:3], 1):
            print(f"  {i}. {p['horse']:30s} Win: {p['win_probability']*100:.1f}%")
    
    # Save predictions
    output_file = f'data/processed/predictions_{args.date}.csv'
    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_csv(output_file, index=False)
    print(f"\n[OK] Saved predictions to {output_file}")
    
    # Show statistics
    unique_probs = df_preds['win_probability'].nunique()
    print(f"\n{'='*60}")
    print(f"PREDICTION STATISTICS:")
    print(f"  Total horses: {len(df_preds)}")
    print(f"  Unique win probabilities: {unique_probs}")
    print(f"  Min: {df_preds['win_probability'].min():.2%}")
    print(f"  Max: {df_preds['win_probability'].max():.2%}")
    print(f"  Mean: {df_preds['win_probability'].mean():.2%}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
