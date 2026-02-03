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
import warnings
from datetime import datetime
from pathlib import Path
import pytz

import pandas as pd
import numpy as np

# Check for XGBoost availability
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import odds converter
sys.path.insert(0, str(project_root / "scripts"))
from odds_converter import probability_to_decimal_odds, probability_to_fractional_odds

# Paths
DATA_DIR = project_root / "data"
WIN_MODEL_FILE = project_root / "models" / ("horse_win_predictor.json" if HAS_XGBOOST else "horse_win_predictor.pkl")
CALIBRATED_MODEL_FILE = project_root / "models" / "horse_win_predictor_calibrated.pkl"
PLACE_MODEL_FILE = project_root / "models" / "horse_place_predictor.pkl"
SHOW_MODEL_FILE = project_root / "models" / "horse_show_predictor.pkl"
FEATURE_COLS_FILE = project_root / "models" / "feature_columns.txt"
DIAGNOSTICS_FILE = DATA_DIR / "processed" / "model_diagnostics.json"
CALIBRATION_METRICS_FILE = project_root / "models" / "calibration_metrics.json"

def get_historical_data_path():
    """Get the best available historical data file (same logic as training)"""
    data_dir = DATA_DIR / "processed"
    
    # Prefer latest version with all enhancements
    connections_v2_path = data_dir / 'race_scores_connections_v2.parquet'
    no_leak_path = data_dir / 'race_scores_with_all_features_no_leakage.parquet'
    legacy_path = data_dir / 'race_scores.parquet'
    
    if connections_v2_path.exists():
        print(f"✓ Using latest data: {connections_v2_path.name} (91 features)")
        return connections_v2_path
    elif no_leak_path.exists():
        print(f"✓ Using enhanced data: {no_leak_path.name} (77 features)")
        return no_leak_path
    elif legacy_path.exists():
        print(f"✓ Using legacy data: {legacy_path.name} (72 features)")
        return legacy_path
    else:
        raise FileNotFoundError("No historical race data found in data/processed/")

HISTORICAL_DATA = get_historical_data_path()


def load_models():
    """Load trained ML models (win, place, show) and feature columns"""
    print("\nLoading ML models...")
    
    # Try to load calibrated model first
    calibrated_model = None
    if CALIBRATED_MODEL_FILE.exists():
        try:
            import joblib
            calibrated_model = joblib.load(CALIBRATED_MODEL_FILE)
            print(f"[OK] Loaded calibrated model: {CALIBRATED_MODEL_FILE.name}")
        except Exception as e:
            print(f"[!] Could not load calibrated model: {e}")
    
    # Suppress XGBoost warnings about model version compatibility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if HAS_XGBOOST:
            from xgboost import XGBClassifier
            win_model = XGBClassifier()
            win_model.load_model(WIN_MODEL_FILE)
        else:
            with open(WIN_MODEL_FILE, 'rb') as f:
                win_model = pickle.load(f)
        
        # Use calibrated model if available
        if calibrated_model is not None:
            win_model = calibrated_model
            print(f"[OK] Using calibrated model for predictions")
        
        # Place and show models are always pickled (if they exist)
        try:
            with open(PLACE_MODEL_FILE, 'rb') as f:
                place_model = pickle.load(f)
        except FileNotFoundError:
            print(f"[!] Place model not found: {PLACE_MODEL_FILE}")
            place_model = None
        
        try:
            with open(SHOW_MODEL_FILE, 'rb') as f:
                show_model = pickle.load(f)
        except FileNotFoundError:
            print(f"[!] Show model not found: {SHOW_MODEL_FILE}")
            show_model = None
    
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


def is_lfs_pointer(filepath):
    """Check if file is an LFS pointer instead of actual data.
    
    LFS pointer files are small text files (<200 bytes) that look like:
    version https://git-lfs.github.com/spec/v1
    oid sha256:...
    size 12345
    """
    if not filepath.exists():
        return False
    
    try:
        # LFS pointers are always < 200 bytes, real Parquet files are MB-GB
        file_size = filepath.stat().st_size
        if file_size < 1000:
            # Double-check by reading first line
            with open(filepath, 'rb') as f:
                first_line = f.read(50)
                if b'version https://git-lfs.github.com' in first_line:
                    return True
            return True  # Suspiciously small file
        return False
    except Exception:
        return False


def load_historical_data():
    """Load historical race data for feature engineering"""
    print("\nLoading historical data...")
    
    # Check if file is an LFS pointer
    if not HISTORICAL_DATA.exists():
        print(f"[!] Historical data file not found: {HISTORICAL_DATA}")
        print("[!] This is expected on first GitHub Actions run with cache disabled.")
        print("[!] The cache will be built after data files are available.")
        sys.exit(0)  # Exit gracefully, not an error
    
    if is_lfs_pointer(HISTORICAL_DATA):
        file_size = HISTORICAL_DATA.stat().st_size
        print(f"[!] Historical data file is an LFS pointer ({file_size} bytes), not actual data")
        print("[!] This happens when:")
        print("    - LFS is disabled in GitHub Actions (lfs: false)")
        print("    - The cache is empty (first run)")
        print("[!] Solution: The historical Parquet files need to be available either:")
        print("    1. Via GitHub Actions cache (built on first successful run)")
        print("    2. Generated locally and committed")
        print("[!] Exiting gracefully - no predictions generated")
        sys.exit(0)  # Exit gracefully, not an error
    
    try:
        df = pd.read_parquet(HISTORICAL_DATA)
    except Exception as e:
        print(f"[!] Error reading Parquet file: {e}")
        print(f"[!] File size: {HISTORICAL_DATA.stat().st_size} bytes")
        print("[!] The file may be corrupted or is not a valid Parquet file")
        sys.exit(1)
    
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


def parse_weight_lbs(weight_str):
    """Parse weight string to pounds (e.g., '9-7' -> 133, '140' -> 140)"""
    if pd.isna(weight_str) or not weight_str or weight_str == '-':
        return 140  # Default weight
    
    weight_str = str(weight_str).strip()
    
    # Format: "9-7" (9 stone 7 lbs) or just "133" (lbs)
    if '-' in weight_str:
        parts = weight_str.split('-')
        try:
            stones = int(parts[0])
            lbs = int(parts[1]) if len(parts) > 1 else 0
            return stones * 14 + lbs
        except:
            return 140
    else:
        try:
            return float(weight_str)
        except:
            return 140


def classify_pace_style_from_history(horse_history, dist_f):
    """Classify horse's running style from historical patterns"""
    if len(horse_history) < 3:
        return 'UNKNOWN'
    
    # Sprint races (< 8f) with low draw and good finishes = likely front-runner
    sprint_races = horse_history[horse_history['dist_f'] < 8]
    if len(sprint_races) >= 3:
        # Check if consistently finishes well in sprints
        low_draw_wins = sprint_races[
            (sprint_races.get('draw', 999) <= 3) & 
            (sprint_races['pos'] <= 3)
        ]
        if len(low_draw_wins) / len(sprint_races) > 0.4:
            return 'LEADER'
    
    # Consistent top-3 finishes = presser
    top3_rate = (horse_history['pos'] <= 3).mean()
    consistency = horse_history['pos'].std()
    
    if top3_rate > 0.40 and consistency < 3.0:
        return 'PRESSER'
    
    # Large variance = closer
    if consistency > 5.0:
        return 'CLOSER'
    
    # Middle ground
    if top3_rate > 0.25:
        return 'MIDPACK'
    
    return 'UNKNOWN'


def calculate_pedigree_features(runner, historical_df, prediction_date):
    """Calculate pedigree features using expanding windows (NO LEAKAGE)"""
    features = {}
    
    sire_id = runner.get('sire_id')
    sire_name = runner.get('sire')
    
    if not sire_id and not sire_name:
        # No sire info - use defaults
        return {
            'sire_win_rate': 0.10,
            'sire_place_rate': 0.33,
            'sire_surface_match': 0.10,
            'sire_distance_match': 0.10,
            'sire_going_match': 0.10,
            'sire_class_match': 0.10
        }
    
    # Get sire's historical progeny results BEFORE prediction date
    sire_mask = (historical_df['sire_id'] == sire_id) if sire_id else (historical_df['sire'].str.lower() == str(sire_name).lower())
    sire_history = historical_df[
        sire_mask & 
        (historical_df['date_dt'] < pd.to_datetime(prediction_date))
    ]
    
    if len(sire_history) < 5:
        # Insufficient sire history
        return {
            'sire_win_rate': 0.10,
            'sire_place_rate': 0.33,
            'sire_surface_match': 0.10,
            'sire_distance_match': 0.10,
            'sire_going_match': 0.10,
            'sire_class_match': 0.10
        }
    
    # Overall sire stats
    features['sire_win_rate'] = sire_history['won'].mean()
    features['sire_place_rate'] = (sire_history['pos'] <= 3).mean()
    
    # Surface-specific (from race_info via runner context - needs to be passed)
    # We'll calculate this in the main function
    features['sire_surface_match'] = features['sire_win_rate']  # Placeholder
    features['sire_distance_match'] = features['sire_win_rate']  # Placeholder
    features['sire_going_match'] = features['sire_win_rate']  # Placeholder
    features['sire_class_match'] = features['sire_win_rate']  # Placeholder
    
    return features


def calculate_recent_form_features(jockey_name, trainer_name, course, historical_df, prediction_date):
    """Calculate 14d/30d recent form for jockey and trainer"""
    features = {}
    
    cutoff_14d = pd.to_datetime(prediction_date) - pd.Timedelta(days=14)
    cutoff_30d = pd.to_datetime(prediction_date) - pd.Timedelta(days=30)
    
    # Jockey form
    jockey_recent_14d = historical_df[
        (historical_df['jockey'].str.lower() == jockey_name.lower()) &
        (historical_df['date_dt'] >= cutoff_14d) &
        (historical_df['date_dt'] < pd.to_datetime(prediction_date))
    ]
    
    jockey_recent_30d = historical_df[
        (historical_df['jockey'].str.lower() == jockey_name.lower()) &
        (historical_df['date_dt'] >= cutoff_30d) &
        (historical_df['date_dt'] < pd.to_datetime(prediction_date))
    ]
    
    features['jockey_form_14d'] = jockey_recent_14d['won'].mean() if len(jockey_recent_14d) >= 3 else 0.10
    features['jockey_form_30d'] = jockey_recent_30d['won'].mean() if len(jockey_recent_30d) >= 5 else 0.10
    features['jockey_in_form'] = 1 if features['jockey_form_14d'] > 0.20 else 0
    
    # Jockey at this course
    jockey_course = jockey_recent_30d[
        jockey_recent_30d['course'].str.lower() == course.lower()
    ]
    features['jockey_course_form_30d'] = jockey_course['won'].mean() if len(jockey_course) >= 2 else features['jockey_form_30d']
    
    # Trainer form
    trainer_recent_14d = historical_df[
        (historical_df['trainer'].str.lower() == trainer_name.lower()) &
        (historical_df['date_dt'] >= cutoff_14d) &
        (historical_df['date_dt'] < pd.to_datetime(prediction_date))
    ]
    
    trainer_recent_30d = historical_df[
        (historical_df['trainer'].str.lower() == trainer_name.lower()) &
        (historical_df['date_dt'] >= cutoff_30d) &
        (historical_df['date_dt'] < pd.to_datetime(prediction_date))
    ]
    
    features['trainer_form_14d'] = trainer_recent_14d['won'].mean() if len(trainer_recent_14d) >= 3 else 0.10
    features['trainer_form_30d'] = trainer_recent_30d['won'].mean() if len(trainer_recent_30d) >= 5 else 0.10
    features['trainer_in_form'] = 1 if features['trainer_form_14d'] > 0.15 else 0
    
    # Trainer at this course
    trainer_course = trainer_recent_30d[
        trainer_recent_30d['course'].str.lower() == course.lower()
    ]
    features['trainer_course_form_30d'] = trainer_course['won'].mean() if len(trainer_course) >= 2 else features['trainer_form_30d']
    
    # Jockey-trainer combination
    jockey_trainer = historical_df[
        (historical_df['jockey'].str.lower() == jockey_name.lower()) &
        (historical_df['trainer'].str.lower() == trainer_name.lower()) &
        (historical_df['date_dt'] >= cutoff_30d) &
        (historical_df['date_dt'] < pd.to_datetime(prediction_date))
    ]
    features['jockey_trainer_form_30d'] = jockey_trainer['won'].mean() if len(jockey_trainer) >= 2 else features['jockey_form_30d']
    
    # Both in form?
    features['connections_in_form'] = 1 if (features['jockey_in_form'] == 1 and features['trainer_in_form'] == 1) else 0
    
    return features


def build_horse_features_from_racecard(runner, race_info, historical_df, prediction_date=None):
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
        # Draw features (may be missing in raw JSON)
        try:
            draw_val = int(runner.get('draw')) if runner.get('draw') not in (None, '-', '') else None
        except Exception:
            draw_val = None
        features['draw'] = draw_val if draw_val is not None else 0
        features['draw_pct'] = (features['draw'] / max(1, features['field_size'])) if features['field_size'] > 0 else 0
        features['draw_group_win_rate'] = 0.0
        
        # Weight features
        weight_str = runner.get('wgt') or runner.get('weight') or runner.get('lbs')
        features['weight_lbs'] = parse_weight_lbs(weight_str)
        features['weight_vs_avg'] = 0  # Can't calculate without other runners
        features['is_top_weight'] = 0  # Can't calculate without other runners
        features['weight_change'] = 0  # No history for new horse
        
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

    # Weight features
    weight_str = runner.get('wgt') or runner.get('weight') or runner.get('lbs')
    features['weight_lbs'] = parse_weight_lbs(weight_str)
    
    # Calculate weight vs race average (need all runners in race)
    if 'all_weights' in race_info and race_info['all_weights']:
        all_weights = [parse_weight_lbs(w) for w in race_info['all_weights']]
        if all_weights:
            avg_weight = sum(all_weights) / len(all_weights)
            features['weight_vs_avg'] = features['weight_lbs'] - avg_weight
            features['is_top_weight'] = 1 if features['weight_lbs'] == max(all_weights) else 0
        else:
            features['weight_vs_avg'] = 0
            features['is_top_weight'] = 0
    else:
        features['weight_vs_avg'] = 0
        features['is_top_weight'] = 0
    
    # Weight change from last race
    if len(horse_history) > 0 and 'wgt' in horse_history.columns:
        prev_weight = horse_history.head(1)['wgt'].apply(parse_weight_lbs).iloc[0]
        features['weight_change'] = features['weight_lbs'] - prev_weight
    else:
        features['weight_change'] = 0

    # Draw extraction from runner JSON
    try:
        draw_val = int(runner.get('draw')) if runner.get('draw') not in (None, '-', '') else None
    except Exception:
        draw_val = None
    features['draw'] = draw_val if draw_val is not None else 0
    features['draw_pct'] = features['draw'] / max(1, features['field_size'])

    # Attempt to compute draw_group_win_rate from historical data if available
    # Use same grouping as training: create cd_key and draw_group bins
    try:
        # Create cd_key for historical data if absent
        if 'cd_key' in historical_df.columns:
            cd_key = historical_df['cd_key']
        else:
            # attempt to construct
            historical_df = historical_df.copy()
            historical_df['distance_band'] = historical_df.get('distance_band', historical_df.get('distance_f', 0)).astype(str)
            historical_df['course_clean'] = historical_df.get('course', historical_df.get('course_clean', ''))
            historical_df['cd_key'] = historical_df['course_clean'].astype(str) + '_' + historical_df['distance_band'].astype(str)

        cd_key_val = race_info.get('course', '') + '_' + str(race_info.get('distance_f', ''))

        # Draw pct grouping
        dp = features['draw_pct'] if features['draw_pct'] is not None else 0
        if dp <= 0.333:
            dg = 'low'
        elif dp <= 0.666:
            dg = 'mid'
        else:
            dg = 'high'

        hist_mask = (historical_df.get('cd_key') == cd_key_val) & (historical_df.get('draw_group') == dg)
        hist_group = historical_df[hist_mask]
        if len(hist_group) > 0 and 'won' in hist_group.columns:
            # Use historical win rate for this draw-group
            features['draw_group_win_rate'] = hist_group['won'].sum() / len(hist_group)
        else:
            # Fallback to cd_win_rate if present
            if 'cd_win_rate' in historical_df.columns:
                cd_hist = historical_df[historical_df.get('cd_key') == cd_key_val]
                features['draw_group_win_rate'] = cd_hist['cd_win_rate'].mean() if len(cd_hist) > 0 else 0.0
            else:
                features['draw_group_win_rate'] = 0.0
    except Exception:
        features['draw_group_win_rate'] = 0.0
    
    # ===== NEW FEATURES (25 total) =====
    
    if prediction_date is None:
        prediction_date = race_info.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Pedigree features (6)
    pedigree_feats = calculate_pedigree_features(runner, historical_df, prediction_date)
    features.update(pedigree_feats)
    
    # Pace features (9)
    pace_style = classify_pace_style_from_history(horse_history, dist_f)
    features['pace_style_leader'] = 1 if pace_style == 'LEADER' else 0
    features['pace_style_presser'] = 1 if pace_style == 'PRESSER' else 0
    features['pace_style_closer'] = 1 if pace_style == 'CLOSER' else 0
    features['pace_style_midpack'] = 1 if pace_style == 'MIDPACK' else 0
    
    # Race-level pace features (calculated after all horses - set defaults for now)
    features['race_leader_count'] = 0  # Will be updated in predict_race
    features['race_closer_count'] = 0  # Will be updated in predict_race
    features['style_advantage'] = 0  # Will be updated in predict_race
    
    # Distance specialization
    sprint_races = horse_history[horse_history['dist_f'] < 7]
    staying_races = horse_history[horse_history['dist_f'] >= 12]
    features['sprint_specialist'] = 1 if (len(sprint_races) >= 3 and (sprint_races['pos'] <= 3).mean() > 0.35) else 0
    features['staying_specialist'] = 1 if (len(staying_races) >= 3 and (staying_races['pos'] <= 3).mean() > 0.35) else 0
    
    # Recent form features (10)
    form_feats = calculate_recent_form_features(
        jockey_name, trainer_name, course, historical_df, prediction_date
    )
    features.update(form_feats)
    
    # Add age-related features (should already be in model from training)
    # These need to be in the racecard or calculated
    horse_age = runner.get('age', 4)  # Default to 4
    features['age'] = int(horse_age) if horse_age else 4
    features['is_peak_age'] = 1 if 3 <= features['age'] <= 5 else 0
    features['is_3yo'] = 1 if features['age'] == 3 else 0
    features['is_veteran'] = 1 if features['age'] >= 7 else 0
    features['age_vs_avg'] = 0  # Will calculate from race average
    
    # Beaten lengths features
    if len(horse_history) > 0 and 'btn' in horse_history.columns:
        # Parse btn column
        def parse_btn(btn_str):
            if pd.isna(btn_str) or btn_str in ['', '-', 'W', 'won']:
                return 0.0
            try:
                btn_str = str(btn_str).strip().lower()
                if 'nk' in btn_str:
                    return 0.25
                if 'hd' in btn_str or 'head' in btn_str:
                    return 0.1
                if 'dist' in btn_str:
                    return 30.0
                return float(btn_str)
            except:
                return 0.0
        
        horse_history['btn_numeric'] = horse_history['btn'].apply(parse_btn)
        features['avg_btn_last_3'] = horse_history.head(3)['btn_numeric'].mean()
        features['unlucky_last'] = 1 if (len(horse_history) > 0 and horse_history.head(1)['btn_numeric'].iloc[0] <= 1.0 and horse_history.head(1)['btn_numeric'].iloc[0] > 0) else 0
    else:
        features['avg_btn_last_3'] = 0.0
        features['unlucky_last'] = 0
    
    # Gear/headgear features
    headgear = runner.get('headgear', '').lower()
    features['has_blinkers'] = 1 if 'b' in headgear else 0
    features['has_visor'] = 1 if 'v' in headgear else 0
    
    prev_headgear = horse_history.head(1).get('headgear', '').iloc[0] if len(horse_history) > 0 and 'headgear' in horse_history.columns else ''
    features['first_time_blinkers'] = 1 if (features['has_blinkers'] == 1 and 'b' not in str(prev_headgear).lower()) else 0
    features['gear_changed'] = 1 if (headgear != str(prev_headgear).lower()) else 0
    
    # Race condition features
    race_type = race_info.get('type', '')
    features['is_handicap'] = 1 if 'handicap' in race_type.lower() or 'hcap' in race_type.lower() else 0
    features['is_maiden'] = 1 if 'maiden' in race_type.lower() else 0
    features['is_pattern'] = 1 if race_info.get('pattern') else 0
    
    # Prize money (log scale)
    try:
        prize_str = str(race_info.get('prize', '0')).replace('£', '').replace(',', '')
        prize_val = float(prize_str) if prize_str else 0
        features['prize_log'] = np.log1p(prize_val)
    except:
        features['prize_log'] = 0.0
    
    # Distance categories
    features['is_sprint'] = 1 if dist_f < 7 else 0
    features['is_mile'] = 1 if 7 <= dist_f < 9 else 0
    features['is_middle'] = 1 if 9 <= dist_f < 12 else 0
    features['is_staying'] = 1 if dist_f >= 12 else 0
    
    return features


def predict_race(racecard, historical_df, win_model, place_model, show_model, feature_cols, prediction_date=None):
    """Generate predictions for all horses in a race"""
    
    if prediction_date is None:
        prediction_date = racecard.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Collect all weights for weight_vs_avg calculation
    all_weights = []
    for runner in racecard.get('runners', []):
        weight_str = runner.get('wgt') or runner.get('weight') or runner.get('lbs')
        if weight_str:
            all_weights.append(weight_str)
    
    # Add to racecard for feature building
    racecard = racecard.copy()
    racecard['all_weights'] = all_weights
    
    # First pass: build features for all horses
    all_features = []
    for runner in racecard.get('runners', []):
        features = build_horse_features_from_racecard(runner, racecard, historical_df, prediction_date)
        all_features.append(features)
    
    # Second pass: calculate race-level pace features
    leader_count = sum(1 for f in all_features if f.get('pace_style_leader', 0) == 1)
    closer_count = sum(1 for f in all_features if f.get('pace_style_closer', 0) == 1)
    field_size = len(all_features)
    
    # Update pace features for all horses
    for features in all_features:
        features['race_leader_count'] = leader_count
        features['race_closer_count'] = closer_count
        
        # Style advantage: closer benefits from fast pace (many leaders)
        # Leader benefits from slow pace (few leaders)
        if features.get('pace_style_closer', 0) == 1 and leader_count >= 3:
            features['style_advantage'] = 1
        elif features.get('pace_style_leader', 0) == 1 and leader_count <= 1:
            features['style_advantage'] = 1
        else:
            features['style_advantage'] = 0
    
    # Calculate age average for age_vs_avg
    all_ages = [f.get('age', 4) for f in all_features]
    avg_age = sum(all_ages) / len(all_ages) if all_ages else 4
    for features in all_features:
        features['age_vs_avg'] = features.get('age', 4) - avg_age
    
    # Third pass: make predictions
    race_predictions = []
    
    for idx, (runner, features) in enumerate(zip(racecard.get('runners', []), all_features)):
        # Features already built above in first pass with all updates from second pass
        
        # Create feature vector in correct order
        feature_vector = [features.get(col, 0) for col in feature_cols]
        
        # Predict all three probabilities
        X = np.array(feature_vector).reshape(1, -1)
        win_prob = win_model.predict_proba(X)[0][1]
        # Place/show models may be absent or have different feature shapes
        if place_model is not None:
            try:
                place_prob = place_model.predict_proba(X)[0][1]
            except Exception:
                place_prob = min(1.0, win_prob * 0.6)
        else:
            place_prob = min(1.0, win_prob * 0.6)
        
        if show_model is not None:
            try:
                show_prob = show_model.predict_proba(X)[0][1]
            except Exception:
                show_prob = min(1.0, win_prob * 0.4)
        else:
            show_prob = min(1.0, win_prob * 0.4)
        
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
    
    # Generate and save diagnostics
    print("\n" + "="*60)
    print("GENERATING DIAGNOSTICS")
    print("="*60)
    
    diagnostics = generate_diagnostics(predictions_df, target_date)
    save_diagnostics(diagnostics, target_date)
    
    print("\n" + "="*60)
    print("PREDICTIONS COMPLETE")
    print("="*60)


def generate_diagnostics(predictions_df, date_str):
    """Generate diagnostic metrics for predictions."""
    diagnostics = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'total_races': predictions_df['course'].nunique(),
        'total_horses': len(predictions_df),
        'avg_field_size': predictions_df.groupby(['course', 'race_time']).size().mean(),
        'probability_distribution': {
            'min': float(predictions_df['win_probability'].min()),
            'max': float(predictions_df['win_probability'].max()),
            'mean': float(predictions_df['win_probability'].mean()),
            'median': float(predictions_df['win_probability'].median()),
            'std': float(predictions_df['win_probability'].std()),
        },
        'top_pick_probabilities': {
            'min': float(predictions_df.groupby(['course', 'race_time'])['win_probability'].max().min()),
            'max': float(predictions_df.groupby(['course', 'race_time'])['win_probability'].max().max()),
            'mean': float(predictions_df.groupby(['course', 'race_time'])['win_probability'].max().mean()),
        },
        'feature_coverage': {},
        'model_info': {
            'using_calibrated': CALIBRATED_MODEL_FILE.exists(),
            'has_calibration_metrics': CALIBRATION_METRICS_FILE.exists(),
        }
    }
    
    # Feature coverage analysis
    feature_cols = ['career_runs', 'career_win_rate', 'cd_win_rate', 'avg_last_3_pos', 'or_numeric']
    for col in feature_cols:
        if col in predictions_df.columns:
            diagnostics['feature_coverage'][col] = {
                'null_pct': float(predictions_df[col].isna().mean() * 100),
                'zero_pct': float((predictions_df[col] == 0).mean() * 100),
            }
    
    # Cold start analysis
    if 'career_runs' in predictions_df.columns:
        cold_start_count = (predictions_df['career_runs'] == 0).sum()
        diagnostics['cold_start_horses'] = {
            'count': int(cold_start_count),
            'percentage': float(cold_start_count / len(predictions_df) * 100),
        }
    
    # Class distribution
    if 'race_class' in predictions_df.columns:
        class_dist = predictions_df['race_class'].value_counts().to_dict()
        diagnostics['class_distribution'] = {str(k): int(v) for k, v in class_dist.items()}
    
    print(f"  [OK] Generated diagnostics for {diagnostics['total_races']} races, {diagnostics['total_horses']} horses")
    print(f"  [OK] Top pick probability range: {diagnostics['top_pick_probabilities']['min']:.1%} - {diagnostics['top_pick_probabilities']['max']:.1%}")
    if 'cold_start_horses' in diagnostics:
        print(f"  [OK] Cold start horses: {diagnostics['cold_start_horses']['percentage']:.1f}%")
    
    return diagnostics


def save_diagnostics(diagnostics, date_str):
    """Save diagnostics to JSON file."""
    # Save current diagnostics
    current_file = DATA_DIR / "processed" / f"diagnostics_{date_str}.json"
    with open(current_file, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  [OK] Saved diagnostics: {current_file}")
    
    # Also update latest diagnostics
    latest_file = DATA_DIR / "processed" / "model_diagnostics.json"
    with open(latest_file, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  [OK] Updated latest: {latest_file}")


if __name__ == '__main__':
    main()
