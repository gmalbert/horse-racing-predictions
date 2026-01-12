"""
Phase 3: Horse-Level Win Prediction Model

Builds a machine learning model to predict horse win probability using:
- Career statistics (win rate, place rate, earnings)
- Course/distance form (CD performance)
- Class step (moving up/down)
- Official Rating trend
- Recent form (last 3 runs)
- Days since last race
- Going suitability
- Jockey/trainer stats

Model: XGBoost classifier (gradient boosting)
Target: Binary win/loss classification
Output: Win probability for each horse + feature importance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle

# ML libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False
    print("[!] XGBoost not installed, using Random Forest instead")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def load_data():
    """Load scored race data"""
    data_dir = Path('data/processed')
    df = pd.read_parquet(data_dir / 'race_scores.parquet')
    print(f"Loaded {len(df):,} horse-race records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df

def engineer_career_features(df):
    """
    Calculate career statistics for each horse up to each race date.
    Uses expanding window to avoid lookahead bias.
    """
    print("\n" + "="*60)
    print("ENGINEERING CAREER FEATURES")
    print("="*60)
    
    # Sort by horse and date
    df = df.sort_values(['horse', 'date']).copy()
    
    # Convert position to win/place flags
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    
    # Career stats (cumulative, excluding current race)
    df['career_runs'] = df.groupby('horse').cumcount()
    df['career_wins'] = df.groupby('horse')['won'].cumsum().shift(1).fillna(0)
    df['career_places'] = df.groupby('horse')['placed'].cumsum().shift(1).fillna(0)
    
    # Win rate (avoid division by zero)
    df['career_win_rate'] = np.where(
        df['career_runs'] > 0,
        df['career_wins'] / df['career_runs'],
        0
    )
    
    df['career_place_rate'] = np.where(
        df['career_runs'] > 0,
        df['career_places'] / df['career_runs'],
        0
    )
    
    # Total prize money won (cumulative)
    df['prize_numeric'] = pd.to_numeric(df['prize_clean'], errors='coerce').fillna(0)
    df['career_earnings'] = df.groupby('horse')['prize_numeric'].cumsum().shift(1).fillna(0)
    
    print(f"  Career features: career_runs, career_win_rate, career_place_rate, career_earnings")
    
    return df

def engineer_course_distance_form(df):
    """
    Calculate course/distance (CD) specific form.
    CD form is critical in UK racing.
    """
    print("\nEngineering course/distance form...")
    
    # Create CD key
    df['cd_key'] = df['course_clean'] + '_' + df['distance_band']
    
    # CD stats (cumulative per horse, excluding current race)
    df['cd_runs'] = df.groupby(['horse', 'cd_key']).cumcount()
    df['cd_wins'] = df.groupby(['horse', 'cd_key'])['won'].cumsum().shift(1).fillna(0)
    
    df['cd_win_rate'] = np.where(
        df['cd_runs'] > 0,
        df['cd_wins'] / df['cd_runs'],
        df['career_win_rate']  # Default to career rate if no CD history
    )
    
    print(f"  CD features: cd_runs, cd_win_rate")
    
    return df


def engineer_draw_features(df):
    """
    Engineer draw-related features where available.
    - draw (numeric)
    - draw_pct (draw / field_size)
    - draw_group_win_rate: historical win rate for horses in the same draw-group
    Uses expanding-window grouping to avoid lookahead bias.
    """
    print("\nEngineering draw features...")

    # Ensure numeric draw and percent
    df['draw'] = pd.to_numeric(df.get('draw'), errors='coerce')
    # Field size is 'ran' in this dataset
    df['draw_pct'] = df['draw'] / df['ran'].replace(0, 1)

    # Bin draw_pct into 3 groups (low/mid/high) for grouping
    df['draw_group'] = pd.cut(df['draw_pct'].fillna(0), bins=[-0.1, 0.333, 0.666, 1.0], labels=['low', 'mid', 'high'])

    # Create cd_key if not present
    if 'cd_key' not in df.columns:
        df['cd_key'] = df['course_clean'] + '_' + df['distance_band']

    # Build a combined key to compute group-level running stats without leakage
    df['cd_draw_key'] = df['cd_key'].astype(str) + '_' + df['draw_group'].astype(str)

    # Compute runs and cumulative wins for each cd_draw_key (shifted to exclude current row)
    df['dg_runs'] = df.groupby('cd_draw_key').cumcount()
    df['dg_wins'] = df.groupby('cd_draw_key')['won'].cumsum().shift(1).fillna(0)

    # Draw-group win rate (fallback to cd_win_rate if insufficient history)
    df['draw_group_win_rate'] = np.where(
        df['dg_runs'] > 0,
        df['dg_wins'] / df['dg_runs'],
        df.get('cd_win_rate', 0)
    )

    # Fill NAs
    df['draw_group_win_rate'] = df['draw_group_win_rate'].fillna(0)

    # Clean up helper columns
    df = df.drop(columns=['cd_draw_key', 'dg_runs', 'dg_wins'])

    print("  Draw features: draw, draw_pct, draw_group_win_rate")
    return df

def engineer_weight_features(df):
    """
    Engineer weight-related features for handicaps.
    Parse weight strings (stone-lb or lbs) and compute race-relative features.
    """
    print("\nEngineering weight features...")
    
    df = df.copy()
    
    # Parse weight (lbs or st-lb format)
    def parse_weight_lbs(weight_str):
        if pd.isna(weight_str):
            return np.nan
        weight_str = str(weight_str).strip()
        
        # Format: "9-7" (9 stone 7 lbs) or just "133" (lbs)
        if '-' in weight_str:
            parts = weight_str.split('-')
            try:
                stones = int(parts[0])
                lbs = int(parts[1]) if len(parts) > 1 else 0
                return stones * 14 + lbs
            except:
                return np.nan
        else:
            try:
                return float(weight_str)
            except:
                return np.nan
    
    # Parse wgt column if present
    if 'wgt' in df.columns:
        df['weight_lbs'] = df['wgt'].apply(parse_weight_lbs)
    else:
        df['weight_lbs'] = 140  # Default weight
    
    # Weight relative to race (top weight = high, bottom = low)
    df['weight_rank'] = df.groupby(['date', 'course_clean', 'off'])['weight_lbs'].rank(ascending=False, method='min')
    df['weight_vs_avg'] = df.groupby(['date', 'course_clean', 'off'])['weight_lbs'].transform(
        lambda x: x - x.mean()
    )
    
    # Is this horse carrying top weight?
    df['is_top_weight'] = (df['weight_rank'] == 1).astype(int)
    
    # Weight change from last race
    df['prev_weight'] = df.groupby('horse')['weight_lbs'].shift(1)
    df['weight_change'] = df['weight_lbs'] - df['prev_weight']
    df['weight_change'] = df['weight_change'].fillna(0)
    
    # Fill NAs
    df['weight_lbs'] = df['weight_lbs'].fillna(140)
    df['weight_vs_avg'] = df['weight_vs_avg'].fillna(0)
    
    print("  Weight features: weight_lbs, weight_vs_avg, is_top_weight, weight_change")
    
    return df

def engineer_age_features(df):
    """
    Engineer age-related features.
    Horses peak at different ages depending on code.
    """
    print("\nEngineering age features...")
    
    df = df.copy()
    
    # Parse age if present
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(4)
    else:
        df['age'] = 4  # Default age
    
    # Age at peak performance (flat horses peak around 4-5)
    df['is_peak_age'] = df['age'].between(4, 5).astype(int)
    
    # Is improving 3yo?
    df['is_3yo'] = (df['age'] == 3).astype(int)
    
    # Is veteran (older, potentially declining)?
    df['is_veteran'] = (df['age'] >= 8).astype(int)
    
    # Age relative to race average
    df['age_vs_avg'] = df.groupby(['date', 'course_clean', 'off'])['age'].transform(
        lambda x: x - x.mean()
    )
    df['age_vs_avg'] = df['age_vs_avg'].fillna(0)
    
    print("  Age features: age, is_peak_age, is_3yo, is_veteran, age_vs_avg")
    
    return df

def engineer_trainer_form(df):
    """
    Calculate trainer recent form (hot/cold streak).
    Uses rolling window to avoid lookahead.
    """
    print("\nEngineering trainer form...")
    
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort chronologically
    df = df.sort_values('date_dt').reset_index(drop=True)
    
    # Track trainer stats over rolling 14-day and 30-day windows
    trainer_stats = {}
    
    features_14d = []
    features_30d = []
    
    for idx, row in df.iterrows():
        trainer = row.get('trainer', 'Unknown')
        race_date = row['date_dt']
        won = 1 if row.get('won', 0) == 1 else 0
        
        # Get stats from last 14/30 days
        if trainer in trainer_stats:
            history = trainer_stats[trainer]
            
            # Filter to last 14 days
            recent_14d = [h for h in history if (race_date - h['date']).days <= 14]
            if recent_14d:
                runs_14d = len(recent_14d)
                wins_14d = sum(h['won'] for h in recent_14d)
                features_14d.append(wins_14d / runs_14d if runs_14d > 0 else 0.0)
            else:
                features_14d.append(0.0)
            
            # Filter to last 30 days
            recent_30d = [h for h in history if (race_date - h['date']).days <= 30]
            if recent_30d:
                runs_30d = len(recent_30d)
                wins_30d = sum(h['won'] for h in recent_30d)
                features_30d.append(wins_30d / runs_30d if runs_30d > 0 else 0.0)
            else:
                features_30d.append(0.0)
        else:
            features_14d.append(0.0)
            features_30d.append(0.0)
        
        # Update trainer history AFTER recording
        if trainer not in trainer_stats:
            trainer_stats[trainer] = []
        trainer_stats[trainer].append({'date': race_date, 'won': won})
        
        # Keep only last 60 days of history (memory optimization)
        trainer_stats[trainer] = [
            h for h in trainer_stats[trainer] 
            if (race_date - h['date']).days <= 60
        ]
    
    df['trainer_win_rate_14d'] = features_14d
    df['trainer_win_rate_30d'] = features_30d
    
    print("  Trainer form: trainer_win_rate_14d, trainer_win_rate_30d")
    
    return df

def engineer_beaten_lengths_features(df):
    """
    Calculate beaten lengths features for form analysis.
    Close finishes indicate better form than distant losses.
    """
    print("\nEngineering beaten lengths features...")
    
    df = df.copy()
    
    # Parse beaten lengths (can be decimal, fraction, or text)
    def parse_btn(btn_str):
        if pd.isna(btn_str) or btn_str in ['', '-', 'W', 'won']:
            return 0.0  # Winner
        try:
            btn_str = str(btn_str).strip().lower()
            if 'nk' in btn_str:
                return 0.25
            if 'hd' in btn_str or 'head' in btn_str:
                return 0.1
            if 'shd' in btn_str or 'short head' in btn_str or 'sh' in btn_str:
                return 0.05
            if 'nse' in btn_str or 'nose' in btn_str:
                return 0.01
            if 'dist' in btn_str:
                return 30.0
            return float(btn_str)
        except:
            return np.nan
    
    # Parse btn column if present
    if 'btn' in df.columns:
        df['btn_lengths'] = df['btn'].apply(parse_btn)
    else:
        df['btn_lengths'] = 0
    
    # Calculate features from recent races
    df = df.sort_values(['horse', 'date']).reset_index(drop=True)
    
    # Average beaten lengths last 3 races (lower = closer finishes)
    df['avg_btn_last_3'] = df.groupby('horse')['btn_lengths'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Was beaten narrowly last race? (unlucky loser)
    df['prev_btn'] = df.groupby('horse')['btn_lengths'].shift(1)
    df['unlucky_last'] = (
        (df['prev_btn'] <= 1.0) & (df['prev_btn'] > 0)
    ).astype(int)
    
    # Fill NAs
    df['btn_lengths'] = df['btn_lengths'].fillna(0)
    df['avg_btn_last_3'] = df['avg_btn_last_3'].fillna(0)
    
    print("  BTN features: btn_lengths, avg_btn_last_3, unlucky_last")
    
    return df

def engineer_gear_features(df):
    """
    Engineer equipment/headgear features.
    First-time blinkers is historically a good angle.
    """
    print("\nEngineering gear features...")
    
    df = df.copy()
    
    # Common headgear codes
    # b = blinkers, v = visor, h = hood, t = tongue strap
    # p = cheekpieces, e = eye shield
    
    if 'headgear' in df.columns:
        df['headgear'] = df['headgear'].fillna('').str.lower()
    else:
        df['headgear'] = ''
    
    # Binary flags for common gear
    df['has_blinkers'] = df['headgear'].str.contains('b').astype(int)
    df['has_visor'] = df['headgear'].str.contains('v').astype(int)
    df['has_cheekpieces'] = df['headgear'].str.contains('p').astype(int)
    df['has_tongue_tie'] = df['headgear'].str.contains('t').astype(int)
    
    # First time with this gear (important!)
    df = df.sort_values(['horse', 'date']).reset_index(drop=True)
    
    df['prev_headgear'] = df.groupby('horse')['headgear'].shift(1).fillna('')
    
    # First-time blinkers (historically good angle)
    df['first_time_blinkers'] = (
        (df['has_blinkers'] == 1) & 
        (~df['prev_headgear'].str.contains('b', na=False))
    ).astype(int)
    
    # First-time visor
    df['first_time_visor'] = (
        (df['has_visor'] == 1) & 
        (~df['prev_headgear'].str.contains('v', na=False))
    ).astype(int)
    
    # Any gear change
    df['gear_changed'] = (df['headgear'] != df['prev_headgear']).astype(int)
    
    print("  Gear features: has_blinkers, has_visor, first_time_blinkers, gear_changed")
    
    return df

def engineer_race_condition_features(df):
    """
    Enhanced race condition features.
    FIX: Use TOTAL race prize pool (sum of all winnings), not individual winnings.
    """
    print("\nEngineering enhanced race condition features...")
    
    df = df.copy()
    
    # Is this a handicap race?
    if 'type' in df.columns:
        df['is_handicap'] = df['type'].str.contains('Handicap|Hcap', case=False, na=False).astype(int)
    else:
        df['is_handicap'] = 0
    
    # Is this a maiden race?
    if 'type' in df.columns:
        df['is_maiden'] = df['type'].str.contains('Maiden', case=False, na=False).astype(int)
    else:
        df['is_maiden'] = 0
    
    # Is this a stakes/pattern race?
    if 'pattern' in df.columns:
        df['is_pattern'] = df['pattern'].notna().astype(int)
    else:
        df['is_pattern'] = 0
    
    # Prize money tier - FIX DATA LEAKAGE!
    # prize_clean contains INDIVIDUAL winnings (winner gets more than losers)
    # We need TOTAL race prize pool (same for all horses in race)
    if 'prize_clean' in df.columns:
        df['individual_prize'] = pd.to_numeric(
            df['prize_clean'], 
            errors='coerce'
        ).fillna(0)
        
        # Calculate total prize pool per race (sum all winnings in the race)
        df['race_prize_total'] = df.groupby(['date', 'course_clean', 'off'])['individual_prize'].transform('sum')
        
        # Use total prize pool (same for all horses) to prevent leakage
        df['prize_log'] = np.log1p(df['race_prize_total'])
        
        print(f"  [FIXED] Using race prize pool instead of individual winnings")
    elif 'prize_numeric' in df.columns:
        df['prize_log'] = np.log1p(df['prize_numeric'])
    else:
        df['prize_log'] = 0
    
    # Distance bands (more granular)
    if 'dist_f' in df.columns:
        df['dist_f_num'] = pd.to_numeric(df['dist_f'], errors='coerce').fillna(8)
    else:
        df['dist_f_num'] = 8
    
    df['is_sprint'] = (df['dist_f_num'] <= 7).astype(int)
    df['is_mile'] = df['dist_f_num'].between(7.5, 9).astype(int)
    df['is_middle'] = df['dist_f_num'].between(9, 12).astype(int)
    df['is_staying'] = (df['dist_f_num'] > 12).astype(int)
    
    print("  Race conditions: is_handicap, is_maiden, is_pattern, prize_log, is_sprint")
    
    return df

def engineer_class_step(df):
    """
    Calculate class movement (stepping up/down in class).
    Class steps are important for predicting performance.
    """
    print("\nEngineering class step...")
    
    # Extract numeric class (Class 1 -> 1, Class 2 -> 2, etc.)
    df['class_num'] = df['class_clean'].str.extract(r'(\d+)').astype(float)
    
    # Previous class (for this horse)
    df['prev_class'] = df.groupby('horse')['class_num'].shift(1)
    
    # Class step (negative = stepping up to better class, positive = stepping down)
    df['class_step'] = df['class_num'] - df['prev_class']
    df['class_step'] = df['class_step'].fillna(0)  # First run = no step
    
    print(f"  Class features: class_num, class_step")
    
    return df

def engineer_rating_trend(df):
    """
    Calculate Official Rating (OR) trend.
    Rising OR = improving form.
    """
    print("\nEngineering rating trend...")
    
    # Convert OR to numeric
    df['or_numeric'] = pd.to_numeric(df['or'], errors='coerce')
    
    # Previous OR
    df['prev_or'] = df.groupby('horse')['or_numeric'].shift(1)
    
    # OR change (positive = improving)
    df['or_change'] = df['or_numeric'] - df['prev_or']
    df['or_change'] = df['or_change'].fillna(0)
    
    # 3-run average OR trend
    df['or_trend_3'] = df.groupby('horse')['or_numeric'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df['or_trend_3'] = df['or_trend_3'].shift(1)  # Exclude current race
    
    print(f"  Rating features: or_numeric, or_change, or_trend_3")
    
    return df

def engineer_recent_form(df):
    """
    Calculate recent form (last 3 runs).
    Recent form often trumps career stats.
    """
    print("\nEngineering recent form...")
    
    # Last 3 positions (excluding current)
    df['last_pos_1'] = df.groupby('horse')['pos_clean'].shift(1)
    df['last_pos_2'] = df.groupby('horse')['pos_clean'].shift(2)
    df['last_pos_3'] = df.groupby('horse')['pos_clean'].shift(3)
    
    # Average last 3 positions (lower = better)
    df['avg_last_3_pos'] = df[['last_pos_1', 'last_pos_2', 'last_pos_3']].mean(axis=1)
    
    # Wins in last 3 runs
    df['wins_last_3'] = (
        (df['last_pos_1'] == 1).astype(int) +
        (df['last_pos_2'] == 1).astype(int) +
        (df['last_pos_3'] == 1).astype(int)
    )
    
    print(f"  Recent form features: avg_last_3_pos, wins_last_3")
    
    return df

def engineer_recency(df):
    """
    Calculate days since last race.
    Horses can be too fresh or too rusty.
    """
    print("\nEngineering recency...")
    
    # Convert date to datetime
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # Previous race date
    df['prev_race_date'] = df.groupby('horse')['date_dt'].shift(1)
    
    # Days since last race
    df['days_since_last'] = (df['date_dt'] - df['prev_race_date']).dt.days
    df['days_since_last'] = df['days_since_last'].fillna(60)  # Default for first run
    
    # Bin into categories
    df['recency_category'] = pd.cut(
        df['days_since_last'],
        bins=[0, 7, 14, 28, 60, 365],
        labels=['<7d', '7-14d', '14-28d', '28-60d', '>60d']
    )
    
    print(f"  Recency features: days_since_last, recency_category")
    
    return df

def engineer_race_context(df):
    """
    Add race context features (field size, surface, going).
    """
    print("\nEngineering race context...")
    
    # Field size
    df['field_size'] = df['ran']
    
    # Surface (turf=1, aw=0)
    df['is_turf'] = (df['surface'] == 'Turf').astype(int)
    
    # Going code (encode as numeric - firm=1 to heavy=9)
    going_map = {
        'Firm': 1, 'Good to Firm': 2, 'Good': 3, 'Good to Soft': 4,
        'Soft': 5, 'Soft to Heavy': 6, 'Heavy': 7,
        'Standard': 3, 'Standard to Slow': 5, 'Slow': 7
    }
    df['going_numeric'] = df['going'].map(going_map).fillna(3)  # Default to 'Good'
    
    print(f"  Context features: field_size, is_turf, going_numeric")
    
    return df

def engineer_jockey_features(df):
    """
    Engineer jockey-specific features from historical data.
    Uses temporal ordering (date + time) to prevent data leakage.
    """
    print("\nEngineering jockey features...")
    
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Create full timestamp (date + race time) to prevent same-day leakage
    # This ensures races are processed in true chronological order
    df['datetime_full'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['off'].astype(str), 
        errors='coerce'
    )
    
    # Sort by full timestamp
    df = df.sort_values('datetime_full').reset_index(drop=True)
    
    # Initialize jockey feature tracking
    jockey_overall = {}   # jockey -> {runs, wins}
    jockey_course = {}    # (jockey, course) -> {runs, wins}
    jockey_trainer = {}   # (jockey, trainer) -> {runs, wins}
    
    jockey_features = {
        'jockey_career_runs': [],
        'jockey_career_win_rate': [],
        'jockey_course_runs': [],
        'jockey_course_win_rate': [],
        'jockey_trainer_runs': [],
        'jockey_trainer_win_rate': []
    }
    
    # Process each race in chronological order
    for idx, row in df.iterrows():
        jockey = row.get('jockey', 'Unknown')
        course = row.get('course_clean', 'Unknown')
        trainer = row.get('trainer', 'Unknown')
        won = 1 if row.get('won', 0) == 1 else 0
        
        # Get historical stats (BEFORE this race)
        j_stats = jockey_overall.get(jockey, {'runs': 0, 'wins': 0})
        jc_stats = jockey_course.get((jockey, course), {'runs': 0, 'wins': 0})
        jt_stats = jockey_trainer.get((jockey, trainer), {'runs': 0, 'wins': 0})
        
        # Calculate features from historical data
        jockey_features['jockey_career_runs'].append(j_stats['runs'])
        jockey_features['jockey_career_win_rate'].append(
            j_stats['wins'] / j_stats['runs'] if j_stats['runs'] > 0 else 0.0
        )
        
        jockey_features['jockey_course_runs'].append(jc_stats['runs'])
        jockey_features['jockey_course_win_rate'].append(
            jc_stats['wins'] / jc_stats['runs'] if jc_stats['runs'] > 0 else 0.0
        )
        
        jockey_features['jockey_trainer_runs'].append(jt_stats['runs'])
        jockey_features['jockey_trainer_win_rate'].append(
            jt_stats['wins'] / jt_stats['runs'] if jt_stats['runs'] > 0 else 0.0
        )
        
        # Update stats AFTER recording features (prevents leakage)
        if jockey not in jockey_overall:
            jockey_overall[jockey] = {'runs': 0, 'wins': 0}
        jockey_overall[jockey]['runs'] += 1
        jockey_overall[jockey]['wins'] += won
        
        if (jockey, course) not in jockey_course:
            jockey_course[(jockey, course)] = {'runs': 0, 'wins': 0}
        jockey_course[(jockey, course)]['runs'] += 1
        jockey_course[(jockey, course)]['wins'] += won
        
        if (jockey, trainer) not in jockey_trainer:
            jockey_trainer[(jockey, trainer)] = {'runs': 0, 'wins': 0}
        jockey_trainer[(jockey, trainer)]['runs'] += 1
        jockey_trainer[(jockey, trainer)]['wins'] += won
    
    # Add features to dataframe
    for feat_name, feat_values in jockey_features.items():
        df[feat_name] = feat_values
    
    print(f"  Jockey features: jockey_career_win_rate, jockey_course_win_rate, jockey_trainer_win_rate")
    print(f"  Processed {len(jockey_overall):,} unique jockeys")
    
    return df

def engineer_all_features(df):
    """Engineer all features in sequence"""
    print("\nEngineering features for {0:,} records...".format(len(df)))
    
    df = engineer_career_features(df)
    df = engineer_course_distance_form(df)
    # Draw features depend on cd_key and historical wins, compute early
    df = engineer_draw_features(df)
    df = engineer_weight_features(df)
    df = engineer_age_features(df)
    df = engineer_trainer_form(df)
    df = engineer_beaten_lengths_features(df)
    df = engineer_gear_features(df)
    df = engineer_race_condition_features(df)
    df = engineer_class_step(df)
    df = engineer_rating_trend(df)
    df = engineer_recent_form(df)
    df = engineer_recency(df)
    df = engineer_race_context(df)
    df = engineer_jockey_features(df)
    
    print("\n[OK] Feature engineering complete")
    
    return df

def prepare_training_data(df):
    """
    Prepare final training dataset.
    Filter to valid records and select features.
    """
    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    
    # Filter to finishers only (need outcome)
    df_train = df[df['pos_clean'].notna()].copy()
    print(f"\nFiltered to finishers: {len(df_train):,} records")
    
    # Filter to horses with at least 1 previous run (need career stats)
    df_train = df_train[df_train['career_runs'] > 0].copy()
    print(f"Filtered to horses with history: {len(df_train):,} records")
    
    # Define feature columns
    feature_cols = [
        # Career stats
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
        
        # CD form
        'cd_runs', 'cd_win_rate',
        
        # Class
        'class_num', 'class_step',
        
        # Rating
        'or_numeric', 'or_change', 'or_trend_3',
        
        # Recent form
        'avg_last_3_pos', 'wins_last_3',
        
        # Recency
        'days_since_last',
        
        # Race context
        'field_size', 'is_turf', 'going_numeric',
        
        # Race quality (from scorer)
        'race_score',
        # Draw related
        'draw', 'draw_pct', 'draw_group_win_rate',
        
        # Weight features
        'weight_lbs', 'weight_vs_avg', 'is_top_weight', 'weight_change',
        
        # Age features
        'age', 'is_peak_age', 'is_3yo', 'is_veteran', 'age_vs_avg',
        
        # Beaten lengths (historical only - btn_lengths excluded as it leaks outcome)
        'avg_btn_last_3', 'unlucky_last',
        
        # Gear/headgear
        'has_blinkers', 'has_visor', 'first_time_blinkers', 'gear_changed',
        
        # Race conditions
        'is_handicap', 'is_maiden', 'is_pattern', 'prize_log',
        'is_sprint', 'is_mile', 'is_middle', 'is_staying',
        
        # Jockey features (runs only - win rates excluded as they leak same-day outcomes)
        'jockey_career_runs', 'jockey_course_runs', 'jockey_trainer_runs'
    ]
    
    # Target variable
    target_col = 'won'
    
    # Select features and target
    X = df_train[feature_cols].copy()
    y = df_train[target_col].copy()
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    print(f"\nFeature matrix: {X.shape[0]:,} records x {X.shape[1]} features")
    print(f"Target distribution:")
    print(f"  Wins: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Losses: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
    
    # Print feature list
    print(f"\nFeatures ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2}. {col}")
    
    return X, y, feature_cols, df_train

def train_model(X, y, feature_cols, df_train):
    """
    Train XGBoost classifier (or Random Forest if XGBoost unavailable).
    Uses temporal split to prevent data leakage.
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Temporal split: train on older data, test on recent data (last 20%)
    # Sort by date to ensure temporal integrity
    df_sorted = df_train.sort_values('date_dt').reset_index(drop=True)
    
    # Calculate split point (80% train, 20% test)
    split_idx = int(len(df_sorted) * 0.8)
    split_date = df_sorted.loc[split_idx, 'date_dt']
    
    # Create temporal train/test masks
    train_mask = df_train['date_dt'] < split_date
    test_mask = df_train['date_dt'] >= split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTemporal Split:")
    print(f"  Train period: {df_train.loc[train_mask, 'date_dt'].min().date()} to {df_train.loc[train_mask, 'date_dt'].max().date()}")
    print(f"  Test period:  {df_train.loc[test_mask, 'date_dt'].min().date()} to {df_train.loc[test_mask, 'date_dt'].max().date()}")
    print(f"  Split date:   {split_date.date()}")
    print(f"\nTrain set: {len(X_train):,} records ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set:  {len(X_test):,} records ({len(X_test)/len(X)*100:.1f}%)")
    
    # Initialize model
    if HAS_XGBOOST:
        print("\nTraining XGBoost classifier...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        print("\nTraining Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    
    # Train
    model.fit(X_train, y_train)
    print("  [OK] Training complete")
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print("\n" + "-"*60)
    print("MODEL PERFORMANCE")
    print("-"*60)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    train_auc = roc_auc_score(y_train, y_pred_proba_train)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.3f}")
    print(f"  Test:  {test_acc:.3f}")
    
    print(f"\nROC AUC:")
    print(f"  Train: {train_auc:.3f}")
    print(f"  Test:  {test_auc:.3f}")
    
    # Detailed classification report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Loss', 'Win']))
    
    # Feature importance
    if HAS_XGBOOST:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n" + "-"*60)
    print("TOP 10 FEATURES")
    print("-"*60)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:25} {row['importance']:.4f}")
    
    return model, feature_importance

def save_model_and_artifacts(model, feature_importance, feature_cols):
    """Save trained model and metadata"""
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    if HAS_XGBOOST:
        model_path = models_dir / 'horse_win_predictor.json'
        model.save_model(model_path)
    else:
        model_path = models_dir / 'horse_win_predictor.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print(f"\n[SAVED] Model: {model_path}")
    
    # Save feature importance
    importance_path = models_dir / 'feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"[SAVED] Feature importance: {importance_path}")
    
    # Save feature columns (for inference)
    features_path = models_dir / 'feature_columns.txt'
    with open(features_path, 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')
    print(f"[SAVED] Feature columns: {features_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'XGBoost' if HAS_XGBOOST else 'RandomForest',
        'n_features': len(feature_cols),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_columns': feature_cols
    }
    
    metadata_path = models_dir / 'model_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[SAVED] Metadata: {metadata_path}")

def main():
    """Run Phase 3: Build horse prediction model"""
    print("="*60)
    print("PHASE 3: HORSE-LEVEL WIN PREDICTION MODEL")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Engineer features
    df = engineer_all_features(df)
    
    # Prepare training data
    X, y, feature_cols, df_train = prepare_training_data(df)
    
    # Train model (with temporal split)
    model, feature_importance = train_model(X, y, feature_cols, df_train)
    
    # Save
    save_model_and_artifacts(model, feature_importance, feature_cols)
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)
    print("\nModel successfully trained and saved!")
    print("\nNext steps:")
    print("  1. Integrate model into Streamlit UI (predictions.py)")
    print("  2. Add prediction interface for upcoming races")
    print("  3. Visualize feature importance in UI")
    print("  4. Combine with Phase 2 race scorer for complete betting system")

if __name__ == '__main__':
    main()
