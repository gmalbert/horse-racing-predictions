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

def engineer_all_features(df):
    """Engineer all features in sequence"""
    print("\nEngineering features for {0:,} records...".format(len(df)))
    
    df = engineer_career_features(df)
    df = engineer_course_distance_form(df)
    df = engineer_class_step(df)
    df = engineer_rating_trend(df)
    df = engineer_recent_form(df)
    df = engineer_recency(df)
    df = engineer_race_context(df)
    
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
        'race_score'
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

def train_model(X, y, feature_cols):
    """
    Train XGBoost classifier (or Random Forest if XGBoost unavailable).
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train):,} records")
    print(f"Test set:  {len(X_test):,} records")
    
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
    
    # Train model
    model, feature_importance = train_model(X, y, feature_cols)
    
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
