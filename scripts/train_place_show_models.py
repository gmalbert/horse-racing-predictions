#!/usr/bin/env python3
"""Train Place and Show prediction models.

Place = finish in top 2 (or top 3 for large fields)
Show = finish in top 3 (or top 4 for large fields)
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from phase3 script
from scripts.phase3_build_horse_model import (
    load_data,
    engineer_all_features
)

MODELS_DIR = project_root / "models"
HISTORICAL_DATA = project_root / "data" / "processed" / "race_scores_with_betting_tiers.parquet"


def create_place_target(df):
    """Create place target: top 2 for small fields (<8), top 3 for large fields."""
    df = df.copy()
    
    # Clean position data to numeric
    df['pos_clean'] = pd.to_numeric(df['pos'], errors='coerce')
    
    # Standard UK/IE rules: 
    # - 5-7 runners: place = 1st or 2nd
    # - 8+ runners: place = 1st, 2nd, or 3rd
    # - 16+ runners (handicaps): place = 1st, 2nd, 3rd, or 4th
    
    df['place_threshold'] = df['field_size'].apply(
        lambda x: 2 if x < 8 else (4 if x >= 16 else 3)
    )
    
    df['placed'] = (df['pos_clean'] <= df['place_threshold']).astype(int)
    
    return df


def create_show_target(df):
    """Create show target: top 3 for small fields, top 4 for large fields."""
    df = df.copy()
    
    # Clean position data to numeric
    df['pos_clean'] = pd.to_numeric(df['pos'], errors='coerce')
    
    # Show is typically one position beyond place
    # - 5-7 runners: show = 1st, 2nd, or 3rd
    # - 8-15 runners: show = 1st, 2nd, 3rd, or 4th
    # - 16+ runners: show = 1st through 5th
    
    df['show_threshold'] = df['field_size'].apply(
        lambda x: 3 if x < 8 else (5 if x >= 16 else 4)
    )
    
    df['showed'] = (df['pos_clean'] <= df['show_threshold']).astype(int)
    
    return df


def train_place_model(df, feature_cols):
    """Train place prediction model."""
    print("\n" + "="*60)
    print("TRAINING PLACE MODEL")
    print("="*60)
    
    # Create place target
    df = create_place_target(df)
    
    # Filter valid data
    valid = df[df['pos_clean'].notna() & df[feature_cols].notna().all(axis=1)].copy()
    
    print(f"\nTotal valid records: {len(valid):,}")
    print(f"Place rate: {valid['placed'].mean():.1%}")
    
    # Prepare data
    X = valid[feature_cols]
    y = valid['placed']
    
    # Temporal split (last 20% for validation)
    cutoff_date = valid['date'].quantile(0.8)
    train_mask = valid['date'] <= cutoff_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    
    print(f"\nTraining set: {len(X_train):,} races")
    print(f"Validation set: {len(X_val):,} races")
    
    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_proba)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n[VALIDATION METRICS]")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Baseline (always predict no place): {(1 - y_val.mean()):.4f}")
    
    return model


def train_show_model(df, feature_cols):
    """Train show prediction model."""
    print("\n" + "="*60)
    print("TRAINING SHOW MODEL")
    print("="*60)
    
    # Create show target
    df = create_show_target(df)
    
    # Filter valid data
    valid = df[df['pos_clean'].notna() & df[feature_cols].notna().all(axis=1)].copy()
    
    print(f"\nTotal valid records: {len(valid):,}")
    print(f"Show rate: {valid['showed'].mean():.1%}")
    
    # Prepare data
    X = valid[feature_cols]
    y = valid['showed']
    
    # Temporal split (last 20% for validation)
    cutoff_date = valid['date'].quantile(0.8)
    train_mask = valid['date'] <= cutoff_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    
    print(f"\nTraining set: {len(X_train):,} races")
    print(f"Validation set: {len(X_val):,} races")
    
    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_proba)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n[VALIDATION METRICS]")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Baseline (always predict no show): {(1 - y_val.mean()):.4f}")
    
    return model


def main():
    """Train place and show models."""
    print("="*60)
    print("TRAIN PLACE & SHOW PREDICTION MODELS")
    print("="*60)
    
    # Load data
    print("\nLoading historical data...")
    df = load_data()
    
    # Engineer features (same as win model)
    df = engineer_all_features(df)
    
    # Load feature columns from win model
    feature_cols_file = MODELS_DIR / "feature_columns.txt"
    with open(feature_cols_file, 'r') as f:
        feature_cols = [line.strip() for line in f]
    
    print(f"\nUsing {len(feature_cols)} features from win model")
    
    # Train place model
    place_model = train_place_model(df, feature_cols)
    
    # Save place model
    place_model_file = MODELS_DIR / "horse_place_predictor.pkl"
    with open(place_model_file, 'wb') as f:
        pickle.dump(place_model, f)
    print(f"\n[SAVED] Place model: {place_model_file}")
    
    # Train show model
    show_model = train_show_model(df, feature_cols)
    
    # Save show model
    show_model_file = MODELS_DIR / "horse_show_predictor.pkl"
    with open(show_model_file, 'wb') as f:
        pickle.dump(show_model, f)
    print(f"\n[SAVED] Show model: {show_model_file}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModels saved:")
    print(f"  - {place_model_file}")
    print(f"  - {show_model_file}")
    print("\nNext: Update predict_todays_races.py to use all three models")


if __name__ == '__main__':
    main()
