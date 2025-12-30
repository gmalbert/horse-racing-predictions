#!/usr/bin/env python3
"""Compare model performance with and without jockey features.

Side-by-side comparison:
- Model A: Current features (no jockey stats)
- Model B: Current features + jockey features

Metrics:
- AUC-ROC
- Log Loss
- Accuracy
- Calibration
- Feature importance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.phase3_build_horse_model import (
    load_data,
    engineer_all_features
)

HISTORICAL_DATA = project_root / "data" / "processed" / "race_scores_with_betting_tiers.parquet"


def engineer_jockey_features(df):
    """Engineer jockey-specific features from historical data."""
    print("\n" + "="*60)
    print("ENGINEERING JOCKEY FEATURES")
    print("="*60)
    
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Create datetime with race time to prevent same-day leakage
    # Combine date + off time (HH:MM) into full timestamp
    df['datetime_full'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['off'].astype(str), 
        errors='coerce'
    )
    
    # Sort by full timestamp to ensure races processed in chronological order
    # This prevents using stats from later races on the same day
    df = df.sort_values('datetime_full')
    
    # Initialize jockey features
    jockey_features = {
        'jockey_career_runs': [],
        'jockey_career_win_rate': [],
        'jockey_course_runs': [],
        'jockey_course_win_rate': [],
        'jockey_trainer_runs': [],
        'jockey_trainer_win_rate': []
    }
    
    # Track jockey stats up to each race date
    jockey_overall = {}  # jockey -> {runs, wins}
    jockey_course = {}   # (jockey, course) -> {runs, wins}
    jockey_trainer = {}  # (jockey, trainer) -> {runs, wins}
    
    print(f"Processing {len(df):,} records...")
    
    for idx, row in df.iterrows():
        jockey = row.get('jockey', 'Unknown')
        course = row.get('course', 'Unknown')
        trainer = row.get('trainer', 'Unknown')
        won = 1 if row.get('won', 0) == 1 else 0
        
        # Get historical stats (prior to this race)
        j_stats = jockey_overall.get(jockey, {'runs': 0, 'wins': 0})
        jc_stats = jockey_course.get((jockey, course), {'runs': 0, 'wins': 0})
        jt_stats = jockey_trainer.get((jockey, trainer), {'runs': 0, 'wins': 0})
        
        # Calculate rates
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
        
        # Update stats after this race
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
    
    print(f"[OK] Created {len(jockey_features)} jockey features")
    print(f"  - Unique jockeys: {len(jockey_overall):,}")
    print(f"  - Jockey-course combinations: {len(jockey_course):,}")
    print(f"  - Jockey-trainer combinations: {len(jockey_trainer):,}")
    
    return df


def prepare_comparison_data(df):
    """Prepare training data with and without jockey features."""
    print("\n" + "="*60)
    print("PREPARING COMPARISON DATA")
    print("="*60)
    
    # Filter to finishers only
    df_train = df[df['pos_clean'].notna()].copy()
    print(f"Finishers: {len(df_train):,}")
    
    # Filter to horses with history
    df_train = df_train[df_train['career_runs'] > 0].copy()
    print(f"Horses with history: {len(df_train):,}")
    
    # Base features (current model)
    base_features = [
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
        'cd_runs', 'cd_win_rate',
        'class_num', 'class_step',
        'or_numeric', 'or_change', 'or_trend_3',
        'avg_last_3_pos', 'wins_last_3',
        'days_since_last',
        'field_size', 'is_turf', 'going_numeric',
        'race_score'
    ]
    
    # Jockey features
    jockey_features = [
        'jockey_career_runs',
        'jockey_career_win_rate',
        'jockey_course_runs',
        'jockey_course_win_rate',
        'jockey_trainer_runs',
        'jockey_trainer_win_rate'
    ]
    
    # Combined features
    combined_features = base_features + jockey_features
    
    # Target
    target_col = 'won'
    
    # Check for missing columns
    missing_base = [f for f in base_features if f not in df_train.columns]
    missing_jockey = [f for f in jockey_features if f not in df_train.columns]
    
    if missing_base:
        print(f"\n[ERROR] Missing base features: {missing_base}")
        return None
    
    if missing_jockey:
        print(f"\n[ERROR] Missing jockey features: {missing_jockey}")
        return None
    
    # Filter to valid records (no NaN in features)
    df_valid = df_train[df_train[combined_features + [target_col]].notna().all(axis=1)].copy()
    print(f"Valid records (no NaN): {len(df_valid):,}")
    
    # Temporal split (80% train, 20% validation)
    cutoff_date = df_valid['date'].quantile(0.8)
    train_mask = df_valid['date'] <= cutoff_date
    
    print(f"\nTrain cutoff date: {cutoff_date}")
    print(f"Train: {train_mask.sum():,} records")
    print(f"Validation: {(~train_mask).sum():,} records")
    
    return df_valid, base_features, jockey_features, combined_features, target_col, train_mask


def train_and_evaluate(X_train, y_train, X_val, y_val, model_name):
    """Train XGBoost and return metrics."""
    print(f"\nTraining {model_name}...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # Metrics
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred_proba)
    
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Brier:    {brier:.4f}")
    
    return {
        'model': model,
        'auc': auc,
        'logloss': logloss,
        'accuracy': accuracy,
        'brier': brier,
        'y_pred_proba': y_pred_proba
    }


def compare_models(results_base, results_jockey, feature_cols_base, feature_cols_jockey):
    """Compare model performance and feature importance."""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 60)
    print(f"{'Metric':<20} {'Base Model':<15} {'+ Jockey':<15} {'Improvement':<15}")
    print("-" * 60)
    
    # AUC-ROC
    auc_diff = results_jockey['auc'] - results_base['auc']
    auc_pct = (auc_diff / results_base['auc']) * 100
    print(f"{'AUC-ROC':<20} {results_base['auc']:>14.4f} {results_jockey['auc']:>14.4f} {auc_diff:>+14.4f} ({auc_pct:+.2f}%)")
    
    # Log Loss (lower is better)
    ll_diff = results_base['logloss'] - results_jockey['logloss']
    ll_pct = (ll_diff / results_base['logloss']) * 100
    print(f"{'Log Loss':<20} {results_base['logloss']:>14.4f} {results_jockey['logloss']:>14.4f} {ll_diff:>+14.4f} ({ll_pct:+.2f}%)")
    
    # Accuracy
    acc_diff = results_jockey['accuracy'] - results_base['accuracy']
    acc_pct = (acc_diff / results_base['accuracy']) * 100
    print(f"{'Accuracy':<20} {results_base['accuracy']:>14.4f} {results_jockey['accuracy']:>14.4f} {acc_diff:>+14.4f} ({acc_pct:+.2f}%)")
    
    # Brier Score (lower is better)
    brier_diff = results_base['brier'] - results_jockey['brier']
    brier_pct = (brier_diff / results_base['brier']) * 100
    print(f"{'Brier Score':<20} {results_base['brier']:>14.4f} {results_jockey['brier']:>14.4f} {brier_diff:>+14.4f} ({brier_pct:+.2f}%)")
    
    print("\n🎯 RECOMMENDATION")
    print("-" * 60)
    
    # Simple recommendation based on AUC improvement
    if auc_diff > 0.005:
        print(f"✅ ADD JOCKEY FEATURES - Significant improvement in AUC (+{auc_diff:.4f})")
    elif auc_diff > 0.001:
        print(f"⚠️  MARGINAL IMPROVEMENT - Small AUC gain (+{auc_diff:.4f}), consider complexity vs benefit")
    else:
        print(f"❌ SKIP JOCKEY FEATURES - No meaningful improvement (AUC: {auc_diff:+.4f})")
    
    # Feature importance for jockey model
    print("\n🔝 TOP 15 FEATURES (Model with Jockey Stats)")
    print("-" * 60)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols_jockey,
        'importance': results_jockey['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(15).iterrows():
        is_jockey = 'jockey' in row['feature']
        marker = "🏇" if is_jockey else "  "
        print(f"{marker} {row['feature']:<30} {row['importance']:>10.4f}")
    
    # Count jockey features in top 15
    jockey_in_top15 = sum(['jockey' in f for f in importance_df.head(15)['feature']])
    print(f"\n{jockey_in_top15} of top 15 features are jockey-related")
    
    return auc_diff > 0.001  # Return True if meaningful improvement


def main():
    """Run side-by-side comparison."""
    print("="*60)
    print("JOCKEY FEATURES MODEL COMPARISON")
    print("="*60)
    
    # Load data
    print("\n1. Loading historical data...")
    df = load_data()
    
    # Engineer standard features
    print("\n2. Engineering standard features...")
    df = engineer_all_features(df)
    
    # Engineer jockey features
    print("\n3. Engineering jockey features...")
    df = engineer_jockey_features(df)
    
    # Prepare comparison data
    print("\n4. Preparing comparison datasets...")
    result = prepare_comparison_data(df)
    
    if result is None:
        print("[ERROR] Failed to prepare data")
        return
    
    df_valid, base_features, jockey_features, combined_features, target_col, train_mask = result
    
    # Split data
    X_base_train = df_valid.loc[train_mask, base_features]
    X_jockey_train = df_valid.loc[train_mask, combined_features]
    y_train = df_valid.loc[train_mask, target_col]
    
    X_base_val = df_valid.loc[~train_mask, base_features]
    X_jockey_val = df_valid.loc[~train_mask, combined_features]
    y_val = df_valid.loc[~train_mask, target_col]
    
    print(f"\nBase model: {len(base_features)} features")
    print(f"Jockey model: {len(combined_features)} features ({len(jockey_features)} jockey features added)")
    
    # Train models
    print("\n5. Training models...")
    print("\n" + "="*60)
    print("MODEL A: BASE (NO JOCKEY STATS)")
    print("="*60)
    results_base = train_and_evaluate(X_base_train, y_train, X_base_val, y_val, "Base Model")
    
    print("\n" + "="*60)
    print("MODEL B: WITH JOCKEY FEATURES")
    print("="*60)
    results_jockey = train_and_evaluate(X_jockey_train, y_train, X_jockey_val, y_val, "Jockey Model")
    
    # Compare
    print("\n6. Comparing results...")
    should_add = compare_models(results_base, results_jockey, base_features, combined_features)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    
    if should_add:
        print("\n✅ Recommendation: ADD jockey features to production model")
    else:
        print("\n❌ Recommendation: KEEP current model (no jockey features)")


if __name__ == "__main__":
    main()
