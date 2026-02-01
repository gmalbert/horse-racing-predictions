#!/usr/bin/env python3
"""
Model Comparison: Baseline vs Enhanced Features

Trains multiple models to measure impact of new feature groups:
1. Baseline (72 features from v2.0)
2. + Enhanced Form (6 features)
3. + Connections V2 (13 features)
4. Full Model (91 features total)

Reports AUC improvement for each feature group.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

from sklearn.metrics import roc_auc_score, accuracy_score

# Paths
DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
OUTPUT_FILE = MODEL_DIR / 'feature_impact_analysis.json'


BASELINE_FEATURES = [
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
    # Race quality
    'race_score',
    # Draw
    'draw', 'draw_pct', 'draw_group_win_rate',
    # Weight
    'weight_lbs', 'weight_vs_avg', 'is_top_weight', 'weight_change',
    # Age
    'age', 'is_peak_age', 'is_3yo', 'is_veteran', 'age_vs_avg',
    # Beaten lengths
    'avg_btn_last_3', 'unlucky_last',
    # Gear
    'has_blinkers', 'has_visor', 'first_time_blinkers', 'gear_changed',
    # Race conditions
    'is_handicap', 'is_maiden', 'is_pattern', 'prize_log',
    'is_sprint', 'is_mile', 'is_middle', 'is_staying',
    # Jockey
    'jockey_career_runs', 'jockey_course_runs', 'jockey_trainer_runs',
    # Pedigree (6)
    'sire_win_rate', 'sire_place_rate', 'sire_surface_match',
    'sire_distance_match', 'sire_going_match', 'sire_class_match',
    # Pace (9)
    'pace_style_leader', 'pace_style_presser', 'pace_style_closer', 'pace_style_midpack',
    'race_leader_count', 'race_closer_count',
    'style_advantage', 'sprint_specialist', 'staying_specialist',
    # Original form (10)
    'jockey_form_14d', 'jockey_form_30d', 'jockey_in_form', 'jockey_course_form_30d',
    'trainer_form_14d', 'trainer_form_30d', 'trainer_in_form', 'trainer_course_form_30d',
    'jockey_trainer_form_30d', 'connections_in_form'
]

ENHANCED_FORM_FEATURES = [
    'weighted_pos_avg', 'pos_pct_last_3', 'form_consistency',
    'form_trend', 'form_at_class', 'runs_at_class'
]

CONNECTIONS_V2_FEATURES = [
    'jockey_form_14d_v2', 'jockey_form_30d_v2', 'jockey_hot_v2',
    'trainer_form_14d_v2', 'trainer_form_30d_v2', 'trainer_hot_v2',
    'combo_form_30d_v2', 'combo_hot_v2',
    'jockey_runs_14d_v2', 'jockey_runs_30d_v2',
    'trainer_runs_14d_v2', 'trainer_runs_30d_v2',
    'combo_runs_30d_v2'
]


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    """Train a model and return performance metrics"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    if HAS_XGBOOST:
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
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nROC AUC:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test:  {test_acc:.4f}")
    
    return {
        'model_name': model_name,
        'n_features': X_train.shape[1],
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'model': model
    }


def main():
    print("="*70)
    print("FEATURE IMPACT ANALYSIS")
    print("="*70)
    
    # Load data
    data_file = DATA_DIR / 'race_scores_connections_v2.parquet'
    print(f"\nLoading: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"  Loaded: {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    
    # Ensure date_dt
    if 'date_dt' not in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'])
    
    # Create target
    if 'won' not in df.columns:
        df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # Filter to valid records
    df = df[df['won'].notna()].copy()
    
    # Temporal split (80/20)
    df_sorted = df.sort_values('date_dt').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    split_date = df_sorted.loc[split_idx, 'date_dt']
    
    train_mask = df['date_dt'] < split_date
    test_mask = df['date_dt'] >= split_date
    
    print(f"\nTemporal Split:")
    print(f"  Train: {df.loc[train_mask, 'date_dt'].min().date()} to {df.loc[train_mask, 'date_dt'].max().date()}")
    print(f"  Test:  {df.loc[test_mask, 'date_dt'].min().date()} to {df.loc[test_mask, 'date_dt'].max().date()}")
    print(f"  Train: {train_mask.sum():,} records")
    print(f"  Test:  {test_mask.sum():,} records")
    
    # Common y
    y_train = df.loc[train_mask, 'won'].values
    y_test = df.loc[test_mask, 'won'].values
    
    results = []
    
    # Model 1: Baseline (72 features)
    print("\n" + "="*70)
    print("MODEL 1: BASELINE (72 features)")
    print("="*70)
    baseline_available = [f for f in BASELINE_FEATURES if f in df.columns]
    print(f"Available: {len(baseline_available)}/{len(BASELINE_FEATURES)} features")
    
    X_train_baseline = df.loc[train_mask, baseline_available].fillna(0)
    X_test_baseline = df.loc[test_mask, baseline_available].fillna(0)
    
    result1 = train_and_evaluate(X_train_baseline, y_train, X_test_baseline, y_test, "Baseline (v2.0)")
    results.append(result1)
    baseline_auc = result1['test_auc']
    
    # Model 2: + Enhanced Form
    print("\n" + "="*70)
    print("MODEL 2: BASELINE + ENHANCED FORM")
    print("="*70)
    form_available = [f for f in ENHANCED_FORM_FEATURES if f in df.columns]
    print(f"Adding: {len(form_available)} enhanced form features")
    
    features_2 = baseline_available + form_available
    X_train_2 = df.loc[train_mask, features_2].fillna(0)
    X_test_2 = df.loc[test_mask, features_2].fillna(0)
    
    result2 = train_and_evaluate(X_train_2, y_train, X_test_2, y_test, "Baseline + Enhanced Form")
    results.append(result2)
    
    print(f"\n📊 Impact of Enhanced Form:")
    print(f"  AUC Change: {result2['test_auc'] - baseline_auc:+.4f}")
    print(f"  Relative Improvement: {(result2['test_auc'] - baseline_auc) / baseline_auc * 100:+.2f}%")
    
    # Model 3: + Connections V2
    print("\n" + "="*70)
    print("MODEL 3: BASELINE + ENHANCED FORM + CONNECTIONS V2")
    print("="*70)
    connections_available = [f for f in CONNECTIONS_V2_FEATURES if f in df.columns]
    print(f"Adding: {len(connections_available)} connections V2 features")
    
    features_3 = features_2 + connections_available
    X_train_3 = df.loc[train_mask, features_3].fillna(0)
    X_test_3 = df.loc[test_mask, features_3].fillna(0)
    
    result3 = train_and_evaluate(X_train_3, y_train, X_test_3, y_test, "Full Model (v2.1)")
    results.append(result3)
    
    print(f"\n📊 Impact of Connections V2:")
    print(f"  AUC Change: {result3['test_auc'] - result2['test_auc']:+.4f}")
    print(f"  Relative Improvement: {(result3['test_auc'] - result2['test_auc']) / result2['test_auc'] * 100:+.2f}%")
    
    print(f"\n📊 Total Impact (vs Baseline):")
    print(f"  AUC Change: {result3['test_auc'] - baseline_auc:+.4f}")
    print(f"  Relative Improvement: {(result3['test_auc'] - baseline_auc) / baseline_auc * 100:+.2f}%")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Model':<30} {'Features':<10} {'Train AUC':<12} {'Test AUC':<12} {'Δ AUC':<10}")
    print("-" * 70)
    
    for i, res in enumerate(results):
        delta = res['test_auc'] - baseline_auc if i > 0 else 0
        print(f"{res['model_name']:<30} {res['n_features']:<10} {res['train_auc']:<12.4f} {res['test_auc']:<12.4f} {delta:+.4f}")
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': [
            {k: v for k, v in r.items() if k != 'model'}  # Exclude model object
            for r in results
        ],
        'baseline_auc': baseline_auc,
        'final_auc': result3['test_auc'],
        'total_improvement': result3['test_auc'] - baseline_auc,
        'relative_improvement_pct': (result3['test_auc'] - baseline_auc) / baseline_auc * 100
    }
    
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    
    # Feature importance for final model
    print("\n" + "="*70)
    print("TOP 20 FEATURES (Full Model)")
    print("="*70)
    
    if HAS_XGBOOST:
        importances = result3['model'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features_3,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n")
        for i, row in feature_importance.head(20).iterrows():
            is_new = row['feature'] in (ENHANCED_FORM_FEATURES + CONNECTIONS_V2_FEATURES)
            marker = "🆕" if is_new else "  "
            print(f"{marker} {row['feature']:<30} {row['importance']:.4f}")
        
        # Save feature importance
        feature_importance.to_csv(MODEL_DIR / 'feature_importance_v2.1.csv', index=False)
        print(f"\n✓ Feature importance saved to: {MODEL_DIR / 'feature_importance_v2.1.csv'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
