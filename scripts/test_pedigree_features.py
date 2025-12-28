#!/usr/bin/env python3
"""Test whether adding pedigree features improves model performance.

Compares:
- Baseline: Current 18 features (no pedigree)
- Enhanced: Current 18 features + pedigree stats

If enhanced model performs better, updates phase3_build_horse_model.py
If not, leaves the model unchanged.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Import existing feature engineering
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.phase3_build_horse_model import (
    load_data,
    engineer_all_features
)

def engineer_pedigree_features(df):
    """
    Calculate sire and dam performance statistics using expanding window.
    NO DATA LEAKAGE: Only uses races before current date.
    """
    print("\nEngineering pedigree features (no lookahead)...")
    
    # Sort by date to ensure temporal order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize pedigree feature columns
    df['sire_win_rate'] = 0.0
    df['sire_place_rate'] = 0.0
    df['dam_win_rate'] = 0.0
    df['dam_place_rate'] = 0.0
    df['damsire_win_rate'] = 0.0
    df['damsire_place_rate'] = 0.0
    
    # Track cumulative stats for each sire/dam/damsire
    sire_stats = {}
    dam_stats = {}
    damsire_stats = {}
    
    for idx, row in df.iterrows():
        sire_id = row['sire_id']
        dam_id = row['dam_id']
        damsire_id = row['damsire_id']
        
        # Get current stats (before this race)
        if sire_id in sire_stats:
            df.loc[idx, 'sire_win_rate'] = sire_stats[sire_id]['win_rate']
            df.loc[idx, 'sire_place_rate'] = sire_stats[sire_id]['place_rate']
        
        if dam_id in dam_stats:
            df.loc[idx, 'dam_win_rate'] = dam_stats[dam_id]['win_rate']
            df.loc[idx, 'dam_place_rate'] = dam_stats[dam_id]['place_rate']
        
        if damsire_id in damsire_stats:
            df.loc[idx, 'damsire_win_rate'] = damsire_stats[damsire_id]['win_rate']
            df.loc[idx, 'damsire_place_rate'] = damsire_stats[damsire_id]['place_rate']
        
        # Update stats AFTER using them (for next race)
        if pd.notna(row['pos_clean']):
            won = int(row['pos_clean'] == 1)
            placed = int(row['pos_clean'] <= 3)
            
            # Update sire
            if sire_id not in sire_stats:
                sire_stats[sire_id] = {'runs': 0, 'wins': 0, 'places': 0, 'win_rate': 0.0, 'place_rate': 0.0}
            sire_stats[sire_id]['runs'] += 1
            sire_stats[sire_id]['wins'] += won
            sire_stats[sire_id]['places'] += placed
            sire_stats[sire_id]['win_rate'] = sire_stats[sire_id]['wins'] / sire_stats[sire_id]['runs']
            sire_stats[sire_id]['place_rate'] = sire_stats[sire_id]['places'] / sire_stats[sire_id]['runs']
            
            # Update dam
            if dam_id not in dam_stats:
                dam_stats[dam_id] = {'runs': 0, 'wins': 0, 'places': 0, 'win_rate': 0.0, 'place_rate': 0.0}
            dam_stats[dam_id]['runs'] += 1
            dam_stats[dam_id]['wins'] += won
            dam_stats[dam_id]['places'] += placed
            dam_stats[dam_id]['win_rate'] = dam_stats[dam_id]['wins'] / dam_stats[dam_id]['runs']
            dam_stats[dam_id]['place_rate'] = dam_stats[dam_id]['places'] / dam_stats[dam_id]['runs']
            
            # Update damsire
            if damsire_id not in damsire_stats:
                damsire_stats[damsire_id] = {'runs': 0, 'wins': 0, 'places': 0, 'win_rate': 0.0, 'place_rate': 0.0}
            damsire_stats[damsire_id]['runs'] += 1
            damsire_stats[damsire_id]['wins'] += won
            damsire_stats[damsire_id]['places'] += placed
            damsire_stats[damsire_id]['win_rate'] = damsire_stats[damsire_id]['wins'] / damsire_stats[damsire_id]['runs']
            damsire_stats[damsire_id]['place_rate'] = damsire_stats[damsire_id]['places'] / damsire_stats[damsire_id]['runs']
    
    print(f"  Sire features: sire_win_rate, sire_place_rate")
    print(f"  Dam features: dam_win_rate, dam_place_rate")
    print(f"  Damsire features: damsire_win_rate, damsire_place_rate")
    print(f"  NO LOOKAHEAD BIAS - using only historical data")
    
    return df


def train_and_evaluate(X, y, feature_cols, model_name):
    """Train model and return validation AUC"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    print(f"Features: {len(feature_cols)}")
    
    # Temporal split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Train
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
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n✓ Validation AUC-ROC: {auc:.4f}")
    
    return auc, model


def main():
    print("="*60)
    print("PEDIGREE FEATURE EVALUATION")
    print("="*60)
    
    # Load and engineer baseline features
    df = load_data()
    df = engineer_all_features(df)
    
    # Filter to valid training data
    df_train = df[df['pos_clean'].notna()].copy()
    df_train = df_train[df_train['career_runs'] > 0].copy()
    
    # Baseline feature set (current 18 features)
    baseline_features = [
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
        'cd_runs', 'cd_win_rate',
        'class_num', 'class_step',
        'or_numeric', 'or_change', 'or_trend_3',
        'avg_last_3_pos', 'wins_last_3',
        'days_since_last',
        'field_size', 'is_turf', 'going_numeric',
        'race_score'
    ]
    
    # Create target
    df_train['target'] = (df_train['pos_clean'] == 1).astype(int)
    
    # Filter to complete cases (baseline)
    valid_baseline = df_train[baseline_features].notna().all(axis=1)
    df_baseline = df_train[valid_baseline].copy()
    X_baseline = df_baseline[baseline_features]
    y_baseline = df_baseline['target']
    
    print(f"\nBaseline data: {len(X_baseline):,} valid records")
    
    # Test 1: Baseline model
    auc_baseline, model_baseline = train_and_evaluate(
        X_baseline, y_baseline, baseline_features, "BASELINE (18 features, no pedigree)"
    )
    
    # Add pedigree features
    df_train = engineer_pedigree_features(df_train)
    
    # Enhanced feature set
    pedigree_features = ['sire_win_rate', 'sire_place_rate', 'dam_win_rate', 
                         'dam_place_rate', 'damsire_win_rate', 'damsire_place_rate']
    enhanced_features = baseline_features + pedigree_features
    
    # Filter to complete cases (enhanced)
    valid_enhanced = df_train[enhanced_features].notna().all(axis=1)
    df_enhanced = df_train[valid_enhanced].copy()
    X_enhanced = df_enhanced[enhanced_features]
    y_enhanced = df_enhanced['target']
    
    print(f"\nEnhanced data: {len(X_enhanced):,} valid records")
    
    # Test 2: Enhanced model
    auc_enhanced, model_enhanced = train_and_evaluate(
        X_enhanced, y_enhanced, enhanced_features, "ENHANCED (24 features, with pedigree)"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nBaseline AUC:  {auc_baseline:.4f} ({len(baseline_features)} features)")
    print(f"Enhanced AUC:  {auc_enhanced:.4f} ({len(enhanced_features)} features)")
    print(f"\nDifference:    {auc_enhanced - auc_baseline:+.4f}")
    print(f"% Improvement: {((auc_enhanced - auc_baseline) / auc_baseline * 100):+.2f}%")
    
    # Decision threshold: Keep if improvement >= 0.5% (0.005 AUC points)
    THRESHOLD = 0.005
    
    print("\n" + "="*60)
    print("DECISION")
    print("="*60)
    
    if auc_enhanced - auc_baseline >= THRESHOLD:
        print(f"\n✅ KEEP pedigree features (improvement: {auc_enhanced - auc_baseline:.4f} >= {THRESHOLD})")
        print("\nPedigree features improve model performance!")
        print("Update phase3_build_horse_model.py to include these features.")
        
        # Show feature importance for pedigree features
        importances = model_enhanced.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': enhanced_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop pedigree features by importance:")
        pedigree_importance = feature_importance[feature_importance['feature'].isin(pedigree_features)]
        for _, row in pedigree_importance.iterrows():
            print(f"  {row['feature']:25} {row['importance']:.4f}")
        
        return True
    else:
        print(f"\n❌ REMOVE pedigree features (improvement: {auc_enhanced - auc_baseline:.4f} < {THRESHOLD})")
        print("\nPedigree features do not meaningfully improve predictions.")
        print("Keep the existing 18-feature model.")
        return False


if __name__ == '__main__':
    keep_pedigree = main()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    
    if keep_pedigree:
        print("\nNext steps:")
        print("1. Manually add engineer_pedigree_features() to phase3_build_horse_model.py")
        print("2. Update feature list to include pedigree columns")
        print("3. Retrain all models (win, place, show)")
    else:
        print("\nNo action needed - current model is optimal without pedigree features.")
