#!/usr/bin/env python3
"""scripts/calibrate_model.py - Add probability calibration."""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import joblib

# Import feature engineering from training script
sys.path.insert(0, str(Path(__file__).parent))
from phase3_build_horse_model import engineer_all_features

def get_historical_data_path():
    """Get the best available historical data file (same logic as training)"""
    from pathlib import Path
    data_dir = Path('data/processed')
    
    # Match the priority order from phase3_build_horse_model.py
    or_context_path = data_dir / 'race_scores_or_context.parquet'
    going_pref_path = data_dir / 'race_scores_going_pref.parquet'
    pedigree_path = data_dir / 'race_scores_pedigree.parquet'
    connections_v2_path = data_dir / 'race_scores_connections_v2.parquet'
    no_leak_path = data_dir / 'race_scores_with_all_features_no_leakage.parquet'
    legacy_path = data_dir / 'race_scores.parquet'
    
    if or_context_path.exists():
        print(f"   [OK] Using latest data: {or_context_path.name}")
        return or_context_path
    elif going_pref_path.exists():
        print(f"   [OK] Using latest data: {going_pref_path.name}")
        return going_pref_path
    elif pedigree_path.exists():
        print(f"   [OK] Using latest data: {pedigree_path.name}")
        return pedigree_path
    elif connections_v2_path.exists():
        print(f"   [OK] Using latest data: {connections_v2_path.name}")
        return connections_v2_path
    elif no_leak_path.exists():
        print(f"   [OK] Using enhanced data: {no_leak_path.name}")
        return no_leak_path
    elif legacy_path.exists():
        print(f"   [OK] Using legacy data: {legacy_path.name}")
        return legacy_path
    else:
        raise FileNotFoundError("No historical race data found in data/processed/")

def load_feature_columns():
    """Load feature column names from models/feature_columns.txt"""
    with open('models/feature_columns.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def calibrate_and_save():
    """Calibrate the existing model."""
    
    print("="*60)
    print("MODEL CALIBRATION")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model('models/horse_win_predictor.json')
    print("   [OK] Model loaded")
    
    # Load training data
    print("\n2. Loading historical data...")
    data_path = get_historical_data_path()
    df = pd.read_parquet(data_path)
    print(f"   [OK] Loaded {len(df):,} records from {data_path.name}")
    
    # Engineer missing features (same as training)
    print("\n3. Engineering features...")
    print("   Running feature engineering pipeline...")
    df = engineer_all_features(df)
    print("   [OK] Feature engineering complete")
    
    # Use last 3 months for calibration (held out)
    print("\n4. Preparing calibration dataset...")
    df['date_dt'] = pd.to_datetime(df['date'])
    calib_start = df['date_dt'].max() - pd.DateOffset(months=3)
    
    calib_data = df[df['date_dt'] >= calib_start]
    train_data = df[df['date_dt'] < calib_start]
    
    print(f"   Training data: {len(train_data):,} records (before {calib_start.date()})")
    print(f"   Calibration data: {len(calib_data):,} records (last 3 months)")
    
    # Get features and target
    print("\n5. Loading feature columns...")
    feature_cols = load_feature_columns()
    print(f"   [OK] Loaded {len(feature_cols)} feature columns")
    
    # Filter to available features
    available_features = [col for col in feature_cols if col in calib_data.columns]
    missing_features = [col for col in feature_cols if col not in calib_data.columns]
    
    if missing_features:
        print(f"   [!] Warning: {len(missing_features)} features not found in data:")
        for feat in missing_features[:5]:
            print(f"      - {feat}")
        if len(missing_features) > 5:
            print(f"      ... and {len(missing_features) - 5} more")
    
    print(f"   Using {len(available_features)} features")
    
    X_calib = calib_data[available_features]
    y_calib = (calib_data['pos_clean'] == 1).astype(int)
    
    print(f"   Target distribution: {y_calib.value_counts().to_dict()}")
    print(f"   Win rate: {y_calib.mean()*100:.2f}%")
    
    # Check if we have enough features for calibration
    if len(available_features) < 50:  # Require at least 50 features for meaningful calibration
        print(f"\n[!] ERROR: Insufficient features for calibration")
        print(f"   Model was trained on 120 features, but data only has {len(available_features)} matching features")
        print(f"   This suggests the training data has been modified since model training")
        print(f"\n   To fix this:")
        print(f"   1. Retrain the model on current data: python scripts/phase3_build_horse_model.py")
        print(f"   2. Or restore the original training data with all features")
        print(f"\n   Skipping calibration for now...")
        return
    
    # Calibrate using isotonic regression
    print("\n6. Calibrating model (isotonic regression)...")
    # Temporarily set feature names to match available features for calibration
    original_feature_names = model.get_booster().feature_names
    model.get_booster().feature_names = available_features
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_calib, y_calib)
    # Restore original feature names
    model.get_booster().feature_names = original_feature_names
    print("   [OK] Calibration complete")
    
    # Verify calibration
    print("\n7. Verifying calibration...")
    y_pred_uncalib = model.predict_proba(X_calib)[:, 1]
    y_pred_calib = calibrated.predict_proba(X_calib)[:, 1]
    
    # Calculate calibration metrics
    from sklearn.metrics import brier_score_loss, log_loss
    
    brier_uncalib = brier_score_loss(y_calib, y_pred_uncalib)
    brier_calib = brier_score_loss(y_calib, y_pred_calib)
    
    logloss_uncalib = log_loss(y_calib, y_pred_uncalib)
    logloss_calib = log_loss(y_calib, y_pred_calib)
    
    print(f"   Brier Score (uncalibrated): {brier_uncalib:.4f}")
    print(f"   Brier Score (calibrated):   {brier_calib:.4f}")
    print(f"   Improvement: {((brier_uncalib - brier_calib) / brier_uncalib * 100):.2f}%")
    print()
    print(f"   Log Loss (uncalibrated): {logloss_uncalib:.4f}")
    print(f"   Log Loss (calibrated):   {logloss_calib:.4f}")
    print(f"   Improvement: {((logloss_uncalib - logloss_calib) / logloss_uncalib * 100):.2f}%")
    
    # Create calibration plots
    print("\n8. Generating calibration plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before calibration
    prob_true, prob_pred = calibration_curve(y_calib, y_pred_uncalib, n_bins=10)
    axes[0].plot(prob_pred, prob_true, 's-', label='Uncalibrated', linewidth=2, markersize=8)
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('True Probability', fontsize=12)
    axes[0].set_title('Before Calibration', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95, f'Brier Score: {brier_uncalib:.4f}', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # After calibration  
    prob_true, prob_pred = calibration_curve(y_calib, y_pred_calib, n_bins=10)
    axes[1].plot(prob_pred, prob_true, 's-', label='Calibrated', linewidth=2, markersize=8)
    axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('True Probability', fontsize=12)
    axes[1].set_title('After Calibration', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.05, 0.95, f'Brier Score: {brier_calib:.4f}', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = 'models/calibration_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved calibration plot to {plot_path}")
    
    # Save calibrated model
    print("\n9. Saving calibrated model...")
    model_path = 'models/horse_win_predictor_calibrated.pkl'
    joblib.dump(calibrated, model_path)
    print(f"   [OK] Saved calibrated model to {model_path}")
    
    # Save calibration metrics
    print("\n10. Saving calibration metrics...")
    metrics = {
        'calibration_date': pd.Timestamp.now().isoformat(),
        'calibration_data_start': calib_start.isoformat(),
        'calibration_data_end': df['date_dt'].max().isoformat(),
        'n_calibration_samples': len(calib_data),
        'n_training_samples': len(train_data),
        'metrics': {
            'brier_score_uncalibrated': float(brier_uncalib),
            'brier_score_calibrated': float(brier_calib),
            'brier_improvement_pct': float((brier_uncalib - brier_calib) / brier_uncalib * 100),
            'log_loss_uncalibrated': float(logloss_uncalib),
            'log_loss_calibrated': float(logloss_calib),
            'log_loss_improvement_pct': float((logloss_uncalib - logloss_calib) / logloss_uncalib * 100),
        },
        'calibration_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        }
    }
    
    metrics_path = Path('models/calibration_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   [OK] Saved calibration metrics to {metrics_path}")
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print(f"\nTo use the calibrated model in predictions:")
    print(f"  import joblib")
    print(f"  model = joblib.load('{model_path}')")
    print(f"  predictions = model.predict_proba(X)[:, 1]")

if __name__ == '__main__':
    calibrate_and_save()
