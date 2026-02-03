#!/usr/bin/env python3
"""scripts/diagnose_model.py - Understand where model fails."""

import pandas as pd
import numpy as np
from pathlib import Path

def load_predictions_with_results():
    """Load predictions and match with actual results."""
    predictions_dir = Path('data/processed')
    
    # Load all prediction files
    pred_files = list(predictions_dir.glob('predictions_*.csv'))
    all_preds = []
    
    for f in pred_files:
        df = pd.read_csv(f)
        df['pred_date'] = f.stem.replace('predictions_', '')
        all_preds.append(df)
    
    predictions = pd.concat(all_preds, ignore_index=True)
    
    # Load historical results
    results = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Match predictions to results
    # This requires fetching results for prediction dates
    return predictions, results

def analyze_failures(predictions, results):
    """Analyze where model predictions fail."""
    
    # Group by race
    race_summary = predictions.groupby(['pred_date', 'course', 'race_time']).apply(
        lambda g: pd.Series({
            'field_size': len(g),
            'top_pick': g.loc[g['win_probability'].idxmax(), 'horse'],
            'top_prob': g['win_probability'].max(),
            'prob_spread': g['win_probability'].std(),
        })
    ).reset_index()
    
    # Analyze characteristics of failures
    print("\n=== MODEL DIAGNOSIS ===\n")
    
    # 1. Probability distribution
    print("1. Win Probability Distribution of Top Picks:")
    print(predictions.groupby(['pred_date', 'course', 'race_time'])['win_probability']
          .max().describe())
    
    # 2. Cold start horses (no career data)
    if 'career_runs' in predictions.columns:
        cold_start = predictions[predictions['career_runs'] == 0]
        print(f"\n2. Cold Start Horses: {len(cold_start)} / {len(predictions)} ({len(cold_start)/len(predictions)*100:.1f}%)")
    else:
        print("\n2. Cold Start Horses: 'career_runs' column not found in predictions")
    
    # 3. Feature availability
    print("\n3. Feature Availability:")
    feature_cols = ['career_win_rate', 'cd_win_rate', 'avg_last_3_pos', 'or_numeric']
    for col in feature_cols:
        if col in predictions.columns:
            null_pct = predictions[col].isna().mean() * 100
            zero_pct = (predictions[col] == 0).mean() * 100
            print(f"   {col}: {null_pct:.1f}% null, {zero_pct:.1f}% zero")
        else:
            print(f"   {col}: NOT IN PREDICTIONS")
    
    # 4. Class distribution
    if 'race_class' in predictions.columns:
        print("\n4. Predictions by Race Class:")
        print(predictions['race_class'].value_counts())
    else:
        print("\n4. Predictions by Race Class: 'race_class' column not found")
    
    return race_summary

def check_feature_values():
    """Quick check of problematic features in historical data."""
    print("\n=== FEATURE VALUE CHECK ===\n")
    
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Features showing 0 importance
    problem_features = ['has_blinkers', 'has_visor', 'first_time_blinkers', 
                       'gear_changed', 'is_maiden', 'is_handicap']
    
    for feat in problem_features:
        if feat in df.columns:
            print(f"{feat}: {df[feat].value_counts().to_dict()}")
        else:
            print(f"{feat}: NOT IN DATAFRAME")

if __name__ == '__main__':
    print("Loading predictions and results...")
    preds, results = load_predictions_with_results()
    
    print(f"\nLoaded {len(preds):,} predictions from {preds['pred_date'].nunique()} dates")
    print(f"Loaded {len(results):,} historical results")
    
    summary = analyze_failures(preds, results)
    
    print("\n" + "="*60)
    check_feature_values()
    
    print("\n" + "="*60)
    print("\nDiagnosis complete!")
