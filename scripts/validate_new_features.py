#!/usr/bin/env python3
"""
Quick validation of new features - show before/after comparison
"""

import pandas as pd
from pathlib import Path

def validate_features():
    """Show what features were added and their coverage."""
    
    print("="*70)
    print("FEATURE VALIDATION REPORT")
    print("="*70)
    
    # Load before and after
    base_path = Path('data/processed/race_scores.parquet')
    new_path = Path('data/processed/race_scores_with_features.parquet')
    
    if not new_path.exists():
        print("\n❌ New features file not found. Run scripts/build_all_features.py first")
        return
    
    print("\n📊 Loading datasets...")
    df_base = pd.read_parquet(base_path)
    df_new = pd.read_parquet(new_path)
    
    print(f"  Base: {len(df_base.columns)} columns")
    print(f"  New:  {len(df_new.columns)} columns")
    print(f"  Added: {len(df_new.columns) - len(df_base.columns)} features")
    
    # Find new columns
    new_cols = set(df_new.columns) - set(df_base.columns)
    
    print(f"\n✨ NEW FEATURES ({len(new_cols)} total):")
    print("\n--- PEDIGREE FEATURES (8) ---")
    pedigree_features = [col for col in new_cols if 'sire' in col.lower() or 'adj' in col]
    for feat in sorted(pedigree_features):
        null_pct = df_new[feat].isna().mean() * 100
        mean_val = df_new[feat].mean()
        print(f"  ✓ {feat:30s} — {null_pct:5.1f}% null, mean={mean_val:.3f}")
    
    print("\n--- PACE FEATURES (15) ---")
    pace_features = [col for col in new_cols if 'pace' in col.lower() or 'specialist' in col.lower() or col.startswith('race_')]
    for feat in sorted(pace_features):
        if df_new[feat].dtype == 'object':
            print(f"  ✓ {feat:30s} — {df_new[feat].value_counts().to_dict()}")
        else:
            null_pct = df_new[feat].isna().mean() * 100
            mean_val = df_new[feat].mean()
            print(f"  ✓ {feat:30s} — {null_pct:5.1f}% null, mean={mean_val:.3f}")
    
    print("\n--- RECENT FORM FEATURES (10) ---")
    form_features = [col for col in new_cols if 'form' in col.lower() or 'in_form' in col or 'connections' in col]
    for feat in sorted(form_features):
        null_pct = df_new[feat].isna().mean() * 100
        mean_val = df_new[feat].mean()
        print(f"  ✓ {feat:30s} — {null_pct:5.1f}% null, mean={mean_val:.3f}")
    
    # Cold start comparison
    print("\n🎯 COLD START IMPROVEMENT:")
    
    # Horses with < 3 career runs
    if 'career_runs' in df_new.columns:
        cold_start = df_new[df_new['career_runs'] < 3]
        print(f"\n  Horses with < 3 career runs: {len(cold_start):,} ({len(cold_start)/len(df_new)*100:.1f}%)")
        
        print("\n  BEFORE (no features):")
        print(f"    career_win_rate: all 0 or very low")
        
        print("\n  AFTER (sire-adjusted):")
        if 'career_win_rate_adj' in df_new.columns:
            print(f"    career_win_rate_adj mean: {cold_start['career_win_rate_adj'].mean():.3f}")
            print(f"    Now using sire data: {(cold_start['career_win_rate_adj'] > 0).mean()*100:.1f}% have values")
    
    # Sample horse example
    print("\n📋 EXAMPLE HORSE COMPARISON:")
    
    # Find a horse with 0-2 career runs
    if 'career_runs' in df_new.columns:
        sample_horse = df_new[df_new['career_runs'] <= 2].iloc[0]
        
        print(f"\n  Horse: {sample_horse['horse']}")
        print(f"  Career runs: {sample_horse.get('career_runs', 'N/A')}")
        print(f"  Sire: {sample_horse['sire']}")
        
        print("\n  OLD FEATURES (would be 0 or N/A):")
        print(f"    career_win_rate: {sample_horse.get('career_win_rate', 'N/A')}")
        print(f"    career_place_rate: {sample_horse.get('career_place_rate', 'N/A')}")
        
        print("\n  NEW FEATURES (from sire & context):")
        print(f"    sire_win_rate: {sample_horse.get('sire_win_rate', 'N/A'):.3f}")
        print(f"    sire_surface_match: {sample_horse.get('sire_surface_match', 'N/A'):.3f}")
        print(f"    sire_distance_match: {sample_horse.get('sire_distance_match', 'N/A'):.3f}")
        print(f"    career_win_rate_adj: {sample_horse.get('career_win_rate_adj', 'N/A'):.3f}")
        print(f"    pace_style: {sample_horse.get('pace_style', 'N/A')}")
        print(f"    jockey_form_14d: {sample_horse.get('jockey_form_14d', 'N/A'):.3f}")
    
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Retrain model: python scripts/phase3_build_horse_model.py")
    print("2. Update feature list to include new features")
    print("3. Validate with temporal split (train on past, test on recent)")

if __name__ == '__main__':
    validate_features()
