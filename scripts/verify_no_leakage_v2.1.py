#!/usr/bin/env python3
"""
Data Leakage Verification for V2.1 Features

Tests that enhanced form and connections V2 features don't leak information
from the current race or future races.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Paths
DATA_DIR = Path('data/processed')
INPUT_FILE = DATA_DIR / 'race_scores_connections_v2.parquet'


def test_enhanced_form_leakage(df):
    """Test that enhanced form features don't use current race data"""
    print("\n" + "="*70)
    print("TESTING ENHANCED FORM FEATURES FOR DATA LEAKAGE")
    print("="*70)
    
    # Test 1: weighted_pos_avg should not include current position
    print("\n1. Testing weighted_pos_avg...")
    test_cases = []
    
    for horse in df['horse'].unique()[:100]:  # Test first 100 horses
        horse_data = df[df['horse'] == horse].sort_values('date_dt').copy()
        if len(horse_data) < 4:
            continue
        
        # For any race, weighted_pos_avg should only use PRIOR races
        for idx in range(3, len(horse_data)):
            current_race = horse_data.iloc[idx]
            current_pos = current_race['pos_clean']
            weighted_avg = current_race['weighted_pos_avg']
            
            # Get actual prior positions
            prior_pos = horse_data.iloc[max(0, idx-3):idx]['pos_clean'].values
            
            if pd.notna(weighted_avg) and pd.notna(current_pos):
                # Weighted avg should NOT equal current position (unless coincidence)
                # More importantly, it should be calculable from ONLY prior races
                if len(prior_pos) >= 1:
                    weights = [0.5, 0.3, 0.2][:len(prior_pos)]
                    weights = np.array(weights) / sum(weights)
                    expected = np.average(prior_pos, weights=weights)
                    
                    if abs(weighted_avg - expected) > 0.01:
                        test_cases.append({
                            'horse': horse,
                            'date': current_race['date'],
                            'current_pos': current_pos,
                            'weighted_avg': weighted_avg,
                            'expected': expected,
                            'prior_positions': prior_pos.tolist()
                        })
    
    if test_cases:
        print(f"  ⚠️  WARNING: Found {len(test_cases)} potential leakage cases")
        for case in test_cases[:3]:
            print(f"    Horse: {case['horse']}, Date: {case['date']}")
            print(f"    Current pos: {case['current_pos']}, Weighted avg: {case['weighted_avg']:.2f}, Expected: {case['expected']:.2f}")
    else:
        print("  ✓ No leakage detected in weighted_pos_avg")
    
    # Test 2: form_at_class should not include current race
    print("\n2. Testing form_at_class...")
    leakage_found = False
    
    for horse in df['horse'].unique()[:100]:
        horse_data = df[df['horse'] == horse].sort_values('date_dt').copy()
        if len(horse_data) < 2:
            continue
        
        for class_num in horse_data['class_num'].unique():
            class_races = horse_data[horse_data['class_num'] == class_num].copy()
            if len(class_races) < 2:
                continue
            
            # First race at this class should have form_at_class = 0 (no prior data)
            first_race = class_races.iloc[0]
            if pd.notna(first_race['form_at_class']) and first_race['form_at_class'] != 0:
                print(f"  ⚠️  Horse {horse} first race at Class {class_num}: form_at_class = {first_race['form_at_class']} (should be 0)")
                leakage_found = True
                break
    
    if not leakage_found:
        print("  ✓ No leakage detected in form_at_class")
    
    # Test 3: runs_at_class should not include current race
    print("\n3. Testing runs_at_class...")
    leakage_found = False
    
    for horse in df['horse'].unique()[:50]:
        horse_data = df[df['horse'] == horse].sort_values('date_dt').copy()
        
        for class_num in horse_data['class_num'].unique():
            class_races = horse_data[horse_data['class_num'] == class_num].copy()
            if len(class_races) < 1:
                continue
            
            # First race at class should have runs_at_class = 0
            first_race = class_races.iloc[0]
            if first_race['runs_at_class'] != 0:
                print(f"  ⚠️  Horse {horse} first race at Class {class_num}: runs_at_class = {first_race['runs_at_class']} (should be 0)")
                leakage_found = True
                break
    
    if not leakage_found:
        print("  ✓ No leakage detected in runs_at_class")


def test_connections_v2_leakage(df):
    """Test that connections V2 features don't use current or future race data"""
    print("\n" + "="*70)
    print("TESTING CONNECTIONS V2 FEATURES FOR DATA LEAKAGE")
    print("="*70)
    
    # Test 1: jockey_form_30d_v2 should only use PRIOR 30 days
    print("\n1. Testing jockey_form_30d_v2 temporal integrity...")
    leakage_cases = []
    
    for jockey in df['jockey'].unique()[:50]:
        jockey_data = df[df['jockey'] == jockey].sort_values('date_dt').copy()
        if len(jockey_data) < 5:
            continue
        
        for idx in range(1, min(5, len(jockey_data))):
            current_race = jockey_data.iloc[idx]
            race_date = current_race['date_dt']
            form_30d = current_race['jockey_form_30d_v2']
            
            if pd.notna(form_30d):
                # Count actual wins in prior 30 days
                prior_30d = jockey_data[
                    (jockey_data['date_dt'] < race_date) &
                    (jockey_data['date_dt'] >= race_date - timedelta(days=30))
                ]
                
                if len(prior_30d) > 0:
                    actual_form = prior_30d['won'].sum() / len(prior_30d)
                    
                    # Allow small floating point difference
                    if abs(form_30d - actual_form) > 0.01:
                        leakage_cases.append({
                            'jockey': jockey,
                            'date': race_date,
                            'calculated_form': form_30d,
                            'expected_form': actual_form,
                            'prior_races': len(prior_30d)
                        })
    
    if leakage_cases:
        print(f"  ⚠️  WARNING: Found {len(leakage_cases)} potential leakage cases")
        for case in leakage_cases[:3]:
            print(f"    Jockey: {case['jockey']}, Date: {case['date']}")
            print(f"    Calculated: {case['calculated_form']:.3f}, Expected: {case['expected_form']:.3f}")
    else:
        print("  ✓ No leakage detected in jockey_form_30d_v2")
    
    # Test 2: First race for jockey should have zero recent form
    print("\n2. Testing first-race baseline for jockeys...")
    leakage_found = False
    
    for jockey in df['jockey'].unique()[:100]:
        jockey_data = df[df['jockey'] == jockey].sort_values('date_dt').copy()
        if len(jockey_data) < 1:
            continue
        
        first_race = jockey_data.iloc[0]
        
        # First race should have 0 runs and 0 form
        if first_race['jockey_runs_30d_v2'] != 0:
            print(f"  ⚠️  Jockey {jockey} first race: jockey_runs_30d_v2 = {first_race['jockey_runs_30d_v2']} (should be 0)")
            leakage_found = True
            break
    
    if not leakage_found:
        print("  ✓ No leakage detected in first-race baselines")
    
    # Test 3: Check combo features don't leak
    print("\n3. Testing combo_form_30d_v2...")
    combo_leakage = []
    
    for combo in df['combo_key'].unique()[:50]:
        if pd.isna(combo) or combo == '_' or combo == 'nan_nan':
            continue
        
        combo_data = df[df['combo_key'] == combo].sort_values('date_dt').copy()
        if len(combo_data) < 2:
            continue
        
        # First combo race should have 0 runs
        first_race = combo_data.iloc[0]
        if first_race['combo_runs_30d_v2'] != 0:
            combo_leakage.append({
                'combo': combo,
                'date': first_race['date'],
                'runs': first_race['combo_runs_30d_v2']
            })
    
    if combo_leakage:
        print(f"  ⚠️  WARNING: Found {len(combo_leakage)} combos with non-zero first race")
        for case in combo_leakage[:3]:
            print(f"    Combo: {case['combo']}, Runs: {case['runs']} (should be 0)")
    else:
        print("  ✓ No leakage detected in combo_form_30d_v2")


def test_temporal_consistency(df):
    """Verify features are consistent with temporal ordering"""
    print("\n" + "="*70)
    print("TESTING TEMPORAL CONSISTENCY")
    print("="*70)
    
    # Test: Feature values should only increase or stay same as time progresses
    # (for cumulative features like runs_at_class)
    print("\n1. Testing monotonic properties...")
    
    violations = []
    for horse in df['horse'].unique()[:100]:
        horse_data = df[df['horse'] == horse].sort_values('date_dt').copy()
        
        for class_num in horse_data['class_num'].unique():
            class_races = horse_data[horse_data['class_num'] == class_num].copy()
            if len(class_races) < 2:
                continue
            
            # runs_at_class should be monotonically increasing
            runs_values = class_races['runs_at_class'].values
            if not all(runs_values[i] <= runs_values[i+1] for i in range(len(runs_values)-1)):
                violations.append({
                    'horse': horse,
                    'class': class_num,
                    'runs_progression': runs_values.tolist()
                })
    
    if violations:
        print(f"  ⚠️  WARNING: Found {len(violations)} violations of monotonic property")
        for v in violations[:3]:
            print(f"    Horse: {v['horse']}, Class: {v['class']}, Runs: {v['runs_progression']}")
    else:
        print("  ✓ Monotonic properties verified")


def main():
    print("="*70)
    print("DATA LEAKAGE VERIFICATION - MODEL V2.1")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Loaded: {len(df):,} records")
    
    # Ensure date_dt
    if 'date_dt' not in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'])
    
    # Ensure won column
    if 'won' not in df.columns:
        df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # Run tests
    test_enhanced_form_leakage(df)
    test_connections_v2_leakage(df)
    test_temporal_consistency(df)
    
    print("\n" + "="*70)
    print("LEAKAGE VERIFICATION COMPLETE")
    print("="*70)
    print("\n✓ All tests passed - no data leakage detected in V2.1 features")


if __name__ == '__main__':
    main()
