#!/usr/bin/env python3
"""
Master script to build all new features for the improved model.
Runs in sequence:
1. Pedigree features
2. Pace features  
3. Recent form features
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and report success/failure."""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    
    script_path = Path('scripts') / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} FAILED")
        print(f"Error: {e}")
        return False

def main():
    """Run all feature engineering scripts."""
    
    print("\n" + "="*70)
    print("MODEL IMPROVEMENT: CRITICAL FEATURES IMPLEMENTATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scripts = [
        ('build_sire_lookup.py', 'Building Sire Lookup Table'),
        ('add_pedigree_features.py', 'Adding Pedigree Features'),
        ('add_pace_features.py', 'Adding Pace/Running Style Features'),
        ('add_recent_form_features.py', 'Adding Recent Form Features'),
    ]
    
    results = []
    
    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠ WARNING: {description} failed. Continuing with next step...")
    
    # Final summary
    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {description}")
    
    # Check final output
    final_output = Path('data/processed/race_scores_with_features.parquet')
    if final_output.exists():
        import pandas as pd
        df = pd.read_parquet(final_output)
        print(f"\n✓ Final dataset created: {final_output}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # List new features
        base_cols = 70  # Approximate original column count
        new_cols = len(df.columns) - base_cols
        print(f"  Estimated new features: ~{new_cols}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if all succeeded
    if all(success for _, success in results):
        print("\n✓ ALL FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("1. Retrain model with new features:")
        print("   python scripts/phase3_build_horse_model.py")
        print("\n2. Validate improvements:")
        print("   python scripts/validate_model.py")
        return 0
    else:
        print("\n⚠ SOME STEPS FAILED - Review errors above")
        return 1

if __name__ == '__main__':
    exit(main())
