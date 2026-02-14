"""Verify predictions data for UI compatibility"""
import pandas as pd
import sys

try:
    # Load predictions
    df = pd.read_csv('data/processed/predictions_2026-02-14.csv')
    print(f"✅ Loaded {len(df)} predictions")
    
    # Check required columns
    required_cols = ['horse', 'jockey', 'race_class', 'distance_f', 'ofr', 'race_name', 
                     'date', 'race_time', 'course', 'win_probability', 'place_probability', 
                     'show_probability']
    
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)
    else:
        print(f"✅ All required columns present")
    
    # Test groupby operation that UI uses
    print("\nTesting UI groupby operation...")
    races = df.groupby(['date', 'race_time', 'course', 'race_name'], observed=False, dropna=False).size().reset_index()
    print(f"✅ Found {len(races)} races")
    
    if len(races) > 0:
        print(f"\nSample race:")
        print(races.iloc[0][['date', 'race_time', 'course', 'race_name']])
    else:
        print("⚠️  No races found - this could cause UI issues")
    
    # Test race_name handling
    print(f"\nRace name stats:")
    print(f"  Null/NaN: {df['race_name'].isna().sum()}")
    print(f"  Empty strings: {(df['race_name'] == '').sum()}")
    
    print("\n✅ All verification checks passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
