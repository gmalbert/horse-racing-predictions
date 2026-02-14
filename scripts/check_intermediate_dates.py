"""Check intermediate files for Jan 2026 data"""
import pandas as pd

files = [
    'race_scores_with_all_features_no_leakage.parquet',
    'race_scores_with_features.parquet',
    'race_scores_enhanced_form.parquet',
]

for filename in files:
    try:
        df = pd.read_parquet(f'data/processed/{filename}')
        df['date_dt'] = pd.to_datetime(df['date'])
        jan_2026 = df[df['date_dt'].between('2026-01-01', '2026-01-31')]
        print(f"{filename}:")
        print(f"  Records: {len(df):,}, Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  January 2026: {len(jan_2026):,} records")
        print()
    except Exception as e:
        print(f"{filename}: Error - {e}\n")
