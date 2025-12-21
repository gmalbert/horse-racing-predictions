import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
PARQUET_FILE = DATA_DIR / 'processed' / 'all_gb_races.parquet'

df = pd.read_parquet(PARQUET_FILE)
print(f"dist_f column type: {df['dist_f'].dtype}")
print(f"First 20 values:")
print(df['dist_f'].head(20).tolist())
print(f"\nLast 20 values:")
print(df['dist_f'].tail(20).tolist())
print(f"\nUnique values (first 50):")
print(df['dist_f'].unique()[:50])
print(f"\nAny non-numeric?")
non_numeric = df[~df['dist_f'].apply(lambda x: isinstance(x, (int, float)))]
print(f"Non-numeric count: {len(non_numeric)}")
if len(non_numeric) > 0:
    print(non_numeric[['date', 'course', 'dist_f']].head())
