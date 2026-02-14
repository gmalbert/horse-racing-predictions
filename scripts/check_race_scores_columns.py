"""Check columns in race_scores.parquet"""
import pandas as pd

df = pd.read_parquet('data/processed/race_scores.parquet')
print(f"race_scores.parquet columns ({len(df.columns)}):")
for col in sorted(df.columns):
    print(f"  - {col}")
