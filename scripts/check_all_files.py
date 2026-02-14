import pandas as pd

print("Checking race_scores.parquet:")
df = pd.read_parquet('data/processed/race_scores.parquet')
print(f"  Records: {len(df):,}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

print("\nChecking race_scores_connections_v2.parquet:")
df2 = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
print(f"  Records: {len(df2):,}")
print(f"  Date range: {df2['date'].min()} to {df2['date'].max()}")

print("\nChecking all_gb_races_cleaned.parquet:")
df3 = pd.read_parquet('data/processed/all_gb_races_cleaned.parquet')
print(f"  Records: {len(df3):,}")
print(f"  Date range: {df3['date'].min()} to {df3['date'].max()}")
