"""Check if connections_v2 was regenerated with Jan 2026 data"""
import pandas as pd

df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
print(f"Records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check January 2026
df['date_dt'] = pd.to_datetime(df['date'])
jan_2026 = df[df['date_dt'].between('2026-01-01', '2026-01-31')]
print(f"\nJanuary 2026 records: {len(jan_2026):,}")
if len(jan_2026) > 0:
    print(f"Latest dates: {sorted(jan_2026['date'].unique())[-5:]}")
