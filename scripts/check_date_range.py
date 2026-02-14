import pandas as pd

# Load the historical data
df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

print(f"Total records: {len(df):,}")
print(f"\nDate column range: {df['date'].min()} to {df['date'].max()}")

# Convert to datetime and check
df['date_dt'] = pd.to_datetime(df['date'])
print(f"Datetime range: {df['date_dt'].min()} to {df['date_dt'].max()}")

# Show latest 15 dates with race counts
print("\nLatest 15 dates in data:")
date_counts = df.groupby('date').size().sort_index(ascending=False).head(15)
for date, count in date_counts.items():
    print(f"  {date}: {count} races")

# Check if we have any January 2026 data
jan_2026 = df[(df['date_dt'] >= '2026-01-01') & (df['date_dt'] < '2026-02-01')]
print(f"\nJanuary 2026 records: {len(jan_2026)}")
if len(jan_2026) > 0:
    print(f"  Date range: {jan_2026['date_dt'].min()} to {jan_2026['date_dt'].max()}")

# Check if we have any February 2026 data
feb_2026 = df[(df['date_dt'] >= '2026-02-01') & (df['date_dt'] < '2026-03-01')]
print(f"February 2026 records: {len(feb_2026)}")

# Check December 2025
dec_2025 = df[(df['date_dt'] >= '2025-12-01') & (df['date_dt'] < '2026-01-01')]
print(f"December 2025 records: {len(dec_2025)}")
if len(dec_2025) > 0:
    print(f"  Last December 2025 date: {dec_2025['date_dt'].max()}")
