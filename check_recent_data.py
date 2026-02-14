import pandas as pd

df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

print(f"Total records: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")

# Check recent data
recent = df[df['date'] >= '2026-01-20'].sort_values('date', ascending=False)
print(f"Races Jan 20-31, 2026: {len(recent):,}")
print(f"\nSample horses from late January 2026:")
sample = recent[['horse', 'date', 'course']].drop_duplicates('horse').head(15)
for _, row in sample.iterrows():
    print(f"  {row['horse']:30s} - {row['date']} at {row['course']}")

# Check if any Feb 14 racecard horses exist
print(f"\nChecking Feb 14 racecard horses:")
test_horses = ['Etna Bianco', 'Catchintsavo', 'Mondoui boy', 'Sam Brown', 'Threeunderthrufive']
for horse in test_horses:
    count = len(df[df['horse'].str.lower() == horse.lower()])
    print(f"  {horse}: {count} races")
