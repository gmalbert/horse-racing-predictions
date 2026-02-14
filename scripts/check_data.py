import pandas as pd

df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
print(f'Total records: {len(df):,}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'Unique horses: {df["horse"].nunique():,}')
print('\nSample horses (first 20):')
for horse in df['horse'].head(20).tolist():
    print(f'  "{horse}"')

# Check if any horses from today's racecards are in the data
test_horses = ['Etna Bianco', 'Catchintsavo', 'Mondouiboy', 'Starzand']
print('\n\nChecking for today\'s horses:')
for horse in test_horses:
    matches = df[df['horse'].str.lower() == horse.lower()]
    print(f'  {horse}: {len(matches)} records')
