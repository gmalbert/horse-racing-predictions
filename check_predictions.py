import pandas as pd

df = pd.read_csv('data/processed/predictions_2025-12-29.csv')
print(f'Total horses: {len(df)}')
print(f'Races: {df["course"].nunique()} courses')
print(f'Columns: {list(df.columns[:10])}')
print(f'\nFirst 3 horses:')
print(df.head(3)[['race_time', 'course', 'horse', 'win_probability', 'win_odds_fractional']])
