import pandas as pd
from pathlib import Path
p = Path('data/processed/predictions_2026-04-15.csv')
if not p.exists():
    raise FileNotFoundError(p)
df = pd.read_csv(p)
missing = df[df['market_odds'].isna()]
print(f'missing count: {len(missing)}')
cols = ['race_date', 'course', 'race_time', 'horse_name', 'win_probability', 'odds_decimal']
print(missing[cols].to_string(index=False))
