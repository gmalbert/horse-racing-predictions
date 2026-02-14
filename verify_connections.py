import pandas as pd

df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

print(f'Records: {len(df):,}')
print(f'Columns: {len(df.columns)}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'\nFeature groups:')
jockey_cols = [c for c in df.columns if 'jockey_' in c and '_v2' in c]
trainer_cols = [c for c in df.columns if 'trainer_' in c and '_v2' in c]
combo_cols = [c for c in df.columns if 'combo_' in c and '_v2' in c]
print(f'  Jockey V2 features: {len(jockey_cols)}')
print(f'  Trainer V2 features: {len(trainer_cols)}')
print(f'  Combo V2 features: {len(combo_cols)}')
print(f'\nSample jockey features: {jockey_cols[:5]}')
print(f'Sample trainer features: {trainer_cols[:5]}')
print(f'Sample combo features: {combo_cols[:3]}')
