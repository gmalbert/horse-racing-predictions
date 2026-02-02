import pandas as pd

df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

print('Pedigree columns available:')
pedigree_cols = [c for c in df.columns if any(x in c.lower() for x in ['sire', 'dam'])]
print(pedigree_cols)

if pedigree_cols:
    print('\nSample pedigree data:')
    print(df[pedigree_cols].head(10))
    
    print('\nCoverage:')
    for col in pedigree_cols:
        non_null = df[col].notna().sum()
        pct = df[col].notna().mean() * 100
        print(f'{col}: {non_null:,} / {len(df):,} ({pct:.1f}%)')
else:
    print('No pedigree columns found!')
