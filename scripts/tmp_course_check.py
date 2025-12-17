import pandas as pd

counts = pd.read_csv('data/processed/bha_2026_course_counts.csv')
top10 = counts.head(10)['Course'].astype(str).str.strip().tolist()
df = pd.read_csv('data/processed/all_gb_races.csv', usecols=['course'], low_memory=False)
uniq = set(df['course'].astype(str).str.strip().unique())
print('Top10:', top10)
print('Intersection:', sorted(set(top10) & uniq))
print('Missing:', [c for c in top10 if c not in uniq])
