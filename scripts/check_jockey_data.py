import pandas as pd

hist = pd.read_parquet('data/processed/race_scores_with_betting_tiers.parquet')
today = pd.read_csv('data/processed/predictions_2025-12-29.csv')

print("Checking jockey data integration...\n")

# Check Harry Cobden
cobden = hist[hist['jockey'].str.contains('Cobden', case=False, na=False)]
print(f"Harry Cobden historical rides: {len(cobden)}")
if len(cobden) > 0:
    wins = (cobden['pos'] == 1).sum()
    print(f"  Wins: {wins}")
    print(f"  Win rate: {wins / len(cobden):.2%}")

# Check today's predictions for Harry Cobden
cobden_today = today[today['jockey'].str.contains('Cobden', case=False, na=False)]
if len(cobden_today) > 0:
    print(f"\nHarry Cobden in today's predictions: {len(cobden_today)} rides")
    print(cobden_today[['horse', 'jockey', 'jockey_career_runs', 'jockey_career_win_rate']].head())

# Check overlap
overlap = set(today['jockey'].unique()) & set(hist['jockey'].unique())
print(f"\n\nJockey name matching:")
print(f"  Today's jockeys: {len(today['jockey'].unique())}")
print(f"  Historical jockeys: {len(hist['jockey'].unique())}")
print(f"  Exact matches: {len(overlap)}")

if len(overlap) > 0:
    print(f"\nSample matching jockeys: {list(overlap)[:10]}")
