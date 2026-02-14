import pandas as pd

# Load predictions
df = pd.read_csv('data/processed/predictions_2026-02-14.csv')

# Filter to first race at Ascot
race1 = df[(df['course'] == 'Ascot') & (df['race_time'] == '8:15 AM ET')].copy()

print("First race at Ascot (8:15 AM ET)")
print(f"Horses: {len(race1)}")
print("\nHorse-specific features that SHOULD vary:")
print("\nHorse        | OR  | Win% | Place% | Career Runs | Avg Pos | Days Last | Win Prob")
print("-" * 90)
for _, row in race1.iterrows():
    print(f"{row['horse']:15s} | {row['or_numeric']:3.0f} | {row['career_win_rate']:4.2f} | {row['career_place_rate']:4.2f} | {row['career_runs']:3.0f} | {row['avg_last_3_pos']:5.1f} | {row['days_since_last']:4.0f} | {row['win_probability']:.3f}")

print("\n\nFeature columns in CSV:")
print(df.columns.tolist()[:30])
