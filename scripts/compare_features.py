import pandas as pd

# Load predictions
preds = pd.read_csv('data/processed/predictions_2026-02-14.csv')

# Get Astronomic View (the outlier)
astro = preds[preds['horse'] == 'Astronomic View'].iloc[0]

# Get a typical horse with 15.8% prediction
typical = preds[(preds['win_probability'] > 0.157) & (preds['win_probability'] < 0.159)].iloc[0]

print("Comparing features:\n")
print(f"{'Feature':<25s} | {'Astronomic (23.9%)':<20s} | {'Typical (15.8%)':<20s}")
print("-" * 70)

# Compare key features
key_features = ['or_numeric', 'career_runs', 'career_win_rate', 'career_place_rate', 
                'cd_runs', 'cd_win_rate', 'avg_last_3_pos', 'wins_last_3', 'days_since_last',
                'age', 'weight_lbs', 'jockey_career_runs', 'jockey_career_win_rate',
                'field_size', 'class_num', 'is_turf', 'going_numeric']

for feat in key_features:
    if feat in preds.columns:
        astro_val = astro[feat]
        typ_val = typical[feat]
        diff = '*' if abs(astro_val - typ_val) > 0.01 else ''
        print(f"{feat:<25s} | {str(astro_val):<20s} | {str(typ_val):<20s} {diff}")

print(f"\n\nAstronomic View race: {astro['course']} at {astro['race_time']}")
print(f"Typical horse: {typical['horse']} at {typical['course']} {typical['race_time']}")
