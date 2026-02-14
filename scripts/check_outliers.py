import pandas as pd

# Load historical data
hist = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

# Load predictions
preds = pd.read_csv('data/processed/predictions_2026-02-14.csv')

# Find horses with non-standard probabilities
non_standard = preds[preds['win_probability'] > 0.17]
print(f"Horses with win probability > 17%: {len(non_standard)}")
print("\nThose horses:")
for _, row in non_standard.head(10).iterrows():
    horse_name = row['horse']
    win_prob = row['win_probability']
    
    # Check if in historical data
    matches = hist[hist['horse'].str.lower() == horse_name.lower()]
    has_history = len(matches) > 0
    
    print(f"{horse_name:30s} | Win: {win_prob:.1%} | Historical data: {'YES (' + str(len(matches)) + ' races)' if has_history else 'NO'}")
