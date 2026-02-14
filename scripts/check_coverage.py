import pandas as pd
import json

# Load today's racecards
with open('data/raw/racecards_2026-02-14.json', 'r') as f:
    data = json.load(f)

# Extract all horses from racecards
all_horses = []
if isinstance(data, dict):
    for region in data.values():
        for course in region.values():
            for race in course.values():
                for runner in race.get('runners', []):
                    horse_name = runner.get('horse') or runner.get('name', 'Unknown')
                    all_horses.append(horse_name)

print(f'Total horses in today\'s racecards: {len(all_horses)}')
print(f'Unique horses: {len(set(all_horses))}')

# Load historical data
df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
print(f'\nHistorical data date range: {df["date"].min()} to {df["date"].max()}')

# Check how many horses have historical data
horses_with_history = 0
horses_without_history = 0

for horse in set(all_horses):
    matches = df[df['horse'].str.lower() == horse.lower()]
    if len(matches) > 0:
        horses_with_history += 1
    else:
        horses_without_history += 1

print(f'\nHorses WITH historical data: {horses_with_history}')
print(f'Horses WITHOUT historical data: {horses_without_history}')
print(f'Percentage without data: {horses_without_history / len(set(all_horses)) * 100:.1f}%')

# Show some horses that DO have history
print('\n\nSample horses WITH history:')
count = 0
for horse in set(all_horses):
    matches = df[df['horse'].str.lower() == horse.lower()]
    if len(matches) > 0:
        print(f'  {horse}: {len(matches)} historical races')
        count += 1
        if count >= 10:
            break
