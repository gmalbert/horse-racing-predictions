import pandas as pd
import json

# Load connections_v2
df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
print(f"Historical data: {len(df):,} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {len(df.columns)}\n")

# Load Feb 14 racecards
with open('data/raw/racecards_2026-02-14.json', 'r') as f:
    racecards_data = json.load(f)

# Extract first race from nested structure {region: {course: {time: race}}}
first_race = None
for region in racecards_data.values():
    for course in region.values():
        for race in course.values():
            first_race = race
            break
        if first_race:
            break
    if first_race:
        break

print(f"First race: {first_race['off_time']} {first_race['course']}")
print(f"Runners: {len(first_race['runners'])}\n")

# Check each horse
for runner in first_race['runners'][:5]:
    horse_name = runner['name']
    matches = df[df['horse'].str.lower() == horse_name.lower()]
    print(f"{horse_name}:")
    print(f"  Historical races: {len(matches)}")
    if len(matches) > 0:
        print(f"  Last race: {matches['date'].max()}")
        print(f"  Career OR: {matches['or'].mean():.1f}")
    else:
        print("  NO HISTORICAL DATA FOUND")
    print()
