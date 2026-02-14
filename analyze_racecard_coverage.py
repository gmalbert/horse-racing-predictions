import pandas as pd
import json

# Load historical data
df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')
print(f"Historical dataset: {len(df):,} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique horses: {df['horse'].nunique():,}\n")

# Load Feb 14 racecards
with open('data/raw/racecards_2026-02-14.json', 'r') as f:
    racecards_data = json.load(f)

# Extract all horses from all races
all_horses = []
race_count = 0
for region in racecards_data.values():
    for course in region.values():
        for race in course.values():
            race_count += 1
            for runner in race['runners']:
                all_horses.append({
                    'name': runner['name'],
                    'course': race['course'],
                    'time': race['off_time'],
                    'jockey': runner.get('jockey', 'Unknown'),
                    'trainer': runner.get('trainer', 'Unknown'),
                    'ofr': runner.get('ofr', None)
                })

print(f"Feb 14, 2026 racecards:")
print(f"  Total races: {race_count}")
print(f"  Total runners: {len(all_horses)}")
print(f"  Unique horses: {len(set(h['name'] for h in all_horses))}\n")

# Check match rate
horses_with_data = 0
horses_without_data = []
horses_with_data_list = []

# Create lowercase horse lookup for faster matching
df_horse_lower = df['horse'].fillna('').str.lower()

for horse_info in all_horses:
    horse_name = horse_info['name']
    # Try exact match first
    matches = df[df_horse_lower == horse_name.lower()]
    
    if len(matches) > 0:
        horses_with_data += 1
        horses_with_data_list.append({
            'name': horse_name,
            'historical_races': len(matches),
            'last_race': matches['date'].max(),
            'avg_or': matches['or'].mean()
        })
    else:
        horses_without_data.append(horse_info)

print(f"MATCH STATISTICS:")
print(f"  Horses WITH historical data: {horses_with_data} ({horses_with_data/len(all_horses)*100:.1f}%)")
print(f"  Horses WITHOUT historical data: {len(horses_without_data)} ({len(horses_without_data)/len(all_horses)*100:.1f}%)")

if horses_with_data > 0:
    print(f"\n✓ HORSES WITH DATA (sample of 20):")
    for h in horses_with_data_list[:20]:
        print(f"  {h['name']:30s} - {h['historical_races']:4d} races, last: {h['last_race']}, avg OR: {h['avg_or']:.1f}")

if len(horses_without_data) > 0:
    print(f"\n✗ HORSES WITHOUT DATA (first 20):")
    for h in horses_without_data[:20]:
        print(f"  {h['name']:30s} - {h['course']:15s} {h['time']} (OFR: {h['ofr']})")

# Check for partial name matches (maybe formatting issue?)
print(f"\n\nCHECKING FOR PARTIAL MATCHES (first 5 missing horses):")
for horse_info in horses_without_data[:5]:
    horse_name = horse_info['name']
    # Try partial match
    partial = df[df['horse'].str.contains(horse_name.split()[0], case=False, na=False)]
    if len(partial) > 0:
        print(f"\n{horse_name}:")
        print(f"  Possible matches: {partial['horse'].unique()[:5].tolist()}")
