import pandas as pd
import json
import re

# Load historical data
df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

# Create lookup without country suffixes
def strip_country(name):
    """Remove country suffixes like (IRE), (GB), (FR), (USA)"""
    return re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', str(name)).strip()

df['horse_no_country'] = df['horse'].apply(strip_country).str.lower()

print(f"Historical dataset: {len(df):,} records")
print(f"Unique horses (with country): {df['horse'].nunique():,}")
print(f"Unique horses (without country): {df['horse_no_country'].nunique():,}\n")

# Load Feb 14 racecards
with open('data/raw/racecards_2026-02-14.json', 'r') as f:
    racecards_data = json.load(f)

# Extract all horses
all_horses = []
for region in racecards_data.values():
    for course in region.values   ():
        for race in course.values():
            for runner in race['runners']:
                all_horses.append({
                    'name': runner['name'],
                    'name_clean': strip_country(runner['name']).lower(),
                    'course': race['course'],
                    'ofr': runner.get('ofr', None)
                })

print(f"Feb 14 racecards: {len(all_horses)} runners\n")

# Check matches WITH country suffix stripping
matches_found = 0
matches_list = []
no_matches = []

for horse in all_horses:
    matching_records = df[df['horse_no_country'] == horse['name_clean']]
    
    if len(matching_records) > 0:
        matches_found += 1
        # Convert 'or' to numeric, handling non-numeric values
        try:
            avg_or = pd.to_numeric(matching_records['or'], errors='coerce').mean()
        except:
            avg_or = 0
        matches_list.append({
            'racecard_name': horse['name'],
            'historical_name': matching_records.iloc[0]['horse'],
            'races': len(matching_records),
            'last_race': matching_records['date'].max(),
            'avg_or': avg_or
        })
    else:
        no_matches.append(horse)

print(f"═══════════════════════════════════════════════════════════")
print(f"MATCH STATISTICS (with country suffix handling):")
print(f"═══════════════════════════════════════════════════════════")
print(f"  ✓ Horses WITH historical data: {matches_found} / {len(all_horses)} ({matches_found/len(all_horses)*100:.1f}%)")
print(f"  ✗ Horses WITHOUT historical data: {len(no_matches)} / {len(all_horses)} ({len(no_matches)/len(all_horses)*100:.1f}%)\n")

if matches_found > 0:
    print(f"✓ SAMPLE MATCHES (first 30):")
    for m in matches_list[:30]:
        print(f"  '{m['racecard_name']:28s}' → '{m['historical_name']:35s}' ({m['races']:4d} races, last: {str(m['last_race'])[:10]}, OR: {m['avg_or']:.0f})")

if len(no_matches) > 0:
    print(f"\n✗ HORSES STILL WITHOUT DATA (first 20):")
    for h in no_matches[:20]:
        print(f"  {h['name']:30s} - {h['course']:15s} (OFR: {h['ofr']})")

print(f"\n{'═'*60}")
print(f"DIAGNOSIS: {'NAME FORMATTING MISMATCH' if matches_found > 500 else 'GENUINE MISSING DATA'}")
print(f"{'═'*60}")
if matches_found > 500:
    print("The prediction script needs to strip country suffixes from horse names")
    print("before matching against historical data!")
