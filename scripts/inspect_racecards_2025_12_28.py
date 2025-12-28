#!/usr/bin/env python
import json
from pathlib import Path
p=Path('data/raw/racecards_2025-12-28.json')
if not p.exists():
    print('MISSING')
    raise SystemExit(1)

d=json.load(p.open(encoding='utf-8'))
race_ids=[]
odds_found=0
sample_odds_keys=set()
for region in d.values():
    for course in region.values():
        for off_time,r in course.items():
            race_ids.append(r.get('race_id'))
            for runner in r.get('runners',[]):
                for k in ['bookmaker_odds','odds','price','bookmakers','best_price','best_odds']:
                    if k in runner and runner.get(k):
                        odds_found+=1
                        sample_odds_keys.add(k)
                if 'bookmakers' in runner and isinstance(runner['bookmakers'],(list,dict)):
                    odds_found+=1
                    sample_odds_keys.add('bookmakers')

print('RACES_TOTAL:', len(race_ids))
print('UNIQUE_RACE_IDS:', len(set(race_ids)))
print('ODDS_ENTRIES_FOUND:', odds_found)
print('SAMPLE_ODDS_KEYS:', list(sample_odds_keys)[:10])
