import json
from collections import Counter

P = 'data/raw/racecards_2025-12-22.json'

def main():
    with open(P, 'r', encoding='utf-8') as f:
        j = json.load(f)

    race_ids = []
    for region in j:
        for course in j[region]:
            for time, race in j[region][course].items():
                race_ids.append(race.get('race_id'))

    print('total races:', len(race_ids))
    print('unique race_ids:', len(set(race_ids)))
    c = Counter(race_ids)
    dup = [k for k,v in c.items() if v>1]
    print('duplicates (if any):', dup[:10])

    # sample race and runner keys
    for region in j:
        for course in j[region]:
            for time, race in j[region][course].items():
                print('\nSample race keys:', list(race.keys())[:40])
                if race.get('runners'):
                    print('Sample runner keys:', list(race['runners'][0].keys())[:60])
                return

if __name__ == '__main__':
    main()
