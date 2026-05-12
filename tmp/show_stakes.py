import json
from pathlib import Path
data = json.loads(Path('data/raw/abr_stakes_2026.json').read_text())
count = data['count']
print(f'Total: {count}')
print()
for s in data['stakes'][:20]:
    date = s['date'] or '??'
    grade = s['grade'] or '-'
    name = s['race_name'][:45]
    track = s['track']
    print(f'{date} | {grade:4} | {name:45} | {track}')
