import json
from pathlib import Path
from datetime import date

data = json.loads(Path('data/raw/abr_stakes_2026.json').read_text())
stakes = data['stakes']
print(f'Total: {data["count"]}')
print()

today = date(2026, 5, 11)
upcoming = [s for s in stakes if s['date'] and s['date'] >= str(today)]
past = [s for s in stakes if s['date'] and s['date'] < str(today)]
no_date = [s for s in stakes if not s['date']]
print(f'Upcoming (>= today): {len(upcoming)}')
print(f'Past: {len(past)}')
print(f'No date: {len(no_date)}')
print()

print('=== Next 30 upcoming G1/G2 ===')
for s in upcoming[:30]:
    grade = s['grade'] or '-'
    if grade in ('G1', 'G2'):
        print(f'  {s["date"]} | {grade:3} | {s["race_name"][:50]:50} | {s["track"]}')
