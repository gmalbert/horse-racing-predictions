from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find ALL div.race.expandable
all_race_items = soup.find_all('div', class_=lambda c: c and 'race' in c and 'expandable' in c)
print(f'Total expandable race divs: {len(all_race_items)}')

# Where are the ones NOT in race-list?
for item in all_race_items:
    name_div = item.find('div', class_='name')
    if not name_div:
        continue
    a = name_div.find('a')
    name = a.get_text(strip=True) if a else '?'

    # Find parent race-list
    rl = item.find_parent('div', class_='race-list')
    # Find parent section with id
    parent_with_id = None
    for anc in item.parents:
        if anc.get('id'):
            parent_with_id = anc
            break

    # What's the grouping container?
    # Walk up to find div with class containing 'month' or 'section'
    month_container = None
    for anc in item.parents:
        cls = anc.get('class', [])
        if any('month' in c or 'section' in c or 'lists' in c for c in cls):
            month_container = anc
            break

    print(f'  {name[:40]:40} | in race-list: {bool(rl)} | month_container: {month_container.name if month_container else None} {(month_container.get("class") or []) if month_container else ""}')
