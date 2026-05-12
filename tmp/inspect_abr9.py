from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find the "race-list" containers — each should be one date group
race_lists = soup.find_all('div', class_='race-list')
print(f'race-list containers: {len(race_lists)}')

for rl in race_lists[:3]:
    print()
    # Find date span
    date_span = rl.find('span', class_='date')
    print(f'  date: {date_span.get_text(strip=True) if date_span else "NONE"}')
    # Find all race expandable divs within
    races = rl.find_all('div', class_=lambda c: c and 'race' in c and 'expandable' in c)
    print(f'  races: {len(races)}')
    for r in races:
        name_div = r.find('div', class_='name')
        if name_div:
            a = name_div.find('a')
            if a:
                print(f'    - {a.get_text(strip=True)}')
