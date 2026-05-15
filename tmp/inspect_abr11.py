from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

race_lists = soup.find_all('div', class_='race-list')
print(f'race-list containers: {len(race_lists)}')
print()

for i, rl in enumerate(race_lists):
    date_span = rl.find('span', class_='date')
    date = date_span.get_text(strip=True) if date_span else 'NONE'
    races = rl.find_all('div', class_=lambda c: c and 'race' in c and 'expandable' in c)
    # Find header section month
    section_header = rl.find_previous(['h2', 'h3', 'h4', 'div'],
                                       class_=lambda c: c and ('header' in c or 'month' in c or 'title' in c))
    header_text = section_header.get_text(strip=True)[:60] if section_header else 'none'
    
    # Find what comes right before this race-list in the DOM
    prev_sib = rl.find_previous_sibling()
    prev_text = ''
    if prev_sib:
        prev_text = prev_sib.get_text(separator=' ', strip=True)[:80]
    
    print(f'=== race-list #{i+1} (date: {date}, races: {len(races)}) ===')
    print(f'  header: {header_text}')
    print(f'  prev_sib text: {prev_text}')
    # Show all races
    for r in races[:5]:
        name_div = r.find('div', class_='name')
        if name_div:
            a = name_div.find('a')
            if a:
                print(f'    - {a.get_text(strip=True)[:50]}')
    print()
    
    # Also look for date in parent structure
    parent = rl.parent
    parent_date = parent.find('span', class_='date') if parent else None
    print(f'  parent date span: {parent_date.get_text(strip=True) if parent_date else "none"}')
    print(f'  parent classes: {parent.get("class") if parent else "none"}')
    print()
