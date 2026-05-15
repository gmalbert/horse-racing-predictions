from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find all race containers
race_divs = soup.find_all('div', class_=re.compile(r'\brace\b'))
print(f'race divs: {len(race_divs)}')

# Find specifically "race align-middle expandable" divs
race_items = [d for d in race_divs if 'expandable' in (d.get('class') or [])]
print(f'race expandable divs: {len(race_items)}')

if race_items:
    item = race_items[0]
    print('\n=== RACE ITEM HTML ===')
    print(str(item)[:1000])
    
    print('\n=== LOOKING FOR NAME ===')
    # Look for name element
    name_el = item.find('a', href=re.compile(r'/races/20\d{2}-'))
    if name_el:
        print('Name from link text:', repr(name_el.get_text(strip=True)))
        print('Href:', name_el['href'])
    
    # Look for track link
    track_el = item.find('a', href=re.compile(r'/tracks/'))
    if track_el:
        print('Track:', repr(track_el.get_text(strip=True)))
    
    # Get date from parent "row collapse"
    row = item.find_parent('div', class_=re.compile(r'\bcollapse\b'))
    if row:
        row_text = row.get_text(separator=' ', strip=True)
        print('Row text (first 200):', row_text[:200])
        # Extract date pattern "M.DD"  
        m = re.search(r'\b(\d{1,2}\.\d{2})\b', row_text)
        if m:
            print('Date found:', m.group(1))
