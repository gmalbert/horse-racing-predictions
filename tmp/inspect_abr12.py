from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find the 4th race-list (monthly May calendar)
race_lists = soup.find_all('div', class_='race-list')
monthly_rl = race_lists[3]  # the one with 28 races

# Get all direct row collapse children
row_collapses = monthly_rl.find_all('div', class_='row', recursive=False)
# Actually let's look at direct children structure
print(f'race-list #4 direct children:')
for child in monthly_rl.children:
    if not hasattr(child, 'get'):
        continue
    cls = child.get('class', [])
    text = child.get_text(separator='|', strip=True)[:100]
    date_spans = child.find_all('span', class_='date') if hasattr(child, 'find_all') else []
    races = child.find_all('div', class_=lambda c: c and 'race' in c and 'expandable' in c) if hasattr(child, 'find_all') else []
    print(f'  <{child.name} class="{cls}"> date_spans={[s.get_text() for s in date_spans]} races={len(races)} text={text[:60]}')

# Now walk the Kentucky Derby's actual path
print('\n\n=== Kentucky Derby parent chain ===')
for item in monthly_rl.find_all('div', class_=lambda c: c and 'race' in c and 'expandable' in c)[:3]:
    name_div = item.find('div', class_='name')
    if not name_div:
        continue
    a = name_div.find('a')
    name = a.get_text(strip=True) if a else '?'
    
    # Walk up from race item
    p1 = item.parent  # loader div
    p2 = p1.parent    # row collapse
    p3 = p2.parent    # group container?
    
    p2_dates = p2.find_all('span', class_='date')
    p3_dates = p3.find_all('span', class_='date') if p3 else []
    
    print(f'\n{name}')
    print(f'  p1 classes: {p1.get("class")}')
    print(f'  p2 classes: {p2.get("class")} | span.date: {[s.get_text() for s in p2_dates[:3]]}')
    print(f'  p3 classes: {p3.get("class") if p3 else None} | span.date: {[s.get_text() for s in p3_dates[:3]]}')
    # Check siblings of p2
    prev_sibs_with_date = []
    for sib in p2.previous_siblings:
        if not hasattr(sib, 'find'):
            continue
        date_s = sib.find('span', class_='date')
        if date_s:
            prev_sibs_with_date.append(date_s.get_text(strip=True))
            break
    print(f'  prev_sib date: {prev_sibs_with_date}')
