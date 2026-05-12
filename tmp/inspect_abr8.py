from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find Kentucky Derby item
race_items = soup.find_all('div', class_=lambda c: c and 'race' in c and 'expandable' in c)

for item in race_items:
    name_div = item.find('div', class_='name')
    if not name_div:
        continue
    name_link = name_div.find('a')
    if not name_link or 'Kentucky Derby' not in name_link.get_text():
        continue
    
    print('=== Kentucky Derby parent chain ===')
    # Level 2 (row collapse) - check its text
    l2 = item.parent.parent  # row collapse
    print(f'Level 2: <{l2.name} class="{l2.get("class")}">')
    print('  Full text:', l2.get_text(separator='|', strip=True)[:400])
    print()
    
    # Level 3
    l3 = l2.parent
    print(f'Level 3: <{l3.name} class="{l3.get("class")}">')
    # Look for any direct text or spans with date
    for child in l3.children:
        if hasattr(child, 'get_text'):
            t = child.get_text(strip=True)
            if t and len(t) < 50:
                print(f'  Child: <{child.name} class="{child.get("class")}"> text={repr(t)}')
        else:
            if str(child).strip():
                print(f'  NavigableString: {repr(str(child).strip())[:50]}')
    print()
    print('  Level 3 full HTML (first 500):')
    print(str(l3)[:500])
    break
