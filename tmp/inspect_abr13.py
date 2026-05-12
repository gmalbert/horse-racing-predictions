from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find month filter buttons
# Look for elements containing month names
months = ['January', 'February', 'March', 'April', 'June', 'July', 'August', 'September']
print('=== Month filter buttons ===')
for m in months:
    els = soup.find_all(string=re.compile(r'\b' + m + r'\b', re.I))
    if els:
        for el in els[:2]:
            parent = el.parent
            print(f'{m}: <{parent.name} class="{parent.get("class")}"> "{str(el).strip()[:60]}"')

# Also look for data-month attributes or any filter-related elements
print('\n=== Filter-related elements ===')
filter_els = soup.find_all(attrs={'data-month': True})
print(f'data-month elements: {len(filter_els)}')

# Look at the filter/tab area
filter_container = soup.find(class_=re.compile(r'filter|tab|month'))
if filter_container:
    print('\nFilter container found:')
    print(str(filter_container)[:500])
