from pathlib import Path
from bs4 import BeautifulSoup

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find the month select
selects = soup.find_all('select')
for sel in selects:
    options = sel.find_all('option')
    print(f'<select name="{sel.get("name", "")}" class="{sel.get("class", "")}">')
    for opt in options:
        print(f'  <option value="{opt.get("value", "")}">{opt.get_text(strip=True)}</option>')
    print()

# Also look for anchor links with month names (mobile version)
month_links = soup.find_all('a', class_=lambda c: c and 'month' in str(c).lower())
print(f'month anchor links: {len(month_links)}')
for a in month_links[:5]:
    print(f'  {a}')

# Look at all anchor tags near the month select
month_filter_div = soup.find('li', class_='month-filter')
if month_filter_div:
    print('\nmonth-filter li HTML:')
    print(str(month_filter_div)[:600])
