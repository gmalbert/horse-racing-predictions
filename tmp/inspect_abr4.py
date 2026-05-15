from pathlib import Path
from bs4 import BeautifulSoup

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find all anchor tags
all_links = soup.find_all('a', href=True)
print(f'Total anchors: {len(all_links)}')

# Show all hrefs
for a in all_links[:20]:
    print(repr(a['href'][:80]))

# search for races pattern in raw html
import re
raw_match = re.search(r'races/2026', html)
print('\nRaw match found:', raw_match)
if raw_match:
    print(html[raw_match.start()-100:raw_match.start()+300])
