from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find all race links
race_links = soup.find_all('a', href=re.compile(r'^/races/20\d{2}-'))
print(f'Found {len(race_links)} race links')

# Show first one's context
if race_links:
    link = race_links[0]
    print('\n--- Link text:', repr(link.get_text(strip=True)))
    print('--- Link href:', link['href'])
    
    # Walk up to find the race card container
    for level in range(1, 6):
        ancestor = link
        for _ in range(level):
            ancestor = ancestor.parent
        if ancestor is None:
            break
        cls = ancestor.get('class', [])
        print(f'\nLevel {level}: <{ancestor.name} class="{" ".join(cls) if cls else ""}">')
        text = ancestor.get_text(separator='|', strip=True)
        print('  Text:', text[:200])
