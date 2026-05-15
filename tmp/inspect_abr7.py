from pathlib import Path
from bs4 import BeautifulSoup
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find Kentucky Derby race item
race_items = soup.find_all(
    'div',
    class_=lambda c: c and 'race' in c and 'expandable' in c
)

print(f'Total race items: {len(race_items)}')
print()

for item in race_items:
    name_div = item.find('div', class_='name')
    if not name_div:
        continue
    name_link = name_div.find('a')
    if not name_link:
        continue
    name = name_link.get_text(strip=True)
    
    if 'Kentucky Derby' in name or 'Preakness' in name or 'Peter Pan' in name:
        print(f'=== {name} ===')
        # Check parent structure
        for level in range(1, 8):
            ancestor = item
            for _ in range(level):
                if ancestor.parent:
                    ancestor = ancestor.parent
                else:
                    break
            cls = ancestor.get('class', [])
            text = ancestor.get_text(separator='|', strip=True)
            dot_dates = re.findall(r'\b\d{1,2}\.\d{2}\b', text)
            print(f'  Level {level}: <{ancestor.name} class="{" ".join(cls) if cls else ""}"> dot-dates: {dot_dates[:3]}')
            if level >= 5:
                break
        print()
