from pathlib import Path
from bs4 import BeautifulSoup

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

# Find first race link and show parent chain
link = soup.find('a', href='/races/2026-george-e-mitchell-black-eyed-susan-s')
if link:
    print('=== RACE LINK ===')
    print('Text:', link.get_text(strip=True))
    print('Parent tag:', link.parent.name, 'class:', link.parent.get('class'))
    print()
    print('=== PARENT HTML ===')
    print(str(link.parent)[:500])
    print()
    print('=== GRANDPARENT ===')
    gp = link.parent.parent
    print(gp.name, gp.get('class'))
    print(str(gp)[:800])
    print()
    print('=== GREAT GRANDPARENT ===')
    ggp = gp.parent
    print(ggp.name, ggp.get('class'))
    print(str(ggp)[:1200])
