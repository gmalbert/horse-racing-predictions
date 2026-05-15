from pathlib import Path
import re

html = Path('data/raw/abr_stakes_2026_raw.html').read_text(encoding='utf-8', errors='replace')
print('Length:', len(html))

links = re.findall(r'/races/\d{4}-[^"\'<>\s]+', html)
print('Race links:', len(links), links[:10])

tlinks = re.findall(r'/tracks/[^"\'<>\s]+', html)
print('Track links:', len(tlinks), tlinks[:5])

dates = re.findall(r'\b\d{1,2}\.\d{2}\b', html)
print('Dot-dates:', len(dates), dates[:10])

# Search for Preakness
idx = html.lower().find('preakness')
print('\nPreakness at:', idx)
if idx >= 0:
    print(html[max(0,idx-200):idx+400])

# Search for "stakes" in links
stake_links = [l for l in links if 'stakes' in l.lower() or 's-s' in l.lower()]
print('\nStake-related links:', stake_links[:10])
