import re

with open('debug_odds_comp_1.html', encoding='utf-8') as f:
    content = f.read()

# Look for horse links
horse_links = re.findall(r'href="([^"]*\/horses\/[^"]+)"', content)
print(f'Horse links found: {len(horse_links)}')
if horse_links:
    print('First 10:', horse_links[:10])
    
    # Extract horse names from the links
    names = [link.split('/')[-1].replace('-', ' ').title() for link in horse_links[:10]]
    print('\nHorse names from links:', names)

# Look for visible horse names
horse_names = re.findall(r'<a[^>]*href="[^"]*\/horses\/[^"]*"[^>]*>([^<]+)</a>', content)
print(f'\nHorse name texts found: {len(horse_names)}')
if horse_names:
    print('First 10:', horse_names[:10])

# Check if there's a login wall
if 'RC-mandatoryLogIn' in content:
    print('\n⚠️  MANDATORY LOGIN ELEMENT FOUND')
    # Check if it's visible
    if 'display: none' in content[content.find('RC-mandatoryLogIn'):content.find('RC-mandatoryLogIn')+500]:
        print('   (but may be hidden)')
