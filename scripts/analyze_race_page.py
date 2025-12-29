import re

with open('debug_race_page_odds.html', encoding='utf-8') as f:
    content = f.read()

print(f'File size: {len(content):,} bytes\n')

# Look for horse names
print('=== Searching for HORSE NAMES ===')
horse_links = re.findall(r'href="[^"]*\/horses\/[^"]*"[^>]*>([^<]+)<', content)
print(f'Found {len(horse_links)} horse links:', horse_links[:10])

# Look for runner names in different patterns
runner_patterns = [
    (r'data-test-id="[^"]*runner[^"]*"[^>]*>([^<]+)', 'runner test-id'),
    (r'class="[^"]*runner[^"]*name[^"]*"[^>]*>([^<]+)', 'runner name class'),
    (r'"horseName":"([^"]+)"', 'JSON horseName'),
    (r'"runnerName":"([^"]+)"', 'JSON runnerName'),
]
for pattern, desc in runner_patterns:
    matches = re.findall(pattern, content, re.I)
    if matches:
        print(f'{desc}: {matches[:10]}')

# Look for odds
print('\n=== Searching for ODDS ===')
odds_patterns = [
    (r'class="[^"]*odds[^"]*"[^>]*>([0-9/]+)', 'odds class'),
    (r'data-test-id="odds"[^>]*>([^<]+)', 'odds test-id'),
    (r'<span[^>]*>([0-9]+/[0-9]+)</span>', 'fractional odds'),
    (r'"fractional":"([^"]+)"', 'JSON fractional'),
    (r'"decimal":([\d.]+)', 'JSON decimal'),
    (r'"oddsDecimal":([\d.]+)', 'JSON oddsDecimal'),
    (r'"oddsFractional":"([^"]+)"', 'JSON oddsFractional'),
]

for pattern, desc in odds_patterns:
    matches = re.findall(pattern, content)
    if matches:
        print(f'{desc}: {matches[:15]}')

# Check for login requirement
print('\n=== Checking for ACCESS RESTRICTIONS ===')
if 'login' in content.lower():
    login_contexts = re.findall(r'.{0,60}login.{0,60}', content, re.I)
    print(f'Found "login" {len(login_contexts)} times (first 2):')
    for ctx in login_contexts[:2]:
        print(f'  - {ctx.strip()[:80]}')
else:
    print('NO LOGIN REQUIRED!')

if 'mandatoryLogIn' in content or 'mandatory' in content.lower():
    print('Found mandatory login elements')
else:
    print('NO MANDATORY LOGIN!')

# Look for bookmaker names (sign of odds comparison page)
print('\n=== Checking for BOOKMAKERS ===')
bookmakers = ['bet365', 'paddy power', 'william hill', 'betfair', 'coral', 'ladbrokes', 'skybet']
for bookie in bookmakers:
    if bookie.replace(' ', '') in content.lower().replace(' ', ''):
        print(f'Found: {bookie}')

