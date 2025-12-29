"""Check if The Odds API supports horse racing."""

import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ODDS_API_KEY')

if not api_key:
    print("❌ No ODDS_API_KEY found in .env")
    exit(1)

response = requests.get(
    'https://api.the-odds-api.com/v4/sports',
    params={'apiKey': api_key}
)

sports = response.json()
horse_racing = [s for s in sports if 'horse' in s.get('title', '').lower() or 'horse' in s.get('key', '')]

print(f'Total sports available: {len(sports)}')
print(f'\nHorse racing sports: {len(horse_racing)}')

if horse_racing:
    for s in horse_racing:
        print(f"  ✓ {s['title']} ({s['key']})")
        print(f"    Group: {s.get('group', 'Unknown')}")
else:
    print("  ❌ No horse racing sports found")
    print("\n  The Odds API may not support horse racing")
    print("  Supported sports:")
    for s in sports[:10]:
        print(f"    - {s['title']}")

print(f'\nAPI calls remaining: {response.headers.get("x-requests-remaining")}')
