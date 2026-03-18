import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('ODDS_API_KEY')
if not key:
    raise SystemExit('ODDS_API_KEY not set')

url = 'https://api.the-odds-api.com/v4/sports'
resp = requests.get(url, params={'apiKey': key}, timeout=10)
resp.raise_for_status()
obj = resp.json()

horse = [s for s in obj if 'horse' in s.get('key', '') or 'horse' in s.get('title', '').lower()]
print('Total sports:', len(obj))
print('Horse-related sports:', len(horse))
for s in horse[:20]:
    print(f"{s.get('key')} - {s.get('title')}")
