import os
import requests
from dotenv import load_dotenv

load_dotenv()
username = os.getenv('RACING_API_USERNAME')
password = os.getenv('RACING_API_PASSWORD')
if not username or not password:
    raise SystemExit('ERROR: Set RACING_API_USERNAME and RACING_API_PASSWORD in .env')
auth = (username, password)
base = 'https://api.theracingapi.com/v1'

endpoints = [
    '/courses',
    '/races',
    '/racecards',
    '/results',
    '/horses',
    '/meetings'
]

print("Testing Racing API free tier endpoints:\n")
for endpoint in endpoints:
    try:
        r = requests.get(f'{base}{endpoint}', auth=auth, timeout=10)
        print(f'{endpoint:15} -> Status: {r.status_code:3} | {r.text[:80]}')
    except Exception as e:
        print(f'{endpoint:15} -> Error: {e}')
