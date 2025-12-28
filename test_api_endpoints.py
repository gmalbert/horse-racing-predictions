import requests

auth = ('vlj0wIJtB3M1sUJxRtaQW9BM', 'nBwqfmSMKJuQuv5LkSrmzskC')
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
