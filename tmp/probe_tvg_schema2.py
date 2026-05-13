import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}

# Get all root Query fields
q = {"query": "{ __schema { queryType { fields { name args { name type { name kind ofType { name kind } } } } } } }"}
r = requests.post(GQL, headers=HDR, json=q, timeout=10)
fields = r.json()["data"]["__schema"]["queryType"]["fields"]

# Print all with "result" in name
print("=== Query fields with 'result' or 'race' in name ===")
for f in fields:
    n = f["name"].lower()
    if "result" in n or "past" in n:
        arg_names = [a["name"] for a in f["args"]]
        print(f"  {f['name']}  args={arg_names}")

# Try getTrackRacesWithRunners since we know it works - check if it has results sub-field
print()
q2 = {"query": "{ __type(name: \"Race\") { fields { name type { name kind ofType { name kind } } } } }"}
r2 = requests.post(GQL, headers=HDR, json=q2, timeout=10)
t = r2.json()["data"]["__type"]
if t:
    print("Race type fields:")
    for f in t["fields"]:
        print("  ", f["name"], "->", f["type"])

# Check RaceType too
print()
q3 = {"query": "{ __type(name: \"RaceType\") { fields { name type { name kind ofType { name kind } } } } }"}
r3 = requests.post(GQL, headers=HDR, json=q3, timeout=10)
t3 = r3.json()["data"]["__type"]
if t3:
    print("RaceType fields:")
    for f in t3["fields"]:
        print("  ", f["name"])
