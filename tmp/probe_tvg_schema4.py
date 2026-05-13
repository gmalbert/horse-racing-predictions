import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}


def post(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = requests.post(GQL, headers=HDR, json=payload, timeout=15)
    return r.json()

# Try with PORT-Generic profile
print("=== pastTracks for 2026-05-09 with profile ===")
d = post("""
{
  pastTracks(profile: "PORT-Generic", date: "2026-05-09") {
    code
    name
    numberOfRaces
    races {
      number
      date
      description
      distance { value }
      surface { name }
      raceClass { name }
      purse
      numRunners
    }
  }
}
""")
if "errors" in d:
    print("ERRORS:", json.dumps(d["errors"]))
else:
    tracks = d.get("data", {}).get("pastTracks", [])
    print(f"Found {len(tracks)} tracks")
    for tr in tracks[:5]:
        print(f"  {tr['code']} - {tr['name']}: {tr['numberOfRaces']} races")

# Inspect PastResultsType
print()
print("=== PastResultsType ===")
d2 = post('{ __type(name: "PastResultsType") { fields { name type { name kind ofType { name kind } } } } }')
t = d2.get("data", {}).get("__type")
if t:
    for f in t["fields"]: print(" ", f["name"], "->", f["type"])

# Inspect PastBettingInterest
print()
print("=== PastBettingInterest ===")
d3 = post('{ __type(name: "PastBettingInterest") { fields { name type { name kind ofType { name kind } } } } }')
t3 = d3.get("data", {}).get("__type")
if t3:
    for f in t3["fields"]: print(" ", f["name"])
