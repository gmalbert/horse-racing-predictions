import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}


def post(q):
    r = requests.post(GQL, headers=HDR, json={"query": q}, timeout=15)
    return r.json()

# Inspect Track type from pastTracks
print("=== PastTrack type ===")
d = post('{ __type(name: "PastTrack") { fields { name type { name kind ofType { name kind } } } } }')
t = d.get("data", {}).get("__type")
if t:
    for f in t["fields"]: print(" ", f["name"], "->", f["type"])

print()

# Inspect PastRace type
print("=== PastRace type ===")
d2 = post('{ __type(name: "PastRace") { fields { name type { name kind ofType { name kind } } } } }')
t2 = d2.get("data", {}).get("__type")
if t2:
    for f in t2["fields"]: print(" ", f["name"], "->", f["type"])

print()

# Try pastTracks query for a specific date
print("=== pastTracks for 2026-05-09 ===")
d3 = post("""
{
  pastTracks(date: "2026-05-09") {
    code
    name
    races {
      number
      distance
    }
  }
}
""")
print(json.dumps(d3, indent=2)[:1500])
