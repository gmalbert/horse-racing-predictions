import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}

# First check PastTrackLocation fields
r0 = requests.post(GQL, headers=HDR, json={"query": '{ __type(name: "PastTrackLocation") { fields { name } } }'}, timeout=10)
print("PastTrackLocation:", [f["name"] for f in r0.json()["data"]["__type"]["fields"]])

q = """
{
  pastTracks(profile: "PORT-Generic", date: "2026-05-09") {
    code
    name
    location { country }
    numberOfRaces
    races {
      id
      number
      date
      description
      distance { value }
      surface { name }
      raceClass { name }
      purse
      numRunners
      results {
        winningTime
        allRunners {
          runnerNumber
          runnerName
          finishPosition
          finishStatus
          winPayoff
          placePayoff
          showPayoff
          scratched
          favorite
        }
      }
    }
  }
}
"""

r = requests.post(GQL, headers=HDR, json={"query": q}, timeout=30)
data = r.json()

if "errors" in data:
    print("ERRORS:", data["errors"])
else:
    tracks = data["data"]["pastTracks"]
    us_tracks = [t for t in tracks if (t.get("location") or {}).get("country") in ("USA", "US", "United States")]
    print(f"\nUS tracks on 2026-05-09: {len(us_tracks)}")
    # Show all countries
    countries = sorted(set((t.get("location") or {}).get("country","?") for t in tracks))
    print("Countries found:", countries)
    # Show first US track details
    for tr in us_tracks[:5]:
        print(f"\n  {tr['code']} - {tr['name']}: {tr['numberOfRaces']} races")
        for race in (tr.get("races") or [])[:2]:
            res = race.get("results")
            if res and res.get("allRunners"):
                runners = res["allRunners"]
                placed = [x for x in runners if x.get("finishPosition")]
                print(f"    Race {race['number']} ({race['description']}) - {len(runners)} runners, {len(placed)} with positions")
                for runner in runners[:4]:
                    print(f"      #{runner['runnerNumber']} {runner['runnerName']} - pos={runner['finishPosition']}")
            else:
                print(f"    Race {race['number']} - no results yet")

    with open("tmp/tvg_results_sample.json", "w") as f:
        json.dump({"us_tracks": us_tracks}, f, indent=2)
    print("\nSaved to tmp/tvg_results_sample.json")
