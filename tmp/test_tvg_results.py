import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}

q = """
{
  pastTracks(profile: "PORT-Generic", date: "2026-05-09") {
    code
    name
    location { country stateAbbr }
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
          currentOdds { numerator denominator }
        }
        payoffs {
          wagerType
          wagerAmount
          selections
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
    # Filter to US tracks only
    us_tracks = [t for t in tracks if t.get("location", {}).get("country") == "USA"]
    print(f"US tracks on 2026-05-09: {len(us_tracks)}")
    for tr in us_tracks[:3]:
        print(f"\n  {tr['code']} - {tr['name']} ({tr['location']['stateAbbr']}): {tr['numberOfRaces']} races")
        for race in tr["races"][:2]:
            res = race.get("results")
            if res and res.get("allRunners"):
                runners = res["allRunners"]
                print(f"    Race {race['number']} ({race['description']}) - {race['distance']['value'] if race['distance'] else 'N/A'}")
                for runner in runners[:4]:
                    print(f"      #{runner['runnerNumber']} {runner['runnerName']} - pos={runner['finishPosition']} status={runner['finishStatus']}")
            else:
                print(f"    Race {race['number']} - no results")

# Save raw sample
with open("tmp/tvg_results_sample.json", "w") as f:
    json.dump({"us_tracks": [t for t in tracks if t.get("location", {}).get("country") == "USA"]}, f, indent=2)
print("\nSaved sample to tmp/tvg_results_sample.json")
