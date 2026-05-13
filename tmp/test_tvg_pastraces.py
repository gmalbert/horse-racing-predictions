import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}


def post(query):
    r = requests.post(GQL, headers=HDR, json={"query": query}, timeout=15)
    return r.json()


# Try pastRaces query directly for a known track+date
print("=== pastRaces for CD 2026-05-09 ===")
d = post("""
{
  pastRaces(profile: "PORT-Generic", trackCode: "CD", date: "2026-05-09") {
    id
    number
    date
    description
    distance { value }
    surface { name }
    raceClass { name }
    purse
    numRunners
    track { code name }
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
      }
    }
  }
}
""")
if "errors" in d:
    print("ERRORS:", d["errors"])
else:
    races = d["data"].get("pastRaces") or []
    print(f"Races returned: {len(races)}")
    for race in races[:3]:
        res = race.get("results")
        runners = (res or {}).get("allRunners") or []
        placed = [x for x in runners if x.get("finishPosition")]
        print(f"  Race {race['number']} {race.get('description','')} - {len(runners)} runners, {len(placed)} placed")
        for rn in placed[:4]:
            print(f"    #{rn['runnerNumber']} {rn['runnerName']} pos={rn['finishPosition']} status={rn['finishStatus']}")
