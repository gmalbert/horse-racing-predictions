import json
with open("tmp/tvg_results_sample.json") as f:
    data = json.load(f)
us = data["us_tracks"]
print(f"US tracks: {len(us)}")
for tr in us:
    races = tr.get("races") or []
    results_races = [r for r in races if r.get("results") and r["results"].get("allRunners")]
    print(f"  {tr['code']} - {tr['name']}: {len(races)} races, {len(results_races)} with results")
    if results_races:
        sample = results_races[0]
        runners = sample["results"]["allRunners"]
        print(f"    Race {sample['number']} ({sample.get('description','')}): {len(runners)} runners")
        for rn in runners[:3]:
            print(f"      #{rn['runnerNumber']} {rn['runnerName']} pos={rn['finishPosition']} status={rn['finishStatus']}")
