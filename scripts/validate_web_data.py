"""Validate web artifacts against their upstream files; optionally enforce freshness."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web/frontend/public/data"


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-days", type=int, help="Fail if predictions/results exceed this age")
    args = parser.parse_args()
    payloads = {str(p.relative_to(DATA)).replace("\\", "/"): json.loads(p.read_text(encoding="utf-8-sig"), parse_constant=reject_constant) for p in DATA.rglob("*.json")}
    required = {"manifest.json", "schedule_upcoming.json", "fixtures_scored.json", "us_schedule.json",
                "us_stakes.json", "us_top_races.json", "model_insights.json", "us_model_insights.json",
                "explorer_overview.json", "backtest.json", "betting_watchlist.json"}
    missing = required - payloads.keys()
    if missing:
        raise ValueError(f"Missing artifacts: {sorted(missing)}")
    manifest = payloads["manifest.json"]
    for variant, prefix in [("gb", "predictions"), ("us", "us_predictions")]:
        dates = manifest[f"{variant}_prediction_dates"]
        assert dates == sorted(set(dates)), f"Unsorted/duplicate {variant} dates"
        for stamp in dates:
            day = payloads[f"{prefix}/{stamp}.json"]
            assert day["date"] == stamp
            assert day["summary"]["races"] == len(day["races"])
            assert day["summary"]["horses"] == sum(len(race["horses"]) for race in day["races"])
        upstream = sorted(p.stem.removeprefix(prefix + "_") for p in (ROOT / "data/processed").glob(prefix + "_????-??-??.csv") if p.stat().st_size > 2)
        if upstream:
            assert dates and dates[-1] == upstream[-1], f"{variant} export lags upstream {upstream[-1]}"
    for window in manifest["stats_windows"]:
        for entity in ("trainer", "jockey"):
            stats = payloads[f"{entity}_stats_{window}.json"]
            assert stats["as_of"] == manifest["stats_as_of"]
            assert stats["window_days"] == window
    freshness = {}
    for name, stamp in [("gb", (manifest["gb_prediction_dates"] or [None])[-1]),
                        ("us", (manifest["us_prediction_dates"] or [None])[-1]),
                        ("results", manifest.get("stats_as_of"))]:
        age = (date.today() - date.fromisoformat(stamp)).days if stamp else None
        freshness[name] = {"latest": stamp, "age_days": age}
        if args.max_age_days is not None:
            assert age is not None and age <= args.max_age_days, f"Stale {name} data: {stamp} ({age} days)"
    print(json.dumps({"status": "valid", "artifacts": len(payloads), "source_freshness": freshness}, indent=2))


if __name__ == "__main__":
    main()
