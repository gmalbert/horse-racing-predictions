"""Export dated GB/US predictions against source market prices, never model fair odds."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"
DATA_FILES = ROOT / "data_files"
EV_THRESHOLD = .02
TOP_N_PER_RACE = 3


def _tier_from_edge(edge):
    return "Elite" if edge >= .06 else "Strong" if edge >= .03 else "Good" if edge >= .01 else "Standard"


def _safe_float(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _fractional_to_decimal(value):
    try:
        value = str(value).strip()
        if value.lower() in {"evs", "evens"}:
            return 2.
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            decimal = float(numerator) / float(denominator) + 1
        else:
            decimal = float(value) + 1
        return decimal if math.isfinite(decimal) and decimal > 1 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _decimal_to_american(decimal):
    if not decimal or not math.isfinite(decimal) or decimal <= 1:
        return None
    return round((decimal - 1) * 100) if decimal >= 2 else round(-100 / (decimal - 1))


def _write(bets, notes="", *, status=None, sources=None):
    DATA_FILES.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {
        "sport": "HorseRacing", "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0", "tier_definition": "edge-v1",
        "season": str(date.today().year), "status": status or ("ok" if bets else "no_picks"),
        "source_files": sources or [], "notes": notes,
    }, "bets": bets}
    path = DATA_FILES / "best_bets_today.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"[HorseRacing] {payload['meta']['status']}: {len(bets)} picks")


def export(for_date=None):
    for_date = for_date or datetime.now(ZoneInfo("America/New_York")).date()
    sources, rows = [], []
    for delta in range(8):
        day = for_date + timedelta(days=delta)
        for prefix in ("predictions", "us_predictions"):
            path = DATA_PROC / f"{prefix}_{day}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"horse", "win_probability", "course", "race_time", "date"}
                if not required.issubset(reader.fieldnames or []):
                    _write([], f"Invalid prediction columns in {path.name}", status="pipeline_failed")
                    raise ValueError(f"Invalid prediction columns in {path.name}")
                rows.extend(reader)
            sources.append(path.name)
    if not sources:
        _write([], "No dated GB or US predictions in the next seven days; historical files were not substituted", status="pipeline_pending")
        return

    candidates, priced = [], 0
    for row in rows:
        try:
            day = date.fromisoformat(str(row.get("date", ""))[:10])
        except ValueError:
            continue
        if not for_date <= day <= for_date + timedelta(days=7):
            continue
        probability = _safe_float(row.get("win_probability"))
        if probability is None or not 0 <= probability <= 1 or not row.get("horse"):
            continue
        decimal = _safe_float(row.get("market_odds"))
        price_type = "bookmaker"
        if decimal is None or decimal <= 1:
            decimal = _fractional_to_decimal(row.get("market_odds_fractional", ""))
            price_type = "morning_line" if "ml_odds" in row.get("odds_source", "") else "source_market"
        # win_odds_fractional is derived from model probability and is not a market quote.
        if decimal is None:
            continue
        priced += 1
        edge = round(probability - 1 / decimal, 4)
        if edge < EV_THRESHOLD:
            continue
        course, race_time = row.get("course", ""), row.get("race_time", "")
        race_name = row.get("race_name") or course
        notes = f"Price source: {row.get('odds_source') or 'merged market odds'}"
        if price_type == "morning_line":
            notes += "; morning-line estimate, not an executable bookmaker quote"
        candidates.append({
            "game_date": day.isoformat(), "game": f"{race_name} ({course})",
            "game_time": race_time or None, "bet_type": "win", "pick": row["horse"],
            "confidence": round(probability, 6), "edge": edge,
            "market_probability": round(1 / decimal, 6),
            "odds": _decimal_to_american(decimal), "tier": _tier_from_edge(edge),
            "price_type": price_type, "notes": notes, "course": course,
        })
    bets, counts, identities = [], Counter(), set()
    for bet in sorted(candidates, key=lambda item: item["edge"], reverse=True):
        race = (bet["game_date"], bet["course"], bet["game_time"])
        identity = (*race, bet["pick"])
        if identity in identities or counts[race] >= TOP_N_PER_RACE:
            continue
        identities.add(identity)
        counts[race] += 1
        bets.append(bet)
    notes = []
    if not priced:
        notes.append("No usable source market prices; model fair odds are excluded")
    if not any(name.startswith("predictions_") for name in sources):
        notes.append("Current GB predictions unavailable; only current US inputs were considered")
    _write(bets, "; ".join(notes),
           status=None if priced else "pipeline_pending", sources=sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    export(date.fromisoformat(args.date) if args.date else None)
