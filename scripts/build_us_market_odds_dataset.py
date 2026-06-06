#!/usr/bin/env python3
"""Build a historical US market-odds dataset from cached racecard files.

Extracts fractional morning-line odds from data/raw/us_racecards_*.json and
normalizes them into a reusable dataset for backtesting joins.

Outputs:
- data/processed/us_market_odds.csv
- data/processed/us_market_odds.parquet
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")


def _fractional_to_decimal(odds_raw: str) -> float | None:
    s = str(odds_raw).strip().lower()
    if not s or s in {"-", "n/a", "none"}:
        return None
    if s in {"evens", "even", "1/1", "1-1"}:
        return 2.0
    m = re.match(r"^(\d+)\s*[/\-]\s*(\d+)$", s)
    if not m:
        return None
    num = int(m.group(1))
    den = int(m.group(2))
    if den <= 0:
        return None
    return round(1.0 + (num / den), 4)


def _strip_country_suffix(name: str) -> str:
    return re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", str(name or "")).strip().lower()


def _extract_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    date = payload.get("date") or path.stem.replace("us_racecards_", "")
    races = payload.get("racecards") or payload.get("races") or []

    rows = []
    for race in races:
        course = race.get("course") or race.get("track") or race.get("track_name") or ""
        race_name = race.get("race_name") or race.get("race") or ""
        race_time = race.get("race_time") or race.get("time") or ""
        for runner in race.get("runners") or []:
            horse = runner.get("horse") or runner.get("name") or ""
            frac = runner.get("ml_odds") or runner.get("odds") or ""
            dec = _fractional_to_decimal(frac)
            if dec is None:
                continue
            rows.append(
                {
                    "date": date,
                    "course": str(course),
                    "race_name": str(race_name),
                    "race_time": str(race_time),
                    "horse": str(horse),
                    "horse_norm": _strip_country_suffix(horse),
                    "market_odds": float(dec),
                    "market_odds_fractional": str(frac),
                    "source_file": path.name,
                }
            )
    return rows


def main() -> int:
    files = sorted(RAW_DIR.glob("us_racecards_*.json"))
    if not files:
        print("No us_racecards_*.json files found in data/raw")
        return 1

    all_rows = []
    for f in files:
        try:
            all_rows.extend(_extract_rows(f))
        except Exception as exc:
            print(f"[WARN] Failed to parse {f.name}: {exc}")

    if not all_rows:
        print("No market odds extracted from US racecards.")
        return 1

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["date", "course", "race_name", "horse_norm"], keep="last")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROC_DIR / "us_market_odds.csv"
    pq_path = PROC_DIR / "us_market_odds.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    print(f"Saved {len(df):,} rows to {csv_path}")
    print(f"Saved {len(df):,} rows to {pq_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
