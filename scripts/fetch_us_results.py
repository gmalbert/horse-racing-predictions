"""
fetch_us_results.py — Fetch US race results from The Racing API and build
                      data/processed/us_races_cleaned.parquet for model training.

This is the US-specific counterpart to fetch_race_results.py.
It reuses the same /results endpoint with region='US', caches raw JSON in
data/raw/us_results_YYYY-MM-DD.json, and appends new runner-rows to
data/processed/us_races_cleaned.parquet.

Usage:
    python scripts/fetch_us_results.py --date 2026-05-11
    python scripts/fetch_us_results.py --start 2026-01-01 --end 2026-05-11
    python scripts/fetch_us_results.py --days 30          # last 30 days
    python scripts/fetch_us_results.py --dry-run          # validate without writing

API call budget: 1 call per day requested.  Responses are cached so re-runs
are free.  Track your monthly quota (500 calls/month on standard plan).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR     = REPO_ROOT / "data" / "raw"
PROC_DIR    = REPO_ROOT / "data" / "processed"
US_RESULTS  = PROC_DIR / "us_races_cleaned.parquet"
BASE_URL    = "https://api.theracingapi.com/v1"

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_credentials() -> tuple[str, str]:
    username = os.getenv("RACING_API_USERNAME")
    password = os.getenv("RACING_API_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Racing API credentials not found. Add to .env:\n"
            "  RACING_API_USERNAME=your_username\n"
            "  RACING_API_PASSWORD=your_password"
        )
    return username, password


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_results_for_date(date_str: str) -> list[dict]:
    """
    Fetch US race results for *date_str* (YYYY-MM-DD).
    Caches raw response in data/raw/us_results_YYYY-MM-DD.json.
    Returns a flat list of race dicts (empty if none found or error).
    """
    cache_file = RAW_DIR / f"us_results_{date_str}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                races = _extract_races(data)
                print(f"  [cache] {date_str}: {len(races)} races")
                return races
            except json.JSONDecodeError:
                print(f"  [warn]  {date_str}: cached file invalid, re-fetching")
                cache_file.unlink()

    username, password = _get_credentials()

    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_URL}/results",
                auth=(username, password),
                params={"date": date_str, "region": "US"},
                timeout=30,
            )
            if resp.status_code == 401:
                body = resp.text or ""
                if "standard plan" in body.lower():
                    raise ValueError(
                        "The Racing API /results endpoint requires a Standard plan. "
                        "Please upgrade your plan."
                    )
                raise ValueError("Authentication failed. Check credentials in .env.")
            if resp.status_code == 404:
                print(f"  [skip]  {date_str}: no US results (404)")
                return []
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  [rate]  {date_str}: rate limited; waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            data = resp.json()
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh)

            races = _extract_races(data)
            print(f"  [api]   {date_str}: {len(races)} races fetched")
            return races

        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  [error] {date_str}: {exc}")
                return []
            time.sleep(2 ** attempt)

    return []


def _extract_races(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "races", "racecards", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


# ---------------------------------------------------------------------------
# Schema mapping — US-adapted version of fetch_race_results.map_race_to_rows
# ---------------------------------------------------------------------------

def map_race_to_rows(race: dict) -> list[dict]:
    """
    Convert one US race dict from the /results endpoint into runner rows
    matching the us_races_cleaned.parquet schema.
    Columns align with what train_us_model.py expects.
    """
    race_id    = race.get("race_id") or race.get("id")
    date_str   = race.get("date") or race.get("date_of_race", "")
    course     = race.get("course") or race.get("venue") or ""
    race_name  = race.get("race_name") or race.get("name") or ""
    race_class = race.get("race_class") or race.get("class") or ""
    distance   = race.get("distance") or race.get("dist") or ""
    dist_f     = race.get("distance_f") or race.get("dist_f")
    going      = race.get("going") or ""
    surface    = race.get("surface") or ""
    prize      = race.get("prize") or race.get("prize_money")
    runners    = race.get("runners") or race.get("horses") or []
    field_size = race.get("field_size") or race.get("num_runners") or len(runners)

    rows = []
    for runner in runners:
        if not isinstance(runner, dict):
            continue

        pos = (
            runner.get("fin_pos")
            or runner.get("position")
            or runner.get("pos")
            or runner.get("finish_position")
        )
        if pos is None:
            continue  # skip unplaced / scratched runners with no position recorded

        horse_name = runner.get("name") or runner.get("horse") or ""
        horse_id   = runner.get("horse_id") or runner.get("id")

        row = {
            # Core fields required by train_us_model.py
            "horse":       horse_name,
            "horse_id":    horse_id,
            "date":        date_str,
            "course":      course,
            "surface":     surface or "Dirt",
            "distance":    distance,
            "distance_f":  dist_f,
            "race_class":  race_class,
            "race_name":   race_name,
            "position":    str(pos),
            "going":       going,
            # Additional context useful for feature engineering
            "race_id":     race_id,
            "field_size":  field_size,
            "prize":       runner.get("prize") or (prize if str(pos) == "1" else None),
            "draw":        runner.get("draw"),
            "age":         runner.get("age"),
            "sex":         runner.get("sex_code") or runner.get("sex"),
            "lbs":         runner.get("lbs") or runner.get("weight_lbs"),
            "headgear":    runner.get("headgear") or runner.get("hg") or "",
            "dec":         runner.get("sp_dec") or runner.get("sp") or runner.get("decimal_odds"),
            "jockey_id":   runner.get("jockey_id"),
            "jockey":      runner.get("jockey"),
            "trainer_id":  runner.get("trainer_id"),
            "trainer":     runner.get("trainer"),
            "official_rating": runner.get("ofr") or runner.get("official_rating") or runner.get("or"),
            "sire":        runner.get("sire"),
            "dam":         runner.get("dam"),
            "damsire":     runner.get("damsire"),
            "owner":       runner.get("owner"),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_existing() -> pd.DataFrame:
    if not US_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(US_RESULTS)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  [load]  us_races_cleaned.parquet: {len(df):,} rows, up to {df['date'].max().date()}")
    return df


def deduplicate(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_df

    if "race_id" in existing.columns and "horse" in existing.columns:
        existing_keys = set(
            (existing["race_id"].astype(str) + "_" + existing["horse"].astype(str)).dropna()
        )
        composite = new_df["race_id"].astype(str) + "_" + new_df["horse"].astype(str)
        return new_df[~composite.isin(existing_keys)].reset_index(drop=True)

    return new_df


def append_and_save(existing: pd.DataFrame, new_df: pd.DataFrame):
    if existing.empty:
        combined = new_df
    else:
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df.reindex(columns=existing.columns)
        combined = pd.concat([existing, new_df], ignore_index=True)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(US_RESULTS, index=False)
    print(f"  [save]  us_races_cleaned.parquet: {len(combined):,} rows total")


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _date_range(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def _auto_start(existing: pd.DataFrame) -> str:
    if existing.empty:
        return "2026-01-01"
    last = existing["date"].max()
    return (last + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch US race results from The Racing API and update us_races_cleaned.parquet"
    )
    parser.add_argument("--date",    help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start",   help="Start of date range (YYYY-MM-DD)")
    parser.add_argument("--end",     help="End of date range (YYYY-MM-DD; default: yesterday)")
    parser.add_argument("--days",    type=int, help="Number of days back from today to fetch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without writing")
    args = parser.parse_args()

    print("=" * 60)
    print("FETCH US RACE RESULTS  (The Racing API — region=US)")
    print("=" * 60)

    existing = load_existing()

    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    if args.date:
        dates = [args.date]
    elif args.days:
        end_d = date.today() - timedelta(days=1)
        start_d = end_d - timedelta(days=args.days - 1)
        dates = list(_date_range(start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")))
    else:
        start = args.start or _auto_start(existing)
        end   = args.end   or yesterday
        dates = list(_date_range(start, end))

    if not dates:
        print("No dates to fetch.")
        return

    print(f"\nFetching US results for {len(dates)} date(s): {dates[0]} → {dates[-1]}")
    if args.dry_run:
        print("[dry-run] No data will be written.\n")

    all_rows: list[dict] = []
    for d in dates:
        races = fetch_results_for_date(d)
        for race in races:
            all_rows.extend(map_race_to_rows(race))

    print(f"\nTotal runner-rows collected: {len(all_rows):,}")

    if not all_rows:
        print("Nothing to save.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])

    deduped = deduplicate(existing, new_df)
    print(f"New rows after deduplication: {len(deduped):,}")

    if deduped.empty:
        print("All rows already present — nothing to append.")
        return

    if args.dry_run:
        print("[dry-run] Would append the above rows. Exiting without writing.")
        print("\nSample:")
        print(deduped[["date", "course", "race_name", "horse", "position"]].head(10).to_string(index=False))
        return

    append_and_save(existing, deduped)
    print(f"\nDone. Run  python scripts/train_us_model.py  when you have enough rows (≥50k).")
    print("For early testing: python scripts/train_us_model.py --min-rows 1000")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
