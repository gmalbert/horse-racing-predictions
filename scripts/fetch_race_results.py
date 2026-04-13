#!/usr/bin/env python3
"""
Fetch Race Results from The Racing API and Append to Historical Dataset

Calls the /results endpoint for a date range, maps the response to the
all_gb_races.parquet schema, and appends new records so the historical
dataset stays current.

Usage:
  # Fetch results for a single date
  python scripts/fetch_race_results.py --date 2026-02-01

  # Backfill a date range
  python scripts/fetch_race_results.py --start 2026-02-01 --end 2026-04-12

  # Dry-run (print what would be fetched without writing)
  python scripts/fetch_race_results.py --start 2026-02-01 --end 2026-04-12 --dry-run

API call budget: ~1 call per day requested.  Calls are cached in
data/raw/results_YYYY-MM-DD.json so re-runs don't cost extra API calls.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
HISTORICAL   = PROCESSED / "all_gb_races.parquet"
CLEANED      = PROCESSED / "all_gb_races_cleaned.parquet"
BACKUP_DIR   = PROCESSED / "backups"

BASE_URL = "https://api.theracingapi.com/v1"

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def get_credentials():
    load_dotenv()
    username = os.getenv("RACING_API_USERNAME")
    password = os.getenv("RACING_API_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Racing API credentials not found.\n"
            "Add to .env:\n"
            "  RACING_API_USERNAME=your_username\n"
            "  RACING_API_PASSWORD=your_password"
        )
    return username, password


# ---------------------------------------------------------------------------
# API fetching
# ---------------------------------------------------------------------------

def fetch_results_for_date(date_str: str, region: str = "GB") -> list:
    """
    Fetch race results from the /results endpoint for one date.
    Caches the raw response in data/raw/results_YYYY-MM-DD.json.

    Returns a list of race dicts (empty list if none found / error).
    """
    cache_file = RAW_DIR / f"results_{date_str}.json"

    # Use cached response if available
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

    username, password = get_credentials()
    params = {"date": date_str, "region": region}

    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_URL}/results",
                auth=(username, password),
                params=params,
                timeout=30,
            )
            if resp.status_code == 401:
                body = resp.text or ""
                if "Standard Plan required" in body or "standard plan" in body.lower():
                    raise ValueError(
                        "The Racing API /results endpoint requires a Standard plan. "
                        "Please upgrade your Racing API plan or use an endpoint your plan supports."
                    )
                raise ValueError("Authentication failed. Check RACING_API_USERNAME / RACING_API_PASSWORD.")
            if resp.status_code == 404:
                print(f"  [skip]  {date_str}: no results (404)")
                return []
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  [rate]  {date_str}: rate limited; waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            data = resp.json()
            # Cache raw response
            RAW_DIR.mkdir(parents=True, exist_ok=True)
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
    """Normalise the API response to a flat list of race dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "races", "racecards", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------

def _safe(obj, *keys, default=None):
    """Safely navigate a nested dict."""
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj


def map_race_to_rows(race: dict) -> list[dict]:
    """
    Convert one race dict from the /results endpoint into a list of rows
    matching the all_gb_races.parquet schema (one row per runner).
    """
    race_id    = race.get("race_id") or race.get("id")
    date_str   = race.get("date") or race.get("date_of_race", "")
    course     = race.get("course") or race.get("venue") or ""
    course_id  = race.get("course_id")
    region     = race.get("region") or race.get("country") or "GB"
    off_time   = race.get("off_time") or race.get("off") or ""
    race_name  = race.get("race_name") or race.get("name") or ""
    race_type  = race.get("race_type") or race.get("type") or "Flat"
    race_class = race.get("race_class") or race.get("class")
    pattern    = race.get("pattern") or ""
    rating_band = race.get("rating_band") or ""
    age_band   = race.get("age_band") or ""
    sex_rest   = race.get("sex_restriction") or race.get("sex_rest") or ""
    distance   = race.get("distance") or race.get("dist") or ""
    dist_f     = race.get("distance_f") or race.get("dist_f")
    dist_m     = race.get("distance_m") or race.get("dist_m")
    dist_y     = race.get("distance_y") or race.get("dist_y")
    going      = race.get("going") or ""
    surface    = race.get("surface") or ""
    prize      = race.get("prize") or race.get("prize_money")
    runners    = race.get("runners") or race.get("horses") or []
    field_size = race.get("field_size") or race.get("num_runners") or len(runners)

    # Winner time (usually on the winning runner or at race level)
    race_time  = race.get("winning_time") or race.get("time") or ""

    rows = []
    for runner in runners:
        if not isinstance(runner, dict):
            continue

        # Position: try several field names used by different API versions
        pos = (
            runner.get("fin_pos")
            or runner.get("position")
            or runner.get("pos")
            or runner.get("finish_position")
        )
        if pos is None:
            continue  # skip runners with no recorded position (scratched etc.)

        horse_name = runner.get("name") or runner.get("horse") or ""
        horse_id   = runner.get("horse_id") or runner.get("id")

        row = {
            "date":        date_str,
            "region":      region,
            "course_id":   course_id,
            "course":      course,
            "course_detail": race.get("course_detail") or "",
            "race_id":     race_id,
            "off":         off_time,
            "race_name":   race_name,
            "type":        race_type,
            "class":       race_class,
            "pattern":     pattern,
            "rating_band": rating_band,
            "age_band":    age_band,
            "sex_rest":    sex_rest,
            "dist":        distance,
            "dist_f":      dist_f,
            "dist_m":      dist_m,
            "dist_y":      dist_y,
            "going":       going,
            "surface":     surface,
            "ran":         field_size,
            "num":         runner.get("number") or runner.get("num"),
            "pos":         str(pos),
            "draw":        runner.get("draw"),
            "ovr_btn":     runner.get("ovr_btn") or runner.get("overall_beaten"),
            "btn":         runner.get("btn") or runner.get("beaten_distance") or runner.get("dist_beaten"),
            "horse_id":    horse_id,
            "horse":       horse_name,
            "age":         runner.get("age"),
            "sex":         runner.get("sex_code") or runner.get("sex"),
            "lbs":         runner.get("lbs") or runner.get("weight_lbs"),
            "hg":          runner.get("headgear") or runner.get("hg") or "",
            "time":        runner.get("time") or race_time,
            "secs":        runner.get("time_secs") or runner.get("secs"),
            "dec":         runner.get("sp_dec") or runner.get("sp") or runner.get("decimal_odds"),
            "jockey_id":   runner.get("jockey_id"),
            "jockey":      runner.get("jockey"),
            "trainer_id":  runner.get("trainer_id"),
            "trainer":     runner.get("trainer"),
            "prize":       runner.get("prize") or runner.get("prize_won") or (prize if str(pos) == "1" else None),
            "or":          runner.get("ofr") or runner.get("official_rating") or runner.get("or"),
            "rpr":         runner.get("rpr"),
            "sire_id":     runner.get("sire_id"),
            "sire":        runner.get("sire"),
            "dam_id":      runner.get("dam_id"),
            "dam":         runner.get("dam"),
            "damsire_id":  runner.get("damsire_id"),
            "damsire":     runner.get("damsire"),
            "owner_id":    runner.get("owner_id"),
            "owner":       runner.get("owner"),
            "silk_url":    runner.get("silk_url") or runner.get("silk_path"),
            "_key":        f"{race_id}_{horse_id}" if race_id and horse_id else None,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_historical() -> pd.DataFrame:
    if not HISTORICAL.exists():
        print("  [warn]  all_gb_races.parquet not found; will create from scratch.")
        return pd.DataFrame()
    df = pd.read_parquet(HISTORICAL)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  [load]  Historical data: {len(df):,} rows, up to {df['date'].max().date()}")
    return df


def backup_historical():
    if not HISTORICAL.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"all_gb_races_backup_{ts}.parquet"
    shutil.copy2(HISTORICAL, dst)
    print(f"  [backup] {dst.name}")


def deduplicate(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows in new_df not already present in existing."""
    if existing.empty:
        return new_df

    existing_keys = set()
    if "_key" in existing.columns:
        existing_keys = set(existing["_key"].dropna())
    else:
        # Fall back to race_id + horse_id composite
        existing_keys = set(
            (existing["race_id"].astype(str) + "_" + existing["horse_id"].astype(str)).dropna()
        )

    if "_key" in new_df.columns:
        mask = ~new_df["_key"].isin(existing_keys)
    else:
        composite = new_df["race_id"].astype(str) + "_" + new_df["horse_id"].astype(str)
        mask = ~composite.isin(existing_keys)

    return new_df[mask].reset_index(drop=True)


def append_and_save(existing: pd.DataFrame, new_df: pd.DataFrame):
    """Append new rows to existing, align dtypes, save parquet."""
    if existing.empty:
        combined = new_df
    else:
        # Align columns
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[existing.columns]
        combined = pd.concat([existing, new_df], ignore_index=True)

    combined.to_parquet(HISTORICAL, index=False)
    print(f"  [save]  all_gb_races.parquet: {len(combined):,} rows total")


# ---------------------------------------------------------------------------
# Cleaned data regeneration (Phase 1)
# ---------------------------------------------------------------------------

def regenerate_cleaned():
    """Re-run phase1_data_cleaning.py to refresh all_gb_races_cleaned.parquet."""
    import subprocess
    script = PROJECT_ROOT / "scripts" / "phase1_data_cleaning.py"
    print("\nRegenerating all_gb_races_cleaned.parquet …")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    if result.returncode != 0:
        print("[warn]  phase1_data_cleaning.py returned non-zero exit code.")


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def date_range(start: str, end: str):
    """Yield YYYY-MM-DD strings from start to end inclusive."""
    d = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def auto_start_date(existing: pd.DataFrame) -> str:
    """Return the day after the last date in the existing dataset."""
    if existing.empty:
        return "2026-01-01"
    last = existing["date"].max()
    return (last + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch race results and update historical dataset")
    parser.add_argument("--date",    help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start",   help="Start of date range (YYYY-MM-DD)")
    parser.add_argument("--end",     help="End of date range (YYYY-MM-DD; default: yesterday)")
    parser.add_argument("--region",  default="GB", help="Region filter (default: GB)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched; don't write")
    parser.add_argument("--no-clean", action="store_true", help="Skip regenerating cleaned parquet")
    args = parser.parse_args()

    print("=" * 60)
    print("FETCH RACE RESULTS")
    print("=" * 60)

    existing = load_historical()

    # Determine date range
    if args.date:
        dates = [args.date]
    else:
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = args.start or auto_start_date(existing)
        end   = args.end   or yesterday
        dates = list(date_range(start, end))

    if not dates:
        print("No dates to fetch.")
        return

    print(f"\nFetching results for {len(dates)} date(s): {dates[0]} -> {dates[-1]}")
    if args.dry_run:
        print("[dry-run] No data will be written.")

    all_rows = []
    for d in dates:
        races = fetch_results_for_date(d, region=args.region)
        for race in races:
            all_rows.extend(map_race_to_rows(race))

    print(f"\nTotal runner-rows collected: {len(all_rows):,}")

    if not all_rows:
        print("Nothing to append.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])

    # Deduplicate
    deduped = deduplicate(existing, new_df)
    print(f"New rows after deduplication: {len(deduped):,}")

    if deduped.empty:
        print("All rows already present in dataset — nothing to append.")
        return

    if args.dry_run:
        print("[dry-run] Would append the above rows. Exiting without writing.")
        return

    backup_historical()
    append_and_save(existing, deduped)

    if not args.no_clean:
        regenerate_cleaned()

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
