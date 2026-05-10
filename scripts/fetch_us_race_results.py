"""
Fetch US Race Results from The Racing API and build a historical dataset.

Calls /results endpoint with region='US' for a date range, maps the response
to a flat Parquet schema (one row per runner), and writes / appends to
data/processed/all_us_races.parquet.

API call budget: 1 call per day requested.
Raw responses are cached in data/raw/us_results_YYYY-MM-DD.json so re-runs
cost zero extra API calls for already-fetched dates.

Usage:
    # Single date
    python scripts/fetch_us_race_results.py --date 2026-05-09

    # Backfill two years of history  (spread over 2 months to stay within 500/month limit)
    python scripts/fetch_us_race_results.py --start 2024-01-01 --end 2025-12-31

    # Dry-run (shows what would be fetched without touching files)
    python scripts/fetch_us_race_results.py --start 2024-01-01 --end 2024-03-31 --dry-run

    # Skip days that have already been cached (default behaviour)
    python scripts/fetch_us_race_results.py --start 2024-01-01 --end 2024-12-31

Rate-limit note
---------------
The Racing API allows 500 calls/month on the base plan.
US history for 2 years ≈ 730 days = 730 calls.
To stay within limits: fetch ~240 days per month, or request a higher-tier plan.
Use --batch-size to control how many days are fetched in a single run:

    python scripts/fetch_us_race_results.py --start 2024-01-01 --end 2026-05-09 --batch-size 200
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

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
HISTORICAL   = PROC_DIR / "all_us_races.parquet"
BACKUP_DIR   = PROC_DIR / "backups"

BASE_URL = "https://api.theracingapi.com/v1"


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_credentials() -> tuple[str, str]:
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


# ── API fetch (with caching) ──────────────────────────────────────────────────

def fetch_results_for_date(date_str: str, dry_run: bool = False) -> list[dict]:
    """
    Fetch US race results for one date from the /results endpoint.

    Caches the raw JSON in data/raw/us_results_YYYY-MM-DD.json.
    Returns a flat list of race dicts (empty list if none / error).
    """
    cache_file = RAW_DIR / f"us_results_{date_str}.json"

    # Return cached response if available
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                races = _extract_races(data)
                print(f"  [cache] {date_str}: {len(races)} races")
                return races
            except json.JSONDecodeError:
                print(f"  [warn]  {date_str}: bad cache; re-fetching")
                cache_file.unlink()

    if dry_run:
        print(f"  [dry]   {date_str}: would fetch from API")
        return []

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
                if "standard plan" in body.lower() or "upgrade" in body.lower():
                    raise SystemExit(
                        "ERROR: The Racing API /results endpoint requires a Standard plan.\n"
                        "       Please upgrade at https://www.theracingapi.com/plans"
                    )
                raise ValueError("Authentication failed. Check RACING_API_USERNAME / RACING_API_PASSWORD.")

            if resp.status_code == 404:
                print(f"  [skip]  {date_str}: no US results (404)")
                # Cache empty result so we don't re-fetch
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text('{"results":[]}', encoding="utf-8")
                return []

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  [rate]  {date_str}: rate limited — waiting {wait}s …")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Cache raw response
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh)

            races = _extract_races(data)
            print(f"  [api]   {date_str}: {len(races)} races")
            return races

        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  [error] {date_str}: {exc}")
                return []
            time.sleep(2 ** attempt)

    return []


def _extract_races(data) -> list:
    """Normalise API response to a flat list of race dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "races", "racecards", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


# ── Schema mapping ────────────────────────────────────────────────────────────

def _safe(obj, *keys, default=None):
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj


def map_race_to_rows(race: dict) -> list[dict]:
    """
    Convert one US race dict into a list of rows (one per runner).

    US-specific fields:
    - surface: Dirt / Turf / Synthetic
    - class: Grade I, Claiming $25000, Allowance, etc.
    - going: Fast, Good, Muddy, Sloppy, Heavy, Yielding, Soft, Firm
    - distance: furlongs or fractional miles (parsed separately)
    """
    race_id    = race.get("race_id") or race.get("id")
    date_str   = race.get("date") or race.get("date_of_race") or ""
    course     = race.get("course") or race.get("venue") or race.get("track") or ""
    course_id  = race.get("course_id")
    off_time   = race.get("off_time") or race.get("off") or race.get("post_time") or ""
    race_name  = race.get("race_name") or race.get("name") or ""
    race_type  = race.get("race_type") or race.get("type") or ""
    race_class = race.get("race_class") or race.get("class") or ""
    pattern    = race.get("pattern") or ""
    distance   = race.get("distance") or race.get("dist") or ""
    dist_f     = race.get("distance_f") or race.get("dist_f")
    going      = race.get("going") or ""
    surface    = race.get("surface") or ""
    prize      = race.get("prize") or race.get("prize_money") or race.get("purse")
    runners    = race.get("runners") or race.get("horses") or []
    field_size = race.get("field_size") or race.get("num_runners") or len(runners)
    state      = race.get("state") or race.get("region_detail") or ""

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
            continue  # skip scratched / no result

        horse_name = runner.get("name") or runner.get("horse") or ""
        horse_id   = runner.get("horse_id") or runner.get("id")

        row = {
            # Race context
            "date":        date_str,
            "region":      "US",
            "state":       state,
            "course_id":   course_id,
            "course":      course,
            "race_id":     race_id,
            "off":         off_time,
            "race_name":   race_name,
            "type":        race_type,
            "class":       race_class,
            "pattern":     pattern,
            "dist":        distance,
            "dist_f":      dist_f,
            "going":       going,
            "surface":     surface,   # Dirt / Turf / Synthetic
            "ran":         field_size,
            "prize":       prize,
            # Runner
            "pos":         str(pos),
            "num":         runner.get("number") or runner.get("num") or runner.get("program_number"),
            "draw":        runner.get("draw") or runner.get("post_position"),
            "btn":         runner.get("btn") or runner.get("beaten_distance") or runner.get("dist_beaten"),
            "ovr_btn":     runner.get("ovr_btn") or runner.get("overall_beaten"),
            "horse_id":    horse_id,
            "horse":       horse_name,
            "age":         runner.get("age"),
            "sex":         runner.get("sex_code") or runner.get("sex"),
            "lbs":         runner.get("lbs") or runner.get("weight_lbs") or runner.get("weight"),
            "hg":          runner.get("headgear") or runner.get("equipment") or "",
            "time":        runner.get("time") or runner.get("winning_time"),
            "secs":        runner.get("time_secs") or runner.get("secs"),
            # Odds — US uses decimal / moneyline (no fractional BHA OR)
            "dec":         runner.get("sp_dec") or runner.get("sp") or runner.get("decimal_odds") or runner.get("win_odds"),
            "ml_odds":     runner.get("morning_line") or runner.get("ml"),
            # Connections
            "jockey_id":   runner.get("jockey_id"),
            "jockey":      runner.get("jockey"),
            "trainer_id":  runner.get("trainer_id"),
            "trainer":     runner.get("trainer"),
            "owner_id":    runner.get("owner_id"),
            "owner":       runner.get("owner"),
            # Pedigree
            "sire_id":     runner.get("sire_id"),
            "sire":        runner.get("sire"),
            "dam_id":      runner.get("dam_id"),
            "dam":         runner.get("dam"),
            "damsire_id":  runner.get("damsire_id"),
            "damsire":     runner.get("damsire"),
            # Prize won by THIS runner
            "prize_won":   runner.get("prize") or runner.get("prize_won"),
            # Unique key
            "_key":        f"{race_id}_{horse_id}" if race_id and horse_id else None,
        }
        rows.append(row)

    return rows


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_historical() -> pd.DataFrame:
    if not HISTORICAL.exists():
        print("  [info]  all_us_races.parquet does not exist — will create from scratch")
        return pd.DataFrame()
    df = pd.read_parquet(HISTORICAL)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  [load]  {len(df):,} existing US rows, up to {df['date'].max().date()}")
    return df


def _backup():
    if not HISTORICAL.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"all_us_races_backup_{ts}.parquet"
    shutil.copy2(HISTORICAL, dst)
    print(f"  [backup] {dst.name}")


def _deduplicate(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_df
    if "_key" in existing.columns and "_key" in new_df.columns:
        existing_keys = set(existing["_key"].dropna())
        return new_df[~new_df["_key"].isin(existing_keys)].reset_index(drop=True)
    # Fall back to date+course+horse
    if {"date", "course", "horse"}.issubset(existing.columns):
        existing_tuples = set(
            zip(existing["date"].astype(str), existing["course"], existing["horse"])
        )
        composite = list(zip(new_df["date"].astype(str), new_df["course"], new_df["horse"]))
        mask = [t not in existing_tuples for t in composite]
        return new_df[mask].reset_index(drop=True)
    return new_df


def _append_and_save(existing: pd.DataFrame, new_df: pd.DataFrame):
    if existing.empty:
        combined = new_df
    else:
        # Align columns
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[[c for c in existing.columns if c in new_df.columns]]
        combined = pd.concat([existing, new_df], ignore_index=True)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(HISTORICAL, index=False)
    print(f"  [save]  all_us_races.parquet: {len(combined):,} rows total")


# ── Date helpers ──────────────────────────────────────────────────────────────

def _date_range(start: date, end: date) -> list[str]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _already_cached(date_str: str) -> bool:
    return (RAW_DIR / f"us_results_{date_str}.json").exists()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch US race results from The Racing API"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date",  type=str, help="Single date YYYY-MM-DD")
    group.add_argument("--start", type=str, help="Start of date range YYYY-MM-DD")
    parser.add_argument("--end",   type=str, default=date.today().strftime("%Y-%m-%d"),
                        help="End of date range YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making API calls or writing files")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Stop after fetching N dates (0 = no limit)")
    parser.add_argument("--skip-cached", action="store_true", default=True,
                        help="Skip dates whose raw JSON is already cached (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if raw JSON already cached")
    args = parser.parse_args()

    # Build date list
    if args.date:
        dates = [args.date]
    else:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").date()
        dates    = _date_range(start_dt, end_dt)

    # Filter already-cached unless --force
    if not args.force:
        pending = [d for d in dates if not _already_cached(d)]
        skipped = len(dates) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-cached date(s) (use --force to refetch)")
        dates = pending

    if not dates:
        print("Nothing to fetch.")
        sys.exit(0)

    if args.batch_size > 0:
        dates = dates[:args.batch_size]

    print(f"\n{'='*60}")
    print(f"  Fetching US race results")
    print(f"  Dates : {dates[0]} → {dates[-1]}  ({len(dates)} calls)")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Load existing data once
    existing_df = pd.DataFrame() if args.dry_run else load_historical()
    _backup()

    # Fetch day by day
    all_new_rows = []
    api_calls = 0

    for date_str in dates:
        races = fetch_results_for_date(date_str, dry_run=args.dry_run)
        if races:
            rows = []
            for race in races:
                rows.extend(map_race_to_rows(race))
            all_new_rows.extend(rows)
            if not _already_cached(date_str):
                api_calls += 1
        time.sleep(0.5)  # polite pacing between calls

    if args.dry_run:
        print(f"\nDry run complete — would have fetched {len(dates)} dates.")
        return

    if not all_new_rows:
        print("\nNo new rows to add.")
        return

    new_df = pd.DataFrame(all_new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")

    unique_new = _deduplicate(existing_df, new_df)
    print(f"\nNew unique rows: {len(unique_new):,}  (from {len(all_new_rows):,} fetched)")

    if len(unique_new) > 0:
        _append_and_save(existing_df, unique_new)

    # Summary
    print(f"\n{'='*60}")
    print(f"  API calls made   : {api_calls}")
    print(f"  New rows added   : {len(unique_new):,}")
    if HISTORICAL.exists():
        df_check = pd.read_parquet(HISTORICAL)
        print(f"  Total rows now   : {len(df_check):,}")
        if "date" in df_check.columns:
            df_check["date"] = pd.to_datetime(df_check["date"])
            print(f"  Date range       : {df_check['date'].min().date()} → {df_check['date'].max().date()}")
        if "course" in df_check.columns:
            print(f"  Unique tracks    : {df_check['course'].nunique()}")
        if "surface" in df_check.columns:
            print(f"  Surfaces         : {df_check['surface'].value_counts().to_dict()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
