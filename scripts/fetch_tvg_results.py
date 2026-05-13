"""
fetch_tvg_results.py — Fetch historical US race results from TVG's public GraphQL API
                       and accumulate data/processed/us_races_cleaned.parquet.

TVG (FanDuel Racing) exposes a public GraphQL endpoint (no auth required) with a
`pastRaces` query that returns full finishing positions, payoffs, and race details
for all tracks available on the TVG platform — covering ~25-35 US tracks per day.

This is the primary free alternative to The Racing API's /results endpoint (which
requires a Standard plan).

Usage:
    python scripts/fetch_tvg_results.py --days 90        # last 90 days
    python scripts/fetch_tvg_results.py --date 2026-05-09
    python scripts/fetch_tvg_results.py --start 2026-01-01 --end 2026-05-11
    python scripts/fetch_tvg_results.py --dry-run        # print stats, no write
    python scripts/fetch_tvg_results.py --days 7 --tracks CD KEE SA GP

Rate limiting:  TVG GraphQL has no documented rate limit, but we pause 0.5 s
                between per-track calls as a courtesy.  For large backfills use
                --delay 1.0 to be safer.

Output:
    data/raw/tvg_results_YYYY-MM-DD.json  — raw response (cached; re-runs free)
    data/processed/us_races_cleaned.parquet — accumulated training dataset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"
PROC_DIR  = REPO_ROOT / "data" / "processed"
US_PARQUET = PROC_DIR / "us_races_cleaned.parquet"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] fetch_tvg_results: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fetch_tvg_results")

# ---------------------------------------------------------------------------
# TVG GraphQL constants
# ---------------------------------------------------------------------------

GQL_URL = "https://api.tvg.com/cosmo/v1/graphql"
GQL_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.tvg.com",
    "Referer": "https://www.tvg.com/",
    "Accept": "application/json",
}
WAGER_PROFILE = "PORT-Generic"

# pastRaces query — one call per track per date
QUERY_PAST_RACES = """
query getPastRaces($trackCode: String!, $date: String!, $profile: String!) {
  pastRaces(profile: $profile, trackCode: $trackCode, date: $date) {
    id
    number
    date
    description
    distance { value }
    surface { name }
    type { name }
    raceClass { name }
    purse
    numRunners
    claimingPrice
    track { code name location { city state country } }
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
    }
  }
}
"""

# pastTracks — one call per date, gives us the track list
QUERY_PAST_TRACKS = """
query getPastTracks($profile: String!, $date: String!) {
  pastTracks(profile: $profile, date: $date) {
    code
    name
    numberOfRaces
    location { country }
  }
}
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _gql(query: str, variables: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.post(
                GQL_URL,
                headers=GQL_HEADERS,
                json={"query": query, "variables": variables},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                logger.warning("GraphQL errors: %s", data["errors"])
                # Partial data may still be present alongside errors
                return data.get("data") or {}
            return data.get("data", {})
        except requests.RequestException as exc:
            if attempt == retries - 1:
                logger.error("Request failed after %d attempts: %s", retries, exc)
                return {}
            time.sleep(2 ** attempt)
    return {}


def _get_us_tracks_for_date(date_str: str) -> list[dict]:
    """Return list of US track dicts {code, name} for a given date."""
    data = _gql(QUERY_PAST_TRACKS, {"profile": WAGER_PROFILE, "date": date_str})
    tracks = data.get("pastTracks") or []
    return [t for t in tracks if (t.get("location") or {}).get("country") == "USA"]


def _fetch_track_results(track_code: str, date_str: str) -> list[dict]:
    """Fetch all races with results for one track on one date."""
    data = _gql(QUERY_PAST_RACES, {
        "profile": WAGER_PROFILE,
        "trackCode": track_code,
        "date": date_str,
    })
    return data.get("pastRaces") or []


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(date_str: str) -> Path:
    return RAW_DIR / f"tvg_results_{date_str}.json"


def _load_or_fetch(date_str: str, force: bool = False, delay: float = 0.5) -> list[dict]:
    """
    Return list of race dicts for all US tracks on date_str.
    Uses per-date cache in data/raw/; skips network if already cached.
    """
    cache = _cache_path(date_str)
    if cache.exists() and not force:
        races = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("[cache] %s: %d races", date_str, len(races))
        return races

    logger.info("[fetch] %s: discovering US tracks …", date_str)
    us_tracks = _get_us_tracks_for_date(date_str)
    logger.info("[fetch] %s: %d US tracks found", date_str, len(us_tracks))

    all_races: list[dict] = []
    for tr in us_tracks:
        code = tr["code"]
        races = _fetch_track_results(code, date_str)
        races_with_results = [r for r in races if (r.get("results") or {}).get("allRunners")]
        logger.info(
            "  %s (%s): %d races, %d with results",
            code, tr.get("name", ""), len(races), len(races_with_results),
        )
        all_races.extend(races_with_results)
        time.sleep(delay)

    # Cache to disk
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(all_races, ensure_ascii=False), encoding="utf-8")
    return all_races


# ---------------------------------------------------------------------------
# Schema mapping  →  us_races_cleaned rows
# ---------------------------------------------------------------------------

def _decimal_odds(odds_obj: dict | None) -> float | None:
    if not odds_obj:
        return None
    num = odds_obj.get("numerator")
    den = odds_obj.get("denominator")
    if den and den != 0:
        return round(num / den + 1, 3)
    return None


def races_to_rows(races: list[dict]) -> list[dict]:
    """
    Flatten a list of TVG pastRace objects into runner-level rows
    compatible with the us_races_cleaned.parquet schema expected by
    scripts/train_us_model.py.
    """
    rows: list[dict] = []
    for race in races:
        track = race.get("track") or {}
        loc   = track.get("location") or {}

        race_id    = race.get("id", "")
        date_str   = race.get("date", "")
        course     = track.get("name") or track.get("code") or ""
        track_code = track.get("code") or ""
        surface    = (race.get("surface") or {}).get("name") or "Dirt"
        distance   = (race.get("distance") or {}).get("value") or ""
        race_class = (race.get("raceClass") or {}).get("name") or ""
        race_type  = (race.get("type") or {}).get("name") or ""
        race_name  = race.get("description") or ""
        purse      = race.get("purse")
        field_size = int(race.get("numRunners") or 0)
        claim_px   = race.get("claimingPrice")
        winning_time = (race.get("results") or {}).get("winningTime")

        all_runners = (race.get("results") or {}).get("allRunners") or []

        for runner in all_runners:
            if runner.get("scratched"):
                continue

            pos_raw = runner.get("finishPosition")
            status  = runner.get("finishStatus") or ""

            # Skip runners with no finish info and no DNF/DQ status
            if pos_raw is None and not status:
                continue

            try:
                position = int(pos_raw) if pos_raw is not None else None
            except (TypeError, ValueError):
                position = None

            row = {
                # Core fields required by train_us_model.py
                "horse":       runner.get("runnerName") or "",
                "date":        date_str,
                "course":      course,
                "track_code":  track_code,
                "surface":     surface,
                "distance":    distance,
                "race_class":  race_class,
                "race_type":   race_type,
                "race_name":   race_name,
                "position":    str(position) if position is not None else status or None,
                "finish_status": status,
                # Enrichment
                "race_id":      race_id,
                "race_number":  race.get("number"),
                "field_size":   field_size or len(all_runners),
                "purse":        purse,
                "claiming_price": claim_px,
                "winning_time": winning_time,
                "draw":         runner.get("runnerNumber"),
                "win_payoff":   runner.get("winPayoff"),
                "place_payoff": runner.get("placePayoff"),
                "show_payoff":  runner.get("showPayoff"),
                "is_favorite":  runner.get("favorite"),
                "dec":          _decimal_odds(runner.get("currentOdds")),
                "city":         loc.get("city"),
                "state":        loc.get("state"),
                "country":      loc.get("country"),
                "source":       "tvg",
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_existing() -> pd.DataFrame:
    if not US_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(US_PARQUET)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    max_date = df["date"].max()
    logger.info("[load] us_races_cleaned.parquet: %d rows, up to %s", len(df), max_date.date() if pd.notna(max_date) else "?")
    return df


def deduplicate(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing.empty or new_df.empty:
        return new_df
    # Key: race_id + horse name (TVG race IDs are globally unique)
    if "race_id" in existing.columns:
        existing_keys = set(
            (existing["race_id"].astype(str) + "_" + existing["horse"].astype(str)).dropna()
        )
        composite = new_df["race_id"].astype(str) + "_" + new_df["horse"].astype(str)
        return new_df[~composite.isin(existing_keys)].reset_index(drop=True)
    return new_df


def append_and_save(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = new_df
    else:
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df.reindex(columns=existing.columns)
        combined = pd.concat([existing, new_df], ignore_index=True)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(US_PARQUET, index=False)
    logger.info("[save] us_races_cleaned.parquet: %d rows total", len(combined))
    return combined


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
        description=(
            "Fetch historical US race results from TVG GraphQL API "
            "and update data/processed/us_races_cleaned.parquet"
        )
    )
    parser.add_argument("--date",    help="Single date YYYY-MM-DD")
    parser.add_argument("--start",   help="Start of date range YYYY-MM-DD")
    parser.add_argument("--end",     help="End of date range YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--days",    type=int, help="Number of days back from today to fetch")
    parser.add_argument("--tracks",  nargs="+", help="Limit to specific track codes e.g. CD KEE SA")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="Seconds to pause between track API calls (default: 0.5)")
    parser.add_argument("--force",   action="store_true", help="Re-fetch even if cache exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and print stats but do not write parquet")
    args = parser.parse_args()

    print("=" * 65)
    print("FETCH TVG HISTORICAL RESULTS  (GraphQL — no auth required)")
    print("=" * 65)

    existing = load_existing()
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    if args.date:
        dates = [args.date]
    elif args.days:
        end_d   = date.today() - timedelta(days=1)
        start_d = end_d - timedelta(days=args.days - 1)
        dates   = list(_date_range(start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")))
    else:
        start = args.start or _auto_start(existing)
        end   = args.end   or yesterday
        dates = list(_date_range(start, end))

    if not dates:
        print("No dates to fetch.")
        return

    print(f"\nFetching {len(dates)} date(s): {dates[0]} to {dates[-1]}")
    if args.dry_run:
        print("[dry-run] No data will be written.\n")

    all_rows: list[dict] = []
    total_races = 0

    for d in dates:
        raw_races = _load_or_fetch(d, force=args.force, delay=args.delay)

        # Optional track filter
        if args.tracks:
            track_set = {t.upper() for t in args.tracks}
            raw_races = [r for r in raw_races if (r.get("track") or {}).get("code", "").upper() in track_set]

        rows = races_to_rows(raw_races)
        total_races += len(raw_races)
        all_rows.extend(rows)

    print(f"\nTotal races: {total_races:,}  →  runner rows: {len(all_rows):,}")

    if not all_rows:
        print("Nothing to save.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")

    deduped = deduplicate(existing, new_df)
    print(f"New rows after dedup: {len(deduped):,}")

    if deduped.empty:
        print("All rows already present — nothing to append.")
        return

    if args.dry_run:
        winners = deduped[deduped["position"].isin(["1", 1])].shape[0]
        tracks  = deduped["track_code"].nunique() if "track_code" in deduped else "?"
        print(f"[dry-run] Would write {len(deduped):,} rows ({winners} winners) "
              f"across {tracks} tracks. Exiting without writing.")
        print("\nSample:")
        cols = ["date", "track_code", "course", "race_number", "horse", "position"]
        print(deduped[[c for c in cols if c in deduped]].head(10).to_string(index=False))
        return

    append_and_save(existing, deduped)
    print(f"\nDone.  Run  python scripts/train_us_model.py --min-rows 1000  to train.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
