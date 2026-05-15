"""
fetch_equibase_entries.py — fetch Equibase daily entries for upcoming dates.

Uses the existing equibase_client.py scraper (already in scraping/).
Output: data/raw/equibase_entries_YYYY-MM-DD.json
        data/processed/equibase_entries_YYYY-MM-DD.csv

Usage:
    python scripts/fetch_equibase_entries.py --date 2026-05-12
    python scripts/fetch_equibase_entries.py --date 2026-05-12 --tracks SA GP KEE
    python scripts/fetch_equibase_entries.py --date 2026-05-12 --force
    python scripts/fetch_equibase_entries.py --days 3          # today + next 2 days
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make scraper importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scraping" / "scrapers" / "us" / "sources"))
sys.path.insert(0, str(REPO_ROOT / "scraping" / "scrapers" / "us"))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] fetch_equibase: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fetch_equibase")

RAW_DIR  = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Default tracks to attempt each day — curated to those most likely to
# have entries available on Equibase's static page.
DEFAULT_TRACKS = [
    # Major / Tier 1
    "CD",   # Churchill Downs
    "BEL",  # Belmont Park
    "SAR",  # Saratoga
    "KEE",  # Keeneland
    "SA",   # Santa Anita
    "DMR",  # Del Mar
    "GP",   # Gulfstream Park
    "OP",   # Oaklawn Park
    "FG",   # Fair Grounds
    "MTH",  # Monmouth Park
    # Tier 1 regional
    "PRX",  # Parx
    "PEN",  # Penn National
    "MNR",  # Mountaineer
    "CT",   # Charles Town
    "TAM",  # Tampa Bay Downs
    "HOU",  # Sam Houston
    "LS",   # Lone Star Park
    "ELP",  # Ellis Park
    "PID",  # Presque Isle Downs
]


def fetch_date(race_date: str, track_codes: list[str], force: bool = False) -> list[dict]:
    """
    Fetch Equibase entries for all specified tracks on race_date.
    Returns a list of entry dicts (serialisable).
    """
    cache_path = RAW_DIR / f"equibase_entries_{race_date}.json"

    if cache_path.exists() and not force:
        logger.info("Loaded cached Equibase entries for %s from %s", race_date, cache_path)
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", [])

    try:
        from equibase_client import fetch_track, TRACKS
        from utils.common import build_session
    except ImportError as exc:
        logger.error("Could not import equibase_client: %s", exc)
        logger.error("Ensure scraping/scrapers/us/sources/ is on the path.")
        return []

    # Use a short per-request timeout so Imperva hangs don't stall forever.
    session = build_session(retries=1, backoff_factor=0.5, timeout=15)

    all_entries: list[dict] = []
    ok_tracks: list[str] = []
    failed_tracks: list[str] = []
    skipped_tracks: list[str] = []

    valid_codes = [c for c in track_codes if c in TRACKS]
    skipped_tracks = [c for c in track_codes if c not in TRACKS]

    # Per-track hard deadline: 20s is enough for a real response;
    # Imperva hangs or retry storms won't block the whole run.
    PER_TRACK_TIMEOUT = 20

    def _fetch_one(code: str):
        try:
            return code, fetch_track(session, code, race_date)
        except Exception as exc:
            return code, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_one, code): code for code in valid_codes}
        for future in concurrent.futures.as_completed(futures, timeout=None):
            code = futures[future]
            try:
                _, result = future.result(timeout=PER_TRACK_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.warning("  %s: timed out after %ds — Imperva likely blocking", code, PER_TRACK_TIMEOUT)
                failed_tracks.append(code)
                continue
            except Exception as exc:
                logger.warning("  %s: fetch failed — %s", code, exc)
                failed_tracks.append(code)
                continue

            if isinstance(result, Exception):
                logger.warning("  %s: fetch error — %s", code, result)
                failed_tracks.append(code)
            elif result:
                all_entries.extend(e.to_dict() for e in result)
                ok_tracks.append(code)
                logger.info("  %s: %d entries across %d races",
                            code, len(result),
                            len({e.race_number for e in result}))
            else:
                logger.info("  %s: no entries (may not be racing today)", code)

    logger.info(
        "%s: %d entries from %d/%d tracks attempted. Failed/timeout: %s. Skipped (unknown): %s",
        race_date, len(all_entries), len(ok_tracks), len(valid_codes),
        failed_tracks or "none", skipped_tracks or "none",
    )

    if not all_entries:
        logger.warning(
            "No Equibase entries for %s. "
            "Equibase may use Imperva bot-protection for this date — "
            "try again later or use Playwright (pip install playwright && playwright install).",
            race_date,
        )

    # Save JSON cache
    payload = {
        "race_date": race_date,
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tracks_attempted": track_codes,
        "tracks_ok": ok_tracks,
        "tracks_failed": failed_tracks,
        "count": len(all_entries),
        "entries": all_entries,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON cache to %s", cache_path)

    # Save CSV
    if all_entries:
        csv_path = PROC_DIR / f"equibase_entries_{race_date}.csv"
        _write_csv(all_entries, csv_path)

    return all_entries


def _write_csv(entries: list[dict], path: Path) -> None:
    if not entries:
        return
    fieldnames = [
        "race_date", "track_code", "track_name", "race_number", "race_time",
        "race_name", "race_class", "surface", "distance", "purse",
        "program_number", "runner_name", "jockey", "trainer", "ml_odds",
        "scratched", "breed", "source_name",
    ]
    present = [f for f in fieldnames if f in entries[0]]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=present, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    logger.info("CSV saved to %s (%d rows)", path, len(entries))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Equibase daily entries")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--date", type=str,
        help="Specific date YYYY-MM-DD (default: today)",
    )
    group.add_argument(
        "--days", type=int, default=None,
        help="Fetch today + next N-1 days (e.g. --days 3)",
    )
    parser.add_argument(
        "--tracks", nargs="+", default=None,
        help="Track codes to fetch (default: curated list of ~19 tracks)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if cached",
    )
    args = parser.parse_args()

    track_codes = [t.upper() for t in args.tracks] if args.tracks else DEFAULT_TRACKS

    if args.days:
        today = datetime.now()
        dates = [
            (today + timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range(args.days)
        ]
    elif args.date:
        dates = [args.date]
    else:
        dates = [datetime.now().strftime("%Y-%m-%d")]

    for race_date in dates:
        logger.info("=== Fetching Equibase entries for %s ===", race_date)
        entries = fetch_date(race_date, track_codes, force=args.force)
        if entries:
            races = len({(e["track_code"], e["race_number"]) for e in entries
                         if "track_code" in e and "race_number" in e})
            tracks = len({e["track_code"] for e in entries if "track_code" in e})
            logger.info("Result: %d entries · %d races · %d tracks", len(entries), races, tracks)
        else:
            logger.warning("No entries returned for %s", race_date)


if __name__ == "__main__":
    main()
