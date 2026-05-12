"""
ingest_us_entries.py — Run TVG scraper and copy results to data/processed/.

This script:
  1. Runs the TVG GraphQL scraper for a given date
  2. Writes raw data to scraping/scrapers/us/output/
  3. Copies the processed CSV to data/processed/us_entries_YYYY-MM-DD.csv
  4. Merges runner data into data/raw/us_racecards_YYYY-MM-DD.json so that
     predict_us_races.py can generate predictions for all TVG-covered tracks.

Usage:
    python scripts/ingest_us_entries.py                    # today
    python scripts/ingest_us_entries.py --date 2026-05-10  # specific date
    python scripts/ingest_us_entries.py --list-tracks      # show available tracks
    python scripts/ingest_us_entries.py --track CD         # single track

Output:
    data/processed/us_entries_YYYY-MM-DD.csv
    data/raw/us_racecards_YYYY-MM-DD.json  (updated with runner lists)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, date, timezone
from pathlib import Path

# Add scraping module to path
REPO_ROOT = Path(__file__).parent.parent
SCRAPERS_PATH = REPO_ROOT / "scraping" / "scrapers" / "us"
sys.path.insert(0, str(SCRAPERS_PATH))

from utils.common import (
    RaceEntry,
    get_logger,
    today_str,
    write_entries_csv,
    OUTPUT_ROOT,
)

logger = get_logger("ingest_us_entries")

DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_RAW = REPO_ROOT / "data" / "raw"


def _entries_to_racecards_json(entries: list[RaceEntry], race_date: str) -> dict:
    """
    Convert a flat list of RaceEntry objects into us_racecards JSON format.

    Groups entries by track+race_number, maps runner fields to the schema
    expected by predict_us_races.py.
    """
    # Group by (track_code, race_number)
    races: dict[tuple, list[RaceEntry]] = {}
    for e in entries:
        if e.scratched:
            continue  # omit scratches from predictions input
        key = (e.track_code, e.race_number)
        races.setdefault(key, []).append(e)

    racecard_list = []
    for (track_code, race_number), runners in sorted(races.items()):
        rep = runners[0]
        runner_dicts = []
        for r in sorted(runners, key=lambda x: int(x.program_number or 0)):
            raw = r.raw_extra or {}
            runner_dicts.append({
                "name":       r.runner_name,
                "horse":      r.runner_name,
                "jockey":     r.jockey,
                "trainer":    r.trainer,
                "age":        raw.get("age", ""),
                "sex":        raw.get("sex", ""),
                "age_sex":    f"{raw.get('age', '')}{raw.get('sex', '')}",
                "weight":     raw.get("weight", 0),
                "lbs":        raw.get("weight", 0),
                "draw":       int(r.program_number) if r.program_number else 0,
                "number":     r.program_number,
                "ml_odds":    r.ml_odds,
                "sire":       raw.get("sire", ""),
                "dam":        raw.get("dam", ""),
                "dam_sire":   raw.get("dam_sire", ""),
                "owner":      raw.get("owner", ""),
                "saddle_color": raw.get("saddle_color", ""),
            })

        racecard_list.append({
            "race":       f"Race {race_number}",
            "race_name":  rep.race_name or "",
            "race_time":  rep.race_time or "",
            "course":     rep.track_name,
            "track":      track_code,
            "surface":    rep.surface,
            "distance":   rep.distance,
            "race_class": rep.race_class,
            "purse":      rep.purse,
            "breed":      rep.breed,
            "source":     "tvg",
            "source_url": rep.source_url,
            "date":       race_date,
            "runners":    runner_dicts,
        })

    return {
        "date":          race_date,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source":        "tvg_graphql",
        "racecards":     racecard_list,
        "source_counts": {
            "tvg_races": len(racecard_list),
            "total_races": len(racecard_list),
        },
    }


def merge_into_us_racecards(entries: list[RaceEntry], race_date: str) -> Path:
    """
    Write TVG entries into data/raw/us_racecards_YYYY-MM-DD.json.

    If the file already exists, races that already have runners are kept;
    races with 0 runners get populated from TVG.  Races from TVG that
    aren't in the existing file are appended.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    rc_path = DATA_RAW / f"us_racecards_{race_date}.json"

    # Build TVG racecards indexed by (track_code, race_number)
    tvg_rc = _entries_to_racecards_json(entries, race_date)
    tvg_by_key: dict[tuple, dict] = {}
    for rc in tvg_rc["racecards"]:
        key = (rc["track"], rc["race"])
        tvg_by_key[key] = rc

    if rc_path.exists():
        try:
            existing = json.loads(rc_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"racecards": []}

        existing_rcs = existing.get("racecards", [])
        seen_keys: set[tuple] = set()
        updated = 0

        for rc in existing_rcs:
            key = (rc.get("track", rc.get("course", "")), rc.get("race", ""))
            seen_keys.add(key)
            # Fill in runners from TVG if currently empty
            if not rc.get("runners") and key in tvg_by_key:
                rc["runners"] = tvg_by_key[key]["runners"]
                # Also fill race-level fields from TVG if blank
                for field in ("surface", "distance", "race_class", "purse", "breed"):
                    if not rc.get(field) and tvg_by_key[key].get(field):
                        rc[field] = tvg_by_key[key][field]
                updated += 1

        # Append TVG races not already in the file
        added = 0
        for key, rc in tvg_by_key.items():
            if key not in seen_keys:
                existing_rcs.append(rc)
                added += 1

        existing["racecards"] = existing_rcs
        # Update counts
        existing["source_counts"] = existing.get("source_counts", {})
        existing["source_counts"]["tvg_races"] = len(tvg_by_key)
        existing["source_counts"]["total_races"] = len(existing_rcs)
        existing["tvg_merged_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        rc_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info(
            "us_racecards: updated %d existing races + added %d new TVG races → %d total",
            updated, added, len(existing_rcs)
        )
    else:
        # No existing file — write TVG data directly
        rc_path.write_text(json.dumps(tvg_rc, indent=2), encoding="utf-8")
        logger.info(
            "us_racecards: created with %d TVG races", len(tvg_rc["racecards"])
        )

    return rc_path


def run_tvg_scraper(race_date: str) -> list[RaceEntry]:
    """Run the TVG GraphQL scraper and return entries."""
    from sources.tvg_client import fetch_all_us_tracks

    logger.info("Fetching US race entries from TVG for %s...", race_date)
    entries, tracks_attempted, tracks_with_data = fetch_all_us_tracks(race_date)
    logger.info(
        "TVG scraper: %d entries from %d/%d US tracks",
        len(entries), tracks_with_data, tracks_attempted
    )
    return entries


def fetch_single_track(track_code: str, race_date: str) -> list[RaceEntry]:
    """Fetch entries for a single track."""
    from sources.tvg_client import fetch_track
    return fetch_track(track_code, race_date)


def list_us_tracks() -> None:
    """List all US tracks available on TVG."""
    from sources.tvg_client import get_us_track_codes, _build_gql_session
    session = _build_gql_session()
    tracks = get_us_track_codes(session)
    if tracks:
        print(f"US tracks with open races ({len(tracks)}):")
        for code, name in sorted(tracks):
            print(f"  {code:6s}  {name}")
    else:
        print("No US tracks found with open races right now.")
        print("(This is normal outside of US racing hours)")


def entries_to_csv_row(entry: RaceEntry) -> dict:
    """Convert a RaceEntry to a flat CSV row."""
    row = entry.to_dict()
    # Add extra fields from raw_extra
    raw = entry.raw_extra or {}
    row["sire"] = raw.get("sire", "")
    row["dam"] = raw.get("dam", "")
    row["dam_sire"] = raw.get("dam_sire", "")
    row["age"] = raw.get("age", "")
    row["sex"] = raw.get("sex", "")
    row["owner"] = raw.get("owner", "")
    row["weight"] = raw.get("weight", "")
    row["current_odds"] = raw.get("current_odds", "")
    row["current_odds_decimal"] = raw.get("current_odds_decimal", "")
    row["ml_odds_decimal"] = raw.get("ml_odds_decimal", "")
    row["race_id"] = raw.get("race_id", "")
    row["post_time_utc"] = raw.get("post_time_utc", "")
    row["num_runners"] = raw.get("num_runners", "")
    return row


def copy_to_data_processed(entries: list[RaceEntry], race_date: str) -> Path:
    """Write entries to data/processed/us_entries_YYYY-MM-DD.csv."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / f"us_entries_{race_date}.csv"

    if not entries:
        logger.warning("No entries to write for %s", race_date)
        return out_path

    rows = [entries_to_csv_row(e) for e in entries]
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Written %d entries to %s", len(entries), out_path)
    return out_path


def print_summary(entries: list[RaceEntry], race_date: str) -> None:
    """Print a human-readable summary of fetched entries."""
    if not entries:
        print(f"\nNo entries found for {race_date}")
        return

    # Group by track
    by_track: dict[str, list[RaceEntry]] = {}
    for e in entries:
        by_track.setdefault(e.track_code, []).append(e)

    print(f"\n{'='*60}")
    print(f"US Race Entries — {race_date}")
    print(f"{'='*60}")
    print(f"Total entries: {len(entries)} across {len(by_track)} tracks\n")

    for track_code in sorted(by_track.keys()):
        track_entries = by_track[track_code]
        track_name = track_entries[0].track_name
        races = sorted(set(e.race_number for e in track_entries))
        scratch_count = sum(1 for e in track_entries if e.scratched)

        print(f"  {track_code:6s} {track_name}")
        print(f"         Races: {len(races)}, Entries: {len(track_entries)}, "
              f"Scratches: {scratch_count}")

        # Show first race sample
        r1 = sorted(track_entries, key=lambda x: (x.race_number, int(x.program_number or 0)))
        if r1:
            sample = r1[0]
            print(f"         Sample: R{sample.race_number} #{sample.program_number} "
                  f"{sample.runner_name} (ML: {sample.ml_odds})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch US race entries from TVG and ingest to data/processed/"
    )
    parser.add_argument("--date", default=today_str(),
                        help="Race date YYYY-MM-DD (default: today)")
    parser.add_argument("--track", help="Fetch specific track only (e.g. CD, SA, LRL)")
    parser.add_argument("--list-tracks", action="store_true",
                        help="List available US tracks and exit")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy to data/processed/ (just show summary)")
    args = parser.parse_args()

    if args.list_tracks:
        list_us_tracks()
        return

    race_date = args.date
    logger.info("Ingesting US race entries for %s", race_date)

    if args.track:
        entries = fetch_single_track(args.track, race_date)
    else:
        entries = run_tvg_scraper(race_date)

    print_summary(entries, race_date)

    if not args.no_copy and entries:
        out_path = copy_to_data_processed(entries, race_date)
        print(f"\nCSV output: {out_path}")
        rc_path = merge_into_us_racecards(entries, race_date)
        print(f"Racecards:  {rc_path}")
    elif not entries:
        sys.exit(1)


if __name__ == "__main__":
    main()
