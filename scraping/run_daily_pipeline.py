"""
run_daily_pipeline.py — master orchestrator for all US horse racing scrapers.

Runs all source clients in sequence, collects a summary, and writes a
per-run report JSON.  Designed to be called from a cron job or scheduler.

Order of execution:
  1. Equibase (Thoroughbred Tier 1 + 2 + some Tier 3)
  2. USTA JSON (all harness tracks)
  3. CDI JSON (Churchill Downs, Fair Grounds)
  4. NYRA JSON (Belmont, Saratoga)  — skips AQU
  5. Quarter Horse client (LAD, RUI, ZIA, SUN, DEL, EVD, REM)
  6. Track site static HTML (Gulfstream, Oaklawn, Monmouth, etc.)
  7. Playwright dynamic (Keeneland, Santa Anita, Del Mar, Laurel, Pimlico, Tampa)

Usage:
    python run_daily_pipeline.py                    # today
    python run_daily_pipeline.py --date 2026-05-10  # specific date
    python run_daily_pipeline.py --sources equibase usta   # subset
    python run_daily_pipeline.py --dry-run          # print plan only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import (
    RaceEntry,
    append_entries_csv,
    build_session,
    get_logger,
    today_str,
    write_entries_csv,
    OUTPUT_ROOT,
)

logger = get_logger("pipeline")

# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

@dataclass
class SourceResult:
    source: str
    tracks_attempted: int = 0
    tracks_with_data: int = 0
    entries_collected: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.tracks_attempted == 0:
            return 0.0
        return self.tracks_with_data / self.tracks_attempted


@dataclass
class PipelineReport:
    run_date: str
    race_date: str
    started_at: str
    finished_at: str = ""
    total_entries: int = 0
    sources: list[SourceResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Source runners
# ---------------------------------------------------------------------------

def run_equibase(session, race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.equibase_client import fetch_track, TRACKS

    result = SourceResult(source="equibase", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(session, code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


def run_usta(session, race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.usta_client import fetch_track, TRACKS

    result = SourceResult(source="usta", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(session, code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


def run_cdi(session, race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.cdi_client import fetch_track, TRACKS

    result = SourceResult(source="cdi", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(session, code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


def run_nyra(session, race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.nyra_client import fetch_track, TRACKS

    result = SourceResult(source="nyra", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(session, code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


def run_quarter_horse(session, race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.quarter_horse_client import fetch_track, TRACKS

    result = SourceResult(source="quarter_horse", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(session, code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


def run_tracksite(session, race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.tracksite_static_client import fetch_track, TRACKS

    result = SourceResult(source="tracksite_static", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(session, code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


def run_playwright(race_date: str) -> tuple[list[RaceEntry], SourceResult]:
    from sources.playwright_dynamic_client import fetch_track, TRACKS

    result = SourceResult(source="playwright_dynamic", tracks_attempted=len(TRACKS))
    entries: list[RaceEntry] = []
    t0 = time.time()

    for code in TRACKS:
        try:
            track_entries = fetch_track(code, race_date)
            entries.extend(track_entries)
            if track_entries:
                result.tracks_with_data += 1
        except Exception as exc:
            result.errors.append(f"{code}: {exc}")

    result.entries_collected = len(entries)
    result.elapsed_sec = round(time.time() - t0, 1)
    return entries, result


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

ALL_SOURCES = ["equibase", "usta", "cdi", "nyra", "quarter_horse", "tracksite", "playwright"]

SKIP_IF_DUPLICATE: dict[str, list[str]] = {
    # If equibase already got these tracks, tracksite and playwright skip them
    # (deduplication happens at the master CSV level anyway via race_date+track+runner key)
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="US horse racing daily pipeline")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--sources", nargs="*", default=ALL_SOURCES,
                        choices=ALL_SOURCES, help="Which source modules to run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    race_date = args.date
    sources = args.sources

    if args.dry_run:
        print(f"Pipeline plan for {race_date}:")
        for s in sources:
            print(f"  - {s}")
        return

    logger.info("=" * 60)
    logger.info("US RACING PIPELINE — %s", race_date)
    logger.info("Sources: %s", ", ".join(sources))
    logger.info("=" * 60)

    started_at = datetime.utcnow().isoformat()
    session = build_session()
    all_entries: list[RaceEntry] = []
    report = PipelineReport(
        run_date=today_str(),
        race_date=race_date,
        started_at=started_at,
    )

    runners: dict[str, Callable] = {
        "equibase":     lambda: run_equibase(session, race_date),
        "usta":         lambda: run_usta(session, race_date),
        "cdi":          lambda: run_cdi(session, race_date),
        "nyra":         lambda: run_nyra(session, race_date),
        "quarter_horse":lambda: run_quarter_horse(session, race_date),
        "tracksite":    lambda: run_tracksite(session, race_date),
        "playwright":   lambda: (run_playwright(race_date)),
    }

    for source_name in sources:
        runner = runners.get(source_name)
        if not runner:
            logger.warning("Unknown source: %s — skipping", source_name)
            continue

        logger.info("--- Running source: %s ---", source_name)
        try:
            result_tuple = runner()
            if isinstance(result_tuple, tuple):
                src_entries, src_result = result_tuple
            else:
                src_entries, src_result = result_tuple, SourceResult(source=source_name)

            all_entries.extend(src_entries)
            report.sources.append(src_result)
            logger.info(
                "[%s] done: %d entries from %d/%d tracks in %.1fs",
                source_name,
                src_result.entries_collected,
                src_result.tracks_with_data,
                src_result.tracks_attempted,
                src_result.elapsed_sec,
            )
            if src_result.errors:
                for err in src_result.errors[:5]:
                    logger.warning("[%s] error: %s", source_name, err)

        except Exception as exc:
            logger.error("Source %s crashed: %s", source_name, exc)
            report.sources.append(SourceResult(source=source_name, errors=[str(exc)]))

    # --- Write combined outputs ---
    report.total_entries = len(all_entries)
    report.finished_at = datetime.utcnow().isoformat()

    if all_entries:
        # Per-run CSV
        combined_csv = write_entries_csv(all_entries, "us_all_entries", race_date)
        logger.info("Combined CSV: %s (%d entries)", combined_csv, len(all_entries))

        # Rolling master (all dates, all tracks)
        master = OUTPUT_ROOT / "processed" / "us_all_canonical_master.csv"
        append_entries_csv(all_entries, master)
        logger.info("Master CSV updated: %s", master)
    else:
        logger.warning("No entries collected for %s across all sources.", race_date)

    # --- Write run report ---
    report_path = OUTPUT_ROOT / "reports" / f"pipeline_report_{race_date}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Run report: %s", report_path)

    # --- Print summary table ---
    logger.info("")
    logger.info("PIPELINE SUMMARY — %s", race_date)
    logger.info("%-20s  %8s  %8s  %8s  %8s", "Source", "Tracks✓", "Tracks", "Entries", "Secs")
    logger.info("-" * 65)
    for sr in report.sources:
        logger.info(
            "%-20s  %8d  %8d  %8d  %8.1f",
            sr.source,
            sr.tracks_with_data,
            sr.tracks_attempted,
            sr.entries_collected,
            sr.elapsed_sec,
        )
    logger.info("-" * 65)
    logger.info("%-20s  %8s  %8s  %8d", "TOTAL", "", "", report.total_entries)

    # Exit with non-zero if no data at all
    if report.total_entries == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
