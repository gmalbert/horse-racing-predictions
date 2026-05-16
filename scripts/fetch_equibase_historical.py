"""
fetch_equibase_historical.py
────────────────────────────
Batch-scrape Equibase result pages for a date range, one date at a time.

Strategy (anti-block):
  • Results-only mode (no entry pages — they expire after ~48h anyway)
  • Configurable delay between requests (default 3.5s with jitter)
  • Browser session rotated every SESSION_ROTATION_DAYS track-days
  • One output file per date in --out-dir; already-complete dates skipped
  • Graceful Ctrl-C: saves progress and exits cleanly

Usage examples
──────────────
# Last 90 days, major tracks only, default delay:
python scripts/fetch_equibase_historical.py --start 2026-02-14 --end 2026-05-14

# All tracks (slow!), custom delay, custom output dir:
python scripts/fetch_equibase_historical.py \\
    --start 2025-01-01 --end 2025-12-31 \\
    --tracks ALL --delay 4.0 --out-dir data/raw/equibase

# Specific tracks only:
python scripts/fetch_equibase_historical.py \\
    --start 2026-01-01 --end 2026-05-14 \\
    --tracks CD SA AQU KEE GP DEL MTH TAM PIM BEL SAR

Output
──────
data/raw/equibase/YYYY-MM-DD.json   — list of CombinedRecord dicts
data/raw/equibase/_progress.json    — checkpoint: set of completed dates
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from datetime import date, timedelta
from pathlib import Path

# Local import — must be run from repo root or with scripts/ on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from equibase_scraper import EquibaseSession, EquibaseScraper  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Curated track list ────────────────────────────────────────────────────────
# Major US tracks that regularly run graded stakes.  Keeps request count
# manageable; add/remove as needed.
MAJOR_TRACKS = [
    "CD",   # Churchill Downs
    "SA",   # Santa Anita
    "AQU",  # Aqueduct
    "KEE",  # Keeneland
    "GP",   # Gulfstream Park
    "DEL",  # Delaware Park
    "MTH",  # Monmouth Park
    "TAM",  # Tampa Bay Downs
    "TP",   # Turfway Park
    "TUP",  # Turf Paradise
    "HAW",  # Hawthorne
    "PIM",  # Pimlico
    "BEL",  # Belmont Park
    "SAR",  # Saratoga
    "DMR",  # Del Mar
    "CHL",  # Churchill (alternate code)
    "LAD",  # Louisiana Downs
    "FG",   # Fair Grounds
    "OP",   # Oaklawn Park
    "FPX",  # Fairplex Park
]

# How many track-days before rotating the Playwright session
SESSION_ROTATION_DAYS = 20


# ── Date helpers ──────────────────────────────────────────────────────────────
def date_range(start: date, end: date):
    """Yield every date from start to end inclusive, skipping Mondays (lightest card)."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_progress(progress_file: Path) -> set[str]:
    if progress_file.exists():
        try:
            return set(json.loads(progress_file.read_text()))
        except Exception:
            pass
    return set()


def save_progress(progress_file: Path, completed: set[str]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(sorted(completed), indent=2))


# ── Core scrape loop ──────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "_progress.json"
    completed = load_progress(progress_file)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    tracks: list[str] | None = None if args.tracks == ["ALL"] else args.tracks

    all_dates = list(date_range(start, end))
    remaining = [d for d in all_dates if d.isoformat() not in completed]

    log.info(
        "Date range %s → %s  (%d total, %d already done, %d to fetch)",
        start, end, len(all_dates), len(completed), len(remaining),
    )
    if not remaining:
        log.info("Nothing to do — all dates already scraped.")
        return

    # Graceful Ctrl-C handler
    _interrupted = False
    def _sigint(sig, frame):
        nonlocal _interrupted
        log.warning("Interrupted — saving progress and exiting…")
        _interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    session = EquibaseSession(delay=args.delay, jitter=args.jitter)
    scraper = EquibaseScraper(session)
    track_days_this_session = 0

    try:
        for i, race_date in enumerate(remaining):
            if _interrupted:
                break

            out_file = out_dir / f"{race_date.isoformat()}.json"

            # Skip if output file already exists (belt-and-suspenders)
            if out_file.exists():
                completed.add(race_date.isoformat())
                continue

            log.info("── %s (%d/%d) ──", race_date, i + 1, len(remaining))

            try:
                if tracks:
                    # Specific tracks: scrape each and merge
                    records = []
                    for track in tracks:
                        track_records = scraper.get_results(race_date, track)
                        records.extend(track_records)
                else:
                    # All tracks: let the scraper discover them from the index
                    records = scraper.get_results(race_date, track=None)

                # Convert dataclasses to dicts
                from dataclasses import asdict
                data = [asdict(r) for r in records]

                if records:
                    # Only mark as done if we actually got data — allows retry for
                    # dates where pages weren't available yet
                    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    completed.add(race_date.isoformat())
                    save_progress(progress_file, completed)
                    log.info("  Saved %d records → %s", len(records), out_file.name)
                else:
                    log.warning("  0 records for %s — pages may not exist yet (not checkpointed)", race_date)

            except Exception as exc:
                log.error("  Failed for %s: %s — skipping", race_date, exc)

            # Session rotation: close and reopen browser once per N *dates* (not tracks)
            track_days_this_session += 1
            if track_days_this_session >= SESSION_ROTATION_DAYS and not _interrupted:
                log.info("Rotating Playwright session after %d track-days…", track_days_this_session)
                session.close()
                session = EquibaseSession(delay=args.delay, jitter=args.jitter)
                scraper = EquibaseScraper(session)
                track_days_this_session = 0

    finally:
        session.close()
        save_progress(progress_file, completed)
        log.info(
            "Done — %d/%d dates completed.  Progress saved to %s",
            len(completed), len(all_dates), progress_file,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-scrape Equibase historical results with checkpoint/resume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=MAJOR_TRACKS,
        metavar="TRACK",
        help=(
            "Track codes to scrape, e.g. --tracks CD SA KEE.  "
            "Pass ALL to scrape every track listed on the results index (slow)."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.5,
        help="Base delay in seconds between HTTP requests (default 3.5).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1.5,
        help="Max random jitter added to delay (default 1.5).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/equibase",
        help="Directory for per-date JSON files (default: data/raw/equibase).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
