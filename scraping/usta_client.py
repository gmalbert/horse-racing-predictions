"""
usta_client.py — pull harness racing entries from the USTA (United States
Trotting Association) JSON endpoints.

USTA maintains a public-facing data API used by their track member sites.
The pattern is:
  https://www.ustrotting.com/api/racecards/<date>/<track_code>/entries

Covers (harness):
  Tier 1:  Northfield Park (NP), Scioto Downs (SD), Rosecroft (RC),
           Running Aces (RUN), Plainridge Park (PLN), Cal Expo (CAL)
  Tier 2:  Meadowlands (M), Yonkers (YO), Hoosier Park (HP)
  JSON tier: Pocono Downs (PCD), Harrah's Philadelphia (PHL),
             Tioga Downs (TGA), Vernon Downs (VD), Freehold Raceway (FH),
             Northville Downs (ND)

Usage:
    python usta_client.py --date 2026-05-10
    python usta_client.py --date 2026-05-10 --tracks M YO HP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    RaceEntry,
    append_entries_csv,
    build_session,
    get_logger,
    polite_get_json,
    today_str,
    write_entries_csv,
    write_raw,
    OUTPUT_ROOT,
)

logger = get_logger("usta")

# ---------------------------------------------------------------------------
# Track registry
# ---------------------------------------------------------------------------

TRACKS: dict[str, tuple[str]] = {
    # code: (full_name,)
    "M":   ("Meadowlands Racetrack",),
    "YO":  ("Yonkers Raceway",),
    "HP":  ("Hoosier Park",),
    "NP":  ("Northfield Park",),
    "SD":  ("Scioto Downs",),
    "RC":  ("Rosecroft Raceway",),
    "RUN": ("Running Aces Harness Park",),
    "PLN": ("Plainridge Park Casino",),
    "CAL": ("Cal Expo",),
    "PCD": ("Pocono Downs",),
    "PHL": ("Harrah's Philadelphia",),
    "TGA": ("Tioga Downs",),
    "VD":  ("Vernon Downs",),
    "FH":  ("Freehold Raceway",),
    "ND":  ("Northville Downs",),
    "DD":  ("Dover Downs",),
}

# Primary USTA endpoint (JSON)
USTA_BASE = "https://www.ustrotting.com/api/racecards/{date}/{track_code}/entries"

# Fallback: USTA race program page (HTML) used only if JSON fails
USTA_HTML_BASE = "https://www.ustrotting.com/race-program/{track_code}/{date}"

SOURCE_NAME = "usta"


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_usta_json(data: dict | list, track_code: str, race_date: str) -> list[RaceEntry]:
    """Map USTA JSON payload to canonical RaceEntry list."""
    track_name = TRACKS.get(track_code, (track_code,))[0]
    entries: list[RaceEntry] = []

    # USTA JSON shape (typical):
    # { "races": [ { "raceNumber": 1, "postTime": "...", "distance": "...",
    #                "purse": ..., "entries": [ { "programNumber": ...,
    #                "horseName": ..., "driver": ..., "trainer": ...,
    #                "morningLineOdds": ..., "scratched": false }, ... ] } ] }

    races = data if isinstance(data, list) else data.get("races", [])

    for race in races:
        race_num = int(race.get("raceNumber", race.get("race_number", 0)))
        race_time = str(race.get("postTime", race.get("post_time", "")))
        race_name = race.get("raceName", race.get("race_name", ""))
        race_class = race.get("raceClass", race.get("class", ""))
        distance = str(race.get("distance", ""))
        purse = str(race.get("purse", ""))

        # Surface is always 'Harness' (standardbred on a half-mile oval)
        runners = race.get("entries", race.get("runners", []))

        for runner in runners:
            scratched = bool(runner.get("scratched", runner.get("isScratched", False)))
            ml_odds_raw = runner.get("morningLineOdds", runner.get("mlOdds", ""))
            entries.append(RaceEntry(
                track_code=track_code,
                track_name=track_name,
                race_date=race_date,
                race_number=race_num,
                race_time=race_time,
                race_name=race_name,
                race_class=race_class,
                surface="Harness",
                distance=distance,
                purse=purse,
                program_number=str(runner.get("programNumber", runner.get("program", ""))),
                runner_name=runner.get("horseName", runner.get("horse_name", runner.get("name", ""))),
                jockey=runner.get("driver", runner.get("driverName", "")),  # driver for harness
                trainer=runner.get("trainer", runner.get("trainerName", "")),
                ml_odds=str(ml_odds_raw),
                scratched=scratched,
                breed="Harness",
                source_name=SOURCE_NAME,
                source_url="",
                raw_extra=runner,
            ))

    return entries


# ---------------------------------------------------------------------------
# Fallback HTML parser (minimal — for when JSON returns 404)
# ---------------------------------------------------------------------------

def parse_usta_html_fallback(html: str, track_code: str, race_date: str) -> list[RaceEntry]:
    """Very basic BeautifulSoup parse of the USTA program HTML page."""
    from bs4 import BeautifulSoup
    import re

    track_name = TRACKS.get(track_code, (track_code,))[0]
    soup = BeautifulSoup(html, "html.parser")
    entries: list[RaceEntry] = []

    race_num = 0
    for section in soup.find_all(["section", "div"], class_=re.compile(r"race|program", re.I)):
        # Try to find race number
        header = section.find(re.compile(r"h\d"))
        if header:
            m = re.search(r"Race\s+(\d+)", header.get_text(), re.I)
            if m:
                race_num = int(m.group(1))

        for row in section.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 2:
                continue
            entries.append(RaceEntry(
                track_code=track_code,
                track_name=track_name,
                race_date=race_date,
                race_number=race_num,
                surface="Harness",
                program_number=cells[0] if cells else "",
                runner_name=cells[1] if len(cells) > 1 else "",
                jockey=cells[2] if len(cells) > 2 else "",
                trainer=cells[3] if len(cells) > 3 else "",
                breed="Harness",
                source_name=SOURCE_NAME + "_html_fallback",
                source_url="",
            ))

    return entries


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    """Fetch entries for one harness track on one date."""
    if track_code not in TRACKS:
        logger.warning("Unknown USTA track: %s", track_code)
        return []

    # USTA date format for API: YYYY-MM-DD
    url = USTA_BASE.format(date=race_date, track_code=track_code)
    logger.info("Fetching USTA [%s] %s → %s", track_code, race_date, url)

    try:
        data = polite_get_json(session, url)
        write_raw(data, SOURCE_NAME, track_code.lower(), race_date)
        entries = parse_usta_json(data, track_code, race_date)
        logger.info("Parsed %d harness entries for %s", len(entries), track_code)
        return entries

    except Exception as exc:
        logger.warning("USTA JSON failed for %s (%s) — trying HTML fallback", track_code, exc)

    # HTML fallback
    try:
        from utils.common import polite_get
        html_url = USTA_HTML_BASE.format(track_code=track_code, date=race_date)
        resp = polite_get(session, html_url)
        if resp.status_code == 404:
            logger.info("No card for %s on %s", track_code, race_date)
            return []
        resp.raise_for_status()
        entries = parse_usta_html_fallback(resp.text, track_code, race_date)
        logger.info("HTML fallback: parsed %d entries for %s", len(entries), track_code)
        return entries
    except Exception as exc2:
        logger.error("HTML fallback also failed for %s: %s", track_code, exc2)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="USTA harness entries scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", help="Track codes (default: all)")
    args = parser.parse_args()

    target_tracks = args.tracks or list(TRACKS.keys())
    session = build_session()
    all_entries: list[RaceEntry] = []

    for code in target_tracks:
        entries = fetch_track(session, code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, "usta_harness_entries", args.date)
        logger.info("Wrote %d entries to %s", len(all_entries), out)
        master = OUTPUT_ROOT / "processed" / "us_harness_canonical_all.csv"
        append_entries_csv(all_entries, master)
    else:
        logger.warning("No harness entries found for %s", args.date)


if __name__ == "__main__":
    main()
