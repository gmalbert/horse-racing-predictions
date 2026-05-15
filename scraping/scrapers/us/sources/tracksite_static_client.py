"""
tracksite_static_client.py — scrape entries directly from individual track
websites that expose clean static HTML (no JS hydration needed).

This covers tracks where Equibase is not the primary source or as a
supplementary direct-source pull.  All tracks here use BeautifulSoup.

Tracks covered:
  Thoroughbred:
    Gulfstream Park  — HRN (Horse Racing Nation) / track site
    Oaklawn Park     — track site
    Monmouth Park    — track site
    Hawthorne        — track site
    Indiana Grand / Horseshoe Indianapolis — track site
    Finger Lakes     — track site
  Harness (HTML fallback when USTA JSON unavailable):
    Freehold Raceway — track site

Usage:
    python tracksite_static_client.py --date 2026-05-10
    python tracksite_static_client.py --date 2026-05-10 --tracks GP OAK MTH
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    RaceEntry,
    append_entries_csv,
    build_session,
    format_date_for_url,
    get_logger,
    polite_get,
    today_str,
    write_entries_csv,
    write_raw,
    OUTPUT_ROOT,
)

logger = get_logger("tracksite_static")
SOURCE_NAME = "tracksite_html"

# ---------------------------------------------------------------------------
# Track registry
# (url_template uses {date} = YYYY-MM-DD, {date_us} = MM/DD/YYYY)
# ---------------------------------------------------------------------------

TRACKS: dict[str, dict] = {
    "GP": {
        "name": "Gulfstream Park",
        "breed": "Thoroughbred",
        "url": "https://www.gulfstreampark.com/racing/entries?date={date}",
        "entry_table_class": re.compile(r"entries|race-entries", re.I),
    },
    "OAK": {
        "name": "Oaklawn Park",
        "breed": "Thoroughbred",
        "url": "https://www.oaklawn.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner|race-card", re.I),
    },
    "MTH": {
        "name": "Monmouth Park",
        "breed": "Thoroughbred",
        "url": "https://www.monmouthpark.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "HAW": {
        "name": "Hawthorne Race Course",
        "breed": "Thoroughbred",
        "url": "https://www.hawthorneracecourse.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "IND": {
        "name": "Horseshoe Indianapolis",
        "breed": "Thoroughbred",
        "url": "https://www.horseshoeindianapolis.com/racing/entries/?date={date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "FL": {
        "name": "Finger Lakes Gaming & Racetrack",
        "breed": "Thoroughbred",
        "url": "https://fingerlakesracingny.com/racing-entries/?date={date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "PEN": {
        "name": "Penn National Race Course",
        "breed": "Thoroughbred",
        "url": "https://www.pennnational.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner|race", re.I),
    },
    "PRX": {
        "name": "Parx Racing",
        "breed": "Thoroughbred",
        "url": "https://www.parxracing.com/racing/entries?date={date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "FH": {
        "name": "Freehold Raceway",
        "breed": "Harness",
        "url": "https://www.freeholdraceway.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "MNR": {
        "name": "Mountaineer Casino Racetrack",
        "breed": "Thoroughbred",
        "url": "https://mountaineerracino.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "CT": {
        "name": "Charles Town Races",
        "breed": "Thoroughbred",
        "url": "https://www.charlestownraces.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "ELP": {
        "name": "Ellis Park",
        "breed": "Thoroughbred",
        "url": "https://www.ellisparkracing.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "LS": {
        "name": "Lone Star Park",
        "breed": "Thoroughbred",
        "url": "https://www.lonestarpark.com/racing/entries/?date={date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
    "HOU": {
        "name": "Sam Houston Race Park",
        "breed": "Thoroughbred",
        "url": "https://www.shrp.com/racing/entries/{date}",
        "entry_table_class": re.compile(r"entri|runner", re.I),
    },
}


# ---------------------------------------------------------------------------
# Parser — generic enough for most track HTML layouts
# ---------------------------------------------------------------------------

def parse_track_html(
    html: str,
    track_code: str,
    race_date: str,
    entry_table_class,
) -> list[RaceEntry]:
    """
    Parse a track website entries page.

    Strategy:
    1. Find all race-level containers (divs with class matching race/entries).
    2. Within each, find the runner table.
    3. Extract row-by-row into RaceEntry.

    The column ordering we assume (most common US track sites):
      PP | Horse Name | Jockey | Trainer | Weight | ML Odds
    We handle variations gracefully.
    """
    cfg = TRACKS[track_code]
    track_name = cfg["name"]
    breed = cfg["breed"]
    soup = BeautifulSoup(html, "html.parser")
    entries: list[RaceEntry] = []

    # --- Try to find race containers ---
    race_containers = soup.find_all(
        True,
        class_=re.compile(r"race[\-_]?(card|container|block|section|wrap|panel|entry|row)", re.I),
    )

    if not race_containers:
        # No structured containers — try tables directly
        race_containers = [soup]

    current_race_num = 0
    current_post_time = ""
    current_race_name = ""
    current_distance = ""
    current_surface = ""
    current_purse = ""

    for container in race_containers:
        # Try to parse race metadata from headers / data attributes
        header = container.find(re.compile(r"h[1-6]|\.race.?header", re.I))
        if header:
            text = header.get_text(" ", strip=True)
            m_num = re.search(r"race\s*#?\s*(\d+)", text, re.I)
            if m_num:
                current_race_num = int(m_num.group(1))

            m_time = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm)?)", text, re.I)
            if m_time:
                current_post_time = m_time.group(1)

        # Look for metadata elements
        for meta_cls in ["distance", "surface", "purse", "post-time", "race-name"]:
            el = container.find(class_=re.compile(meta_cls, re.I))
            if el:
                val = el.get_text(" ", strip=True)
                if "distance" in meta_cls:
                    current_distance = val
                elif "surface" in meta_cls:
                    current_surface = val
                elif "purse" in meta_cls:
                    current_purse = val
                elif "post" in meta_cls:
                    current_post_time = val
                elif "name" in meta_cls:
                    current_race_name = val

        # --- Find runner table ---
        tables = container.find_all("table")
        if not tables and entry_table_class:
            tables = soup.find_all("table", class_=entry_table_class)

        for table in tables:
            col_map = _detect_columns(table)
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if not cells or len(cells) < 2:
                    continue
                cell_texts = [c.get_text(" ", strip=True) for c in cells]

                # Skip if this looks like a header
                if all(t.lower() in ["pp", "#", "horse", "jockey", "trainer", "wt", "ml", ""] for t in cell_texts[:4]):
                    continue

                prog, name, jockey, trainer, ml_odds, scratched = _extract_runner(cell_texts, col_map)
                if not name:
                    continue

                entries.append(RaceEntry(
                    track_code=track_code,
                    track_name=track_name,
                    race_date=race_date,
                    race_number=current_race_num,
                    race_time=current_post_time,
                    race_name=current_race_name,
                    surface=_norm_surface(current_surface, breed),
                    distance=current_distance,
                    purse=current_purse,
                    program_number=prog,
                    runner_name=name,
                    jockey=jockey,
                    trainer=trainer,
                    ml_odds=ml_odds,
                    scratched=scratched,
                    breed=breed,
                    source_name=SOURCE_NAME,
                    source_url="",
                ))

    return entries


def _detect_columns(table) -> dict[str, int]:
    """Attempt to detect column indices from header row."""
    header_row = table.find("tr")
    if not header_row:
        return {}
    headers = [th.get_text(" ", strip=True).lower() for th in header_row.find_all(["th", "td"])]
    col_map: dict[str, int] = {}
    for i, h in enumerate(headers):
        if re.search(r"pp|prog|#|no\b", h):
            col_map.setdefault("pp", i)
        elif re.search(r"horse|name", h):
            col_map.setdefault("name", i)
        elif re.search(r"jock|rider|driver", h):
            col_map.setdefault("jockey", i)
        elif re.search(r"train", h):
            col_map.setdefault("trainer", i)
        elif re.search(r"ml|morning|odds", h):
            col_map.setdefault("ml", i)
    return col_map


def _extract_runner(cells: list[str], col_map: dict) -> tuple:
    """Map cell list to (prog, name, jockey, trainer, ml_odds, scratched)."""
    n = len(cells)
    # Use column map if available, else positional defaults
    prog    = cells[col_map["pp"]]      if "pp" in col_map and col_map["pp"] < n else (cells[0] if n > 0 else "")
    name    = cells[col_map["name"]]    if "name" in col_map and col_map["name"] < n else (cells[1] if n > 1 else "")
    jockey  = cells[col_map["jockey"]]  if "jockey" in col_map and col_map["jockey"] < n else (cells[2] if n > 2 else "")
    trainer = cells[col_map["trainer"]] if "trainer" in col_map and col_map["trainer"] < n else (cells[3] if n > 3 else "")
    ml_odds = cells[col_map["ml"]]      if "ml" in col_map and col_map["ml"] < n else ""

    # Fallback: scan reversed cells for ML pattern
    if not ml_odds:
        for c in reversed(cells):
            if re.match(r"^\d+[-/]\d+$|^\d+$", c.strip()):
                ml_odds = c.strip()
                break

    # Scratch detection
    scratched = any("scr" in c.lower() for c in cells)

    return prog, name, jockey, trainer, ml_odds, scratched


def _norm_surface(raw: str, breed: str) -> str:
    if breed == "Harness":
        return "Harness"
    raw = raw.lower()
    if "turf" in raw:
        return "Turf"
    if "synth" in raw or "tapeta" in raw or "poly" in raw:
        return "Synthetic"
    return "Dirt"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    cfg = TRACKS.get(track_code)
    if not cfg:
        logger.warning("Unknown track: %s", track_code)
        return []

    url = cfg["url"].format(date=race_date, date_us=format_date_for_url(race_date, "%m/%d/%Y"))
    logger.info("Track site fetch [%s] %s → %s", track_code, race_date, url)

    try:
        resp = polite_get(session, url)
        if resp.status_code == 404:
            logger.info("No card for %s on %s (404)", track_code, race_date)
            return []
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Fetch failed for %s: %s", track_code, exc)
        return []

    write_raw(resp.text, SOURCE_NAME, track_code.lower(), race_date)
    entries = parse_track_html(resp.text, track_code, race_date, cfg.get("entry_table_class"))
    logger.info("Parsed %d entries for %s via HTML", len(entries), track_code)
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track site static HTML scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS.keys()))
    args = parser.parse_args()

    session = build_session()
    all_entries: list[RaceEntry] = []

    for code in args.tracks:
        entries = fetch_track(session, code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, "tracksite_entries", args.date)
        logger.info("Wrote %d entries to %s", len(all_entries), out)
        master = OUTPUT_ROOT / "processed" / "us_tracksite_canonical_all.csv"
        append_entries_csv(all_entries, master)
    else:
        logger.warning("No tracksite entries for %s", args.date)


if __name__ == "__main__":
    main()
