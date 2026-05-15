"""
equibase_client.py — scrape daily entries from Equibase.com

Equibase is the official data provider for thoroughbred racing.
Their entries pages are clean, consistent HTML tables — ideal for parsing.

Covers (via Equibase track codes):
  Thoroughbred Tier 1:
    Tampa Bay Downs (TAM), Charles Town (CT), Mountaineer (MNR),
    Canterbury Park (CBY), Presque Isle Downs (PID), Parx (PRX),
    Penn National (PEN), Ellis Park (ELP), Lone Star Park (LS),
    Sam Houston (HOU)
  Thoroughbred Tier 2:
    Santa Anita (SA), Del Mar (DMR), Gulfstream Park (GP),
    Oaklawn Park (OP), Fair Grounds (FG), Monmouth Park (MTH)
  Thoroughbred Tier 3 (fallback):
    Keeneland (KEE), Belmont (BEL), Saratoga (SAR)

Usage:
    python equibase_client.py --date 2026-05-10
    python equibase_client.py --date 2026-05-10 --tracks TAM CT MNR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

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

logger = get_logger("equibase")

# ---------------------------------------------------------------------------
# Track registry — (equibase_code, full_name, breed)
# ---------------------------------------------------------------------------

TRACKS: dict[str, tuple[str, str]] = {
    # Tier 1 Thoroughbred
    "TAM": ("Tampa Bay Downs", "Thoroughbred"),
    "CT":  ("Charles Town", "Thoroughbred"),
    "MNR": ("Mountaineer", "Thoroughbred"),
    "CBY": ("Canterbury Park", "Thoroughbred"),
    "PID": ("Presque Isle Downs", "Thoroughbred"),
    "PRX": ("Parx Racing", "Thoroughbred"),
    "PEN": ("Penn National", "Thoroughbred"),
    "ELP": ("Ellis Park", "Thoroughbred"),
    "LS":  ("Lone Star Park", "Thoroughbred"),
    "HOU": ("Sam Houston Race Park", "Thoroughbred"),
    # Tier 2 Thoroughbred
    "SA":  ("Santa Anita Park", "Thoroughbred"),
    "DMR": ("Del Mar", "Thoroughbred"),
    "GP":  ("Gulfstream Park", "Thoroughbred"),
    "OP":  ("Oaklawn Park", "Thoroughbred"),
    "FG":  ("Fair Grounds", "Thoroughbred"),
    "MTH": ("Monmouth Park", "Thoroughbred"),
    # Tier 3 Thoroughbred (fallback)
    "KEE": ("Keeneland", "Thoroughbred"),
    "BEL": ("Belmont Park", "Thoroughbred"),
    "SAR": ("Saratoga", "Thoroughbred"),
    "CD":  ("Churchill Downs", "Thoroughbred"),
    # Quarter Horse
    "AZD": ("Arizona Downs", "Quarter Horse"),
    "ZIA": ("Zia Park", "Quarter Horse"),
    "SUN": ("Sunland Park", "Quarter Horse"),
    "DEL": ("Delta Downs", "Quarter Horse"),
    "EVD": ("Evangeline Downs", "Quarter Horse"),
}

BASE_URL = "https://www.equibase.com/static/entry/{track_code}{date_str}USA.html"

SOURCE_NAME = "equibase"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_equibase_entries(html: str, track_code: str, race_date: str) -> list[RaceEntry]:
    """Parse Equibase static entries HTML into RaceEntry records."""
    track_name, breed = TRACKS.get(track_code, (track_code, "Thoroughbred"))
    soup = BeautifulSoup(html, "html.parser")
    entries: list[RaceEntry] = []

    # Each race is wrapped in a div with class 'race-nav' or an <article>
    # Equibase uses <div class="race-entries"> containing tables.
    # We look for race headers (h2 or .race-number) then runner tables.

    race_blocks = soup.find_all("div", class_=re.compile(r"race-entries|raceCard", re.I))
    if not race_blocks:
        # fallback: find all tables that look like entry tables
        race_blocks = soup.find_all("table", class_=re.compile(r"entry", re.I))

    for block in race_blocks:
        # --- Race header metadata ---
        race_num = _extract_race_number(block)
        race_time = _safe_text(block.find(class_=re.compile(r"post.?time", re.I)))
        race_name = _safe_text(block.find(class_=re.compile(r"race.?name|race-title", re.I)))
        race_class = _safe_text(block.find(class_=re.compile(r"race.?class|class.?label", re.I)))
        surface_raw = _safe_text(block.find(class_=re.compile(r"surface", re.I)))
        distance_raw = _safe_text(block.find(class_=re.compile(r"distance", re.I)))
        purse_raw = _safe_text(block.find(class_=re.compile(r"purse", re.I)))

        surface = _normalize_surface(surface_raw, breed)

        # --- Runner rows ---
        runner_rows = _find_runner_rows(block)
        for row in runner_rows:
            cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) < 3:
                continue

            prog_num, name, jockey, trainer, ml_odds, scratched = _parse_runner_cells(cells)

            entries.append(RaceEntry(
                track_code=track_code,
                track_name=track_name,
                race_date=race_date,
                race_number=race_num,
                race_time=race_time,
                race_name=race_name,
                race_class=race_class,
                surface=surface,
                distance=distance_raw,
                purse=purse_raw,
                program_number=prog_num,
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


def _extract_race_number(block) -> int:
    """Try multiple ways to find a race number in a block."""
    for selector in [
        {"class": re.compile(r"race.?num", re.I)},
        {"class": re.compile(r"race.?header", re.I)},
        {"id": re.compile(r"race.?\d", re.I)},
    ]:
        el = block.find(attrs=selector)
        if el:
            m = re.search(r"\d+", el.get_text())
            if m:
                return int(m.group())
    # Check block's own id/class
    for attr in ["id", "class"]:
        val = " ".join(block.get(attr, [])) if attr == "class" else block.get(attr, "")
        m = re.search(r"race[_\-]?(\d+)", str(val), re.I)
        if m:
            return int(m.group(1))
    return 0


def _find_runner_rows(block):
    """Return <tr> elements that represent individual runners."""
    table = block.find("table")
    if not table:
        return []
    rows = table.find_all("tr")
    # Skip header rows (th-heavy rows)
    return [r for r in rows if r.find("td")]


def _parse_runner_cells(cells: list[str]) -> tuple:
    """
    Map cell values to (prog_num, name, jockey, trainer, ml_odds, scratched).
    Equibase column order varies slightly but is generally:
      PP | Horse | Jockey | Trainer | Wgt | ML
    """
    scratched = any("scr" in c.lower() or "scratch" in c.lower() for c in cells)
    prog_num = cells[0] if cells else ""
    name = cells[1] if len(cells) > 1 else ""
    jockey = cells[2] if len(cells) > 2 else ""
    trainer = cells[3] if len(cells) > 3 else ""
    ml_odds = ""
    # Last numeric-looking cell is often ML
    for c in reversed(cells):
        if re.match(r"^\d+/\d+$|^\d+$|^[\d.]+$", c.strip()):
            ml_odds = c.strip()
            break
    return prog_num, name, jockey, trainer, ml_odds, scratched


def _safe_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def _normalize_surface(raw: str, breed: str) -> str:
    if breed == "Harness":
        return "Harness"
    raw = raw.lower()
    if "turf" in raw or "grass" in raw:
        return "Turf"
    if "synth" in raw or "tapeta" in raw or "poly" in raw:
        return "Synthetic"
    if "dirt" in raw:
        return "Dirt"
    return raw.title() or "Dirt"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    """Fetch and parse entries for one track on one date."""
    if track_code not in TRACKS:
        logger.warning("Unknown track code: %s", track_code)
        return []

    date_str = format_date_for_url(race_date, "%m%d%y")  # e.g. 051026
    url = BASE_URL.format(track_code=track_code, date_str=date_str)
    logger.info("Fetching %s [%s] from %s", track_code, race_date, url)

    try:
        resp = polite_get(session, url)
        if resp.status_code == 404:
            logger.info("No card for %s on %s (404)", track_code, race_date)
            return []
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Fetch failed for %s: %s", track_code, exc)
        return []

    track_name, _ = TRACKS[track_code]
    write_raw(resp.text, SOURCE_NAME, track_code.lower(), race_date)

    entries = parse_equibase_entries(resp.text, track_code, race_date)
    logger.info("Parsed %d entries for %s", len(entries), track_code)
    return entries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Equibase entries scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", help="Track codes (default: all)")
    parser.add_argument("--output", default="equibase_entries", help="Output CSV label")
    args = parser.parse_args()

    target_tracks = args.tracks or list(TRACKS.keys())
    session = build_session()
    all_entries: list[RaceEntry] = []

    for code in target_tracks:
        entries = fetch_track(session, code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, args.output, args.date)
        logger.info("Wrote %d entries to %s", len(all_entries), out)

        master = OUTPUT_ROOT / "processed" / "us_equibase_canonical_all.csv"
        append_entries_csv(all_entries, master)
        logger.info("Appended to master: %s", master)
    else:
        logger.warning("No entries collected for %s", args.date)


if __name__ == "__main__":
    main()
