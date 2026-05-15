"""
quarter_horse_client.py — entries for US Quarter Horse tracks.

Sources:
  Los Alamitos  — track JSON API (best QH data)
  Ruidoso Downs — track JSON API
  Remington Park— track JSON API (QH meet only)
  Delta Downs   — Equibase (re-uses equibase_client as library)
  Evangeline    — Equibase (re-uses equibase_client as library)
  Zia Park      — track site HTML
  Sunland Park  — track site HTML

Usage:
    python quarter_horse_client.py --date 2026-05-10
    python quarter_horse_client.py --date 2026-05-10 --tracks LAD RUI ZIA
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
    polite_get_json,
    today_str,
    write_entries_csv,
    write_raw,
    OUTPUT_ROOT,
)

logger = get_logger("quarter_horse")

SOURCE_NAME = "qh_tracksite"

# ---------------------------------------------------------------------------
# Track config
# ---------------------------------------------------------------------------

TRACKS: dict[str, dict] = {
    "LAD": {
        "name": "Los Alamitos Race Course",
        "method": "json",
        "url": "https://www.losalamitosmeet.com/wp-json/racetrack/v1/entries?date={date}",
        "date_fmt": "%Y-%m-%d",
    },
    "RUI": {
        "name": "Ruidoso Downs",
        "method": "json",
        "url": "https://www.ruidosodowns.com/api/entries/{date}",
        "date_fmt": "%Y-%m-%d",
    },
    "REM": {
        "name": "Remington Park",
        "method": "json",
        "url": "https://www.remingtonpark.com/api/v1/race-entries/{date}",
        "date_fmt": "%Y-%m-%d",
    },
    "ZIA": {
        "name": "Zia Park",
        "method": "html",
        "url": "https://ziapark.com/racing/entries/{date_slash}",
        "date_fmt": "%m/%d/%Y",   # slash date used in path
    },
    "SUN": {
        "name": "Sunland Park",
        "method": "html",
        "url": "https://www.sunland-park.com/racing/entries/?date={date}",
        "date_fmt": "%Y-%m-%d",
    },
    "EVD": {
        "name": "Evangeline Downs",
        "method": "equibase",
        "equibase_code": "EVD",
    },
    "DEL": {
        "name": "Delta Downs",
        "method": "equibase",
        "equibase_code": "DEL",
    },
    "AZD": {
        "name": "Arizona Downs",
        "method": "html",
        "url": "https://www.arizonadowns.com/racing/daily-entries/",
        "date_fmt": "%Y-%m-%d",
    },
}


# ---------------------------------------------------------------------------
# Generic JSON parser
# ---------------------------------------------------------------------------

def parse_qh_json(data: dict | list, track_code: str, race_date: str) -> list[RaceEntry]:
    """Map common QH track JSON shapes to canonical entries."""
    track_name = TRACKS[track_code]["name"]
    entries: list[RaceEntry] = []

    # Normalize to list of races
    if isinstance(data, list):
        races = data
    elif "races" in data:
        races = data["races"]
    elif "data" in data:
        inner = data["data"]
        races = inner if isinstance(inner, list) else inner.get("races", [])
    else:
        races = [data]

    for race in races:
        if not isinstance(race, dict):
            continue

        race_num = int(race.get("raceNumber", race.get("race_number", race.get("number", 0))))
        runners = race.get("entries", race.get("runners", race.get("horses", [])))

        for runner in runners:
            entries.append(RaceEntry(
                track_code=track_code,
                track_name=track_name,
                race_date=race_date,
                race_number=race_num,
                race_time=str(race.get("postTime", race.get("post_time", ""))),
                race_name=race.get("raceName", race.get("name", "")),
                race_class=race.get("class", race.get("raceClass", "")),
                surface=race.get("surface", "Dirt"),
                distance=str(race.get("distance", "")),
                purse=str(race.get("purse", "")),
                program_number=str(runner.get("program", runner.get("programNumber", runner.get("pp", "")))),
                runner_name=runner.get("horseName", runner.get("horse_name", runner.get("name", ""))),
                jockey=runner.get("jockey", runner.get("jockeyName", "")),
                trainer=runner.get("trainer", runner.get("trainerName", "")),
                ml_odds=str(runner.get("morningLine", runner.get("morning_line", runner.get("ml", "")))),
                scratched=bool(runner.get("scratched", False)),
                breed="Quarter Horse",
                source_name=SOURCE_NAME,
                source_url="",
                raw_extra=runner,
            ))

    return entries


# ---------------------------------------------------------------------------
# Generic HTML parser (for Zia, Sunland, Arizona Downs)
# ---------------------------------------------------------------------------

def parse_qh_html(html: str, track_code: str, race_date: str) -> list[RaceEntry]:
    """Parse a typical racing track HTML entries page."""
    track_name = TRACKS[track_code]["name"]
    soup = BeautifulSoup(html, "html.parser")
    entries: list[RaceEntry] = []
    race_num = 0

    # Try to find race sections
    sections = soup.find_all(
        lambda tag: tag.name in ["div", "section", "article"]
        and re.search(r"race|card|entries", " ".join(tag.get("class", [])), re.I)
    )

    if not sections:
        # Fall back to scanning the whole page for tables
        sections = [soup]

    for section in sections:
        # Detect race number from heading
        header = section.find(re.compile(r"h[1-6]"))
        if header:
            m = re.search(r"race\s+(\d+)|#(\d+)", header.get_text(), re.I)
            if m:
                race_num = int(m.group(1) or m.group(2))

        for table in section.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue
                # Skip header rows
                if cells[0].lower() in ["pp", "program", "#", "no"]:
                    continue

                prog = cells[0] if cells else ""
                name = cells[1] if len(cells) > 1 else ""
                jockey = cells[2] if len(cells) > 2 else ""
                trainer = cells[3] if len(cells) > 3 else ""
                ml = ""
                for c in reversed(cells):
                    if re.match(r"^\d+/\d+$|^\d+-\d+$", c.strip()):
                        ml = c.strip()
                        break

                if not name or name.lower() in ["horse", "horse name"]:
                    continue

                entries.append(RaceEntry(
                    track_code=track_code,
                    track_name=track_name,
                    race_date=race_date,
                    race_number=race_num,
                    surface="Dirt",
                    program_number=prog,
                    runner_name=name,
                    jockey=jockey,
                    trainer=trainer,
                    ml_odds=ml,
                    breed="Quarter Horse",
                    source_name=SOURCE_NAME,
                    source_url="",
                ))

    return entries


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_json_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    cfg = TRACKS[track_code]
    url = cfg["url"].format(date=race_date)
    logger.info("QH JSON fetch [%s] %s", track_code, url)
    try:
        data = polite_get_json(session, url)
        write_raw(data, SOURCE_NAME, track_code.lower(), race_date)
        entries = parse_qh_json(data, track_code, race_date)
        logger.info("Parsed %d QH JSON entries for %s", len(entries), track_code)
        return entries
    except Exception as exc:
        logger.error("QH JSON failed for %s: %s", track_code, exc)
        return []


def fetch_html_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    cfg = TRACKS[track_code]
    date_slash = format_date_for_url(race_date, "%m/%d/%Y")
    url = cfg["url"].format(date=race_date, date_slash=date_slash)
    logger.info("QH HTML fetch [%s] %s", track_code, url)
    try:
        resp = polite_get(session, url)
        if resp.status_code == 404:
            logger.info("No card for %s on %s", track_code, race_date)
            return []
        resp.raise_for_status()
        write_raw(resp.text, SOURCE_NAME, track_code.lower(), race_date)
        entries = parse_qh_html(resp.text, track_code, race_date)
        logger.info("Parsed %d QH HTML entries for %s", len(entries), track_code)
        return entries
    except Exception as exc:
        logger.error("QH HTML failed for %s: %s", track_code, exc)
        return []


def fetch_equibase_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    """Delegate to equibase_client for DEL, EVD."""
    try:
        from sources.equibase_client import fetch_track as eq_fetch
    except ImportError:
        from equibase_client import fetch_track as eq_fetch

    eq_code = TRACKS[track_code]["equibase_code"]
    entries = eq_fetch(session, eq_code, race_date)
    # Re-tag as Quarter Horse breed
    for e in entries:
        e.breed = "Quarter Horse"
        e.track_code = track_code
    return entries


def fetch_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    cfg = TRACKS.get(track_code)
    if not cfg:
        logger.warning("Unknown QH track: %s", track_code)
        return []

    method = cfg["method"]
    if method == "json":
        return fetch_json_track(session, track_code, race_date)
    elif method == "html":
        return fetch_html_track(session, track_code, race_date)
    elif method == "equibase":
        return fetch_equibase_track(session, track_code, race_date)
    else:
        logger.error("Unknown method %s for %s", method, track_code)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quarter Horse entries scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS.keys()))
    args = parser.parse_args()

    session = build_session()
    all_entries: list[RaceEntry] = []

    for code in args.tracks:
        entries = fetch_track(session, code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, "qh_entries", args.date)
        logger.info("Wrote %d QH entries to %s", len(all_entries), out)
        master = OUTPUT_ROOT / "processed" / "us_qh_canonical_all.csv"
        append_entries_csv(all_entries, master)
    else:
        logger.warning("No QH entries for %s", args.date)


if __name__ == "__main__":
    main()
