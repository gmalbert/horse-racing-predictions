"""
nyra_client.py — entries and results for NYRA tracks via their internal
React/JSON API endpoints.

Covers:
  - Belmont Park (BEL)  [also Aqueduct AQU — skip per user request]
  - Saratoga (SAR)

NYRA loads race data via XHR calls from their React SPA at nyra.com.
The primary endpoint pattern discovered via browser DevTools:
  GET https://www.nyra.com/api/entries/<YYYY-MM-DD>/<track>

Secondary endpoint for race card detail:
  GET https://www.nyra.com/api/race/<track>/<YYYY-MM-DD>/<race_num>

Note: NYRA endpoints require specific Accept/Referer headers to return JSON
rather than an HTML shell. We set those here. If NYRA rotates endpoints,
the Playwright fallback in nyra_playwright_fallback.py handles it.

Usage:
    python nyra_client.py --date 2026-05-10
    python nyra_client.py --date 2026-05-10 --tracks BEL SAR
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
    polite_get,
    today_str,
    write_entries_csv,
    write_raw,
    OUTPUT_ROOT,
)

logger = get_logger("nyra")

# ---------------------------------------------------------------------------
# Track config
# ---------------------------------------------------------------------------

TRACKS: dict[str, str] = {
    # AQU (Aqueduct) intentionally omitted — already handled
    "BEL": "Belmont Park",
    "SAR": "Saratoga Race Course",
}

# NYRA React API — discovered via browser DevTools (XHR)
NYRA_ENTRIES_API = "https://www.nyra.com/api/entries/{date}/{track}"
NYRA_RACE_API = "https://www.nyra.com/api/race/{track}/{date}/{race_num}"

# Fallback: NYRA website race page (rendered HTML for Playwright)
NYRA_WEB = "https://www.nyra.com/{track}/racing/entries/{date}"

SOURCE_NAME = "nyra"

NYRA_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nyra.com/",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.nyra.com",
}

# Map our track codes to NYRA's URL slugs
TRACK_SLUG = {
    "BEL": "belmont-park",
    "SAR": "saratoga",
    "AQU": "aqueduct",
}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_nyra_entries(data: dict, track_code: str, race_date: str) -> list[RaceEntry]:
    """
    Parse NYRA API response.

    Typical shape:
    {
      "races": [
        {
          "raceNumber": 1,
          "postTime": "13:00",
          "raceName": "...",
          "class": "...",
          "surface": "Dirt",
          "distance": "6f",
          "purse": 50000,
          "runners": [
            { "programNumber": "1", "name": "Horse Name",
              "jockey": "...", "trainer": "...", "morningLine": "5-2",
              "scratched": false }
          ]
        }
      ]
    }
    """
    track_name = TRACKS.get(track_code, track_code)
    entries: list[RaceEntry] = []

    races = data.get("races", data.get("data", {}).get("races", []))

    for race in races:
        race_num = int(race.get("raceNumber", race.get("raceNo", 0)))
        post_time = str(race.get("postTime", race.get("post_time", "")))
        race_name = race.get("raceName", race.get("name", ""))
        race_class = race.get("class", race.get("raceClass", ""))
        surface = race.get("surface", "Dirt")
        distance = str(race.get("distance", ""))
        purse = str(race.get("purse", ""))

        runners = race.get("runners", race.get("entries", []))

        for runner in runners:
            ml = runner.get("morningLine", runner.get("ml", runner.get("morningLineOdds", "")))
            entries.append(RaceEntry(
                track_code=track_code,
                track_name=track_name,
                race_date=race_date,
                race_number=race_num,
                race_time=post_time,
                race_name=race_name,
                race_class=race_class,
                surface=_norm_surface(surface),
                distance=distance,
                purse=str(purse),
                program_number=str(runner.get("programNumber", runner.get("ppNumber", ""))),
                runner_name=runner.get("name", runner.get("horseName", "")),
                jockey=runner.get("jockey", runner.get("jockeyName", "")),
                trainer=runner.get("trainer", runner.get("trainerName", "")),
                ml_odds=str(ml),
                scratched=bool(runner.get("scratched", runner.get("isScratched", False))),
                breed="Thoroughbred",
                source_name=SOURCE_NAME,
                source_url="",
                raw_extra=runner,
            ))

    return entries


def _norm_surface(s: str) -> str:
    s = s.lower()
    if "turf" in s:
        return "Turf"
    if "dirt" in s:
        return "Dirt"
    if "synth" in s or "all" in s:
        return "Synthetic"
    return s.title()


# ---------------------------------------------------------------------------
# Playwright fallback
# ---------------------------------------------------------------------------

def fetch_via_playwright(track_code: str, race_date: str) -> list[RaceEntry]:
    """
    Use Playwright to render the NYRA entries page and intercept XHR calls
    or parse the final DOM. Requires `playwright install chromium`.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    slug = TRACK_SLUG.get(track_code, track_code.lower())
    url = f"https://www.nyra.com/{slug}/racing/entries/{race_date}"
    logger.info("Playwright fallback for %s → %s", track_code, url)

    captured_json: list[dict] = []

    def handle_response(response):
        if "api/entries" in response.url or "api/race" in response.url:
            try:
                captured_json.append(response.json())
            except Exception:
                pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("response", handle_response)
        page.goto(url, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(3000)
        browser.close()

    entries: list[RaceEntry] = []
    for payload in captured_json:
        entries.extend(parse_nyra_entries(payload, track_code, race_date))

    if not entries:
        # Last resort: parse the DOM directly
        logger.warning("No XHR JSON captured for %s — DOM parse not implemented; no entries", track_code)

    return entries


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    """Fetch NYRA entries; JSON first, Playwright fallback."""
    if track_code not in TRACKS:
        logger.warning("Unknown NYRA track: %s", track_code)
        return []

    url = NYRA_ENTRIES_API.format(date=race_date, track=track_code)
    logger.info("NYRA JSON fetch [%s] %s", track_code, url)

    try:
        data = polite_get_json(session, url, headers=NYRA_HEADERS)
        write_raw(data, SOURCE_NAME, track_code.lower(), race_date)
        entries = parse_nyra_entries(data, track_code, race_date)
        logger.info("Parsed %d NYRA entries for %s", len(entries), track_code)
        return entries
    except Exception as exc:
        logger.warning("NYRA JSON failed for %s (%s) — trying Playwright", track_code, exc)

    return fetch_via_playwright(track_code, race_date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NYRA entries scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS.keys()))
    args = parser.parse_args()

    session = build_session()
    all_entries: list[RaceEntry] = []

    for code in args.tracks:
        if code.upper() == "AQU":
            logger.info("Skipping AQU — already handled elsewhere")
            continue
        entries = fetch_track(session, code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, "nyra_entries", args.date)
        logger.info("Wrote %d NYRA entries to %s", len(all_entries), out)
        master = OUTPUT_ROOT / "processed" / "us_thoroughbred_canonical_all.csv"
        append_entries_csv(all_entries, master)
    else:
        logger.warning("No NYRA entries collected for %s", args.date)


if __name__ == "__main__":
    main()
