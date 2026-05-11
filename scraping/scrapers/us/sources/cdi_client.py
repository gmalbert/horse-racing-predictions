"""
cdi_client.py — entries and results for Churchill Downs Inc. (CDI) tracks
via the shared CDI internal JSON API.

CDI owns: Churchill Downs, Fair Grounds, and historically provides data
feeds used by TwinSpires.  The API is reachable at:
  https://www.twinspires.com/api/v2/entries/<YYYY-MM-DD>/<track_code>

(TwinSpires is the CDI ADW platform that powers DraftKings Racing.)

Covers:
  - Churchill Downs (CD)
  - Fair Grounds Race Course (FG)

Both tracks share the same TwinSpires/CDI data pipeline.

Usage:
    python cdi_client.py --date 2026-05-10
    python cdi_client.py --date 2026-05-10 --tracks CD FG
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

logger = get_logger("cdi")

# ---------------------------------------------------------------------------
# Track registry
# ---------------------------------------------------------------------------

TRACKS: dict[str, str] = {
    "CD":  "Churchill Downs",
    "FG":  "Fair Grounds Race Course",
    "TP":  "Turfway Park",           # also CDI-affiliated
}

# TwinSpires (CDI) entries API
CDI_API = "https://www.twinspires.com/api/v2/entries/{date}/{track}"

# Alternate CDI direct endpoint
CDI_DIRECT = "https://racing.churchilldowns.com/api/entries/{track}/{date}"

# Churchill Downs official race cards page (Playwright fallback)
CDI_WEB = "https://www.churchilldowns.com/racing/race-entries"

SOURCE_NAME = "cdi"

CDI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.twinspires.com/",
    "Origin": "https://www.twinspires.com",
}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_cdi_entries(data: dict | list, track_code: str, race_date: str) -> list[RaceEntry]:
    """
    CDI/TwinSpires API response shape:
    {
      "entries": {
        "1": {
          "race_number": 1,
          "post_time": "12:30",
          "race_name": "...",
          "race_class": "...",
          "surface": "Dirt",
          "distance": "6 Furlongs",
          "purse": 50000,
          "runners": [
            { "program": "1", "horse_name": "...", "jockey": "...",
              "trainer": "...", "morning_line": "4/1", "scratched": 0 }
          ]
        }
      }
    }
    """
    track_name = TRACKS.get(track_code, track_code)
    entries: list[RaceEntry] = []

    # Normalize top-level structure
    if isinstance(data, list):
        races_raw = {str(i+1): r for i, r in enumerate(data)}
    elif "entries" in data:
        raw = data["entries"]
        races_raw = raw if isinstance(raw, dict) else {str(i+1): r for i, r in enumerate(raw)}
    elif "races" in data:
        races_list = data["races"]
        races_raw = {str(r.get("race_number", i+1)): r for i, r in enumerate(races_list)}
    else:
        races_raw = data

    for race_key, race in races_raw.items():
        if not isinstance(race, dict):
            continue

        race_num = int(race.get("race_number", race.get("raceNumber", race_key)))
        post_time = str(race.get("post_time", race.get("postTime", "")))
        race_name = race.get("race_name", race.get("raceName", ""))
        race_class = race.get("race_class", race.get("raceClass", ""))
        surface = race.get("surface", "Dirt")
        distance = str(race.get("distance", ""))
        purse = str(race.get("purse", ""))

        runners = race.get("runners", race.get("entries", []))

        for runner in runners:
            ml = runner.get("morning_line", runner.get("morningLine", runner.get("ml_odds", "")))
            scratched = bool(int(runner.get("scratched", 0)) if str(runner.get("scratched", "0")).isdigit()
                            else runner.get("scratched", False))
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
                purse=purse,
                program_number=str(runner.get("program", runner.get("programNumber", ""))),
                runner_name=runner.get("horse_name", runner.get("horseName", runner.get("name", ""))),
                jockey=runner.get("jockey", runner.get("jockeyName", "")),
                trainer=runner.get("trainer", runner.get("trainerName", "")),
                ml_odds=str(ml),
                scratched=scratched,
                breed="Thoroughbred",
                source_name=SOURCE_NAME,
                source_url="",
                raw_extra=runner,
            ))

    return entries


def _norm_surface(s: str) -> str:
    s = s.lower()
    if "turf" in s or "grass" in s:
        return "Turf"
    if "dirt" in s:
        return "Dirt"
    if "synth" in s or "all weather" in s or "tapeta" in s:
        return "Synthetic"
    return s.title()


# ---------------------------------------------------------------------------
# Playwright fallback for CDI
# ---------------------------------------------------------------------------

def fetch_via_playwright(track_code: str, race_date: str) -> list[RaceEntry]:
    """
    Use Playwright to open Churchill Downs entries page and capture API calls.
    Used when JSON endpoints rotate or return auth errors.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed.")
        return []

    url = CDI_WEB
    logger.info("Playwright fallback for CDI [%s] %s", track_code, url)

    captured: list[dict] = []

    def handle_response(response):
        url_lower = response.url.lower()
        if any(kw in url_lower for kw in ["entries", "racecard", "program"]):
            try:
                data = response.json()
                captured.append(data)
            except Exception:
                pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("response", handle_response)
        # Navigate and wait for content to load
        page.goto(url, wait_until="networkidle", timeout=45000)
        # Try to click the target track if a selector exists
        try:
            page.click(f"[data-track='{track_code}'], [data-track-code='{track_code}']", timeout=3000)
            page.wait_for_timeout(2000)
        except Exception:
            pass
        browser.close()

    entries: list[RaceEntry] = []
    for payload in captured:
        entries.extend(parse_cdi_entries(payload, track_code, race_date))

    return entries


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_track(session, track_code: str, race_date: str) -> list[RaceEntry]:
    """Try CDI JSON endpoints in order; Playwright as last resort."""
    if track_code not in TRACKS:
        logger.warning("Unknown CDI track: %s", track_code)
        return []

    endpoints = [
        CDI_API.format(date=race_date, track=track_code),
        CDI_DIRECT.format(track=track_code, date=race_date),
    ]

    for url in endpoints:
        logger.info("CDI fetch [%s] %s → %s", track_code, race_date, url)
        try:
            data = polite_get_json(session, url, headers=CDI_HEADERS)
            write_raw(data, SOURCE_NAME, track_code.lower(), race_date)
            entries = parse_cdi_entries(data, track_code, race_date)
            if entries:
                logger.info("Parsed %d CDI entries for %s via %s", len(entries), track_code, url)
                return entries
        except Exception as exc:
            logger.warning("CDI endpoint failed (%s): %s", url, exc)

    logger.warning("All CDI JSON endpoints failed for %s — trying Playwright", track_code)
    return fetch_via_playwright(track_code, race_date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CDI (Churchill Downs Inc.) entries scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS.keys()))
    args = parser.parse_args()

    session = build_session()
    all_entries: list[RaceEntry] = []

    for code in args.tracks:
        entries = fetch_track(session, code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, "cdi_entries", args.date)
        logger.info("Wrote %d CDI entries to %s", len(all_entries), out)
        master = OUTPUT_ROOT / "processed" / "us_thoroughbred_canonical_all.csv"
        append_entries_csv(all_entries, master)
    else:
        logger.warning("No CDI entries for %s", args.date)


if __name__ == "__main__":
    main()
