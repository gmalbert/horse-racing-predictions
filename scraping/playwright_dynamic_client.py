"""
playwright_dynamic_client.py — scrape entries from tracks whose pages
require JavaScript execution (React/Vue SPAs).

This is the "nuclear option" — slower and heavier than JSON/static HTML,
but reliable for tracks that don't expose clean endpoints.

Tracks covered:
  - Keeneland (KEE)          — React SPA, intercepts XHR JSON
  - Santa Anita (SA)         — dynamic calendar / entries
  - Del Mar (DMR)            — dynamic entries page
  - Laurel Park (LRL)        — Stronach Group dynamic site
  - Pimlico (PIM)            — Stronach Group (Preakness home)
  - Golden Gate Fields (GGF) — Stronach Group
  - Tampa Bay Downs (TAM)    — light dynamic (JS-rendered tables)

Strategy per track:
  1. Navigate to entries page with Playwright (Chromium, headless)
  2. Intercept XHR/fetch responses matching entry patterns → parse JSON
  3. If no JSON captured, parse final DOM state with BeautifulSoup

Usage:
    python playwright_dynamic_client.py --date 2026-05-10
    python playwright_dynamic_client.py --date 2026-05-10 --tracks KEE SA DMR
    
Requirements:
    pip install playwright beautifulsoup4
    playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    RaceEntry,
    append_entries_csv,
    build_session,
    get_logger,
    today_str,
    write_entries_csv,
    write_raw,
    OUTPUT_ROOT,
)

logger = get_logger("playwright_dynamic")
SOURCE_NAME = "playwright_dynamic"

# ---------------------------------------------------------------------------
# Track config
# ---------------------------------------------------------------------------

TRACKS: dict[str, dict] = {
    "KEE": {
        "name": "Keeneland",
        "breed": "Thoroughbred",
        "url": "https://www.keeneland.com/racing/entries/{year}/{month}/{day}",
        # Keywords to match API responses during interception
        "api_keywords": ["entries", "racecard", "race-card", "program"],
    },
    "SA": {
        "name": "Santa Anita Park",
        "breed": "Thoroughbred",
        "url": "https://www.santaanita.com/racing/entries/{year}/{month}/{day}",
        "api_keywords": ["entries", "race", "runners"],
    },
    "DMR": {
        "name": "Del Mar",
        "breed": "Thoroughbred",
        "url": "https://www.dmtc.com/racing/entries/?date={date}",
        "api_keywords": ["entries", "race"],
    },
    "LRL": {
        "name": "Laurel Park",
        "breed": "Thoroughbred",
        "url": "https://www.marylandracing.com/racing/entries/?date={date}",
        "api_keywords": ["entries", "race"],
    },
    "PIM": {
        "name": "Pimlico Race Course",
        "breed": "Thoroughbred",
        "url": "https://www.marylandracing.com/racing/entries/?track=PIM&date={date}",
        "api_keywords": ["entries", "race"],
    },
    "TAM": {
        "name": "Tampa Bay Downs",
        "breed": "Thoroughbred",
        "url": "https://www.tampabaydowns.com/racing/entries/{date}",
        "api_keywords": ["entries", "race", "program"],
    },
    "CBY": {
        "name": "Canterbury Park",
        "breed": "Thoroughbred",
        "url": "https://www.canterburypark.com/racing/entries/{date}",
        "api_keywords": ["entries", "race"],
    },
    "PID": {
        "name": "Presque Isle Downs",
        "breed": "Thoroughbred",
        "url": "https://www.presqueisledowns.com/racing/entries/{date}",
        "api_keywords": ["entries", "race"],
    },
}


# ---------------------------------------------------------------------------
# Generic parsers
# ---------------------------------------------------------------------------

def parse_captured_json(
    payloads: list[dict | list],
    track_code: str,
    race_date: str,
) -> list[RaceEntry]:
    """Parse any captured JSON payloads into canonical entries."""
    cfg = TRACKS[track_code]
    track_name = cfg["name"]
    breed = cfg["breed"]
    entries: list[RaceEntry] = []

    for data in payloads:
        # Normalize to list of races
        if isinstance(data, list):
            races = data
        elif isinstance(data, dict):
            races = (
                data.get("races")
                or data.get("entries")
                or data.get("data", {}).get("races")
                or []
            )
            if isinstance(races, dict):
                races = list(races.values())
        else:
            continue

        for race in races:
            if not isinstance(race, dict):
                continue

            race_num = int(
                race.get("raceNumber")
                or race.get("race_number")
                or race.get("number")
                or 0
            )
            runners = (
                race.get("runners")
                or race.get("entries")
                or race.get("horses")
                or []
            )
            surface = race.get("surface") or race.get("trackSurface") or "Dirt"

            for runner in runners:
                entries.append(RaceEntry(
                    track_code=track_code,
                    track_name=track_name,
                    race_date=race_date,
                    race_number=race_num,
                    race_time=str(race.get("postTime") or race.get("post_time") or ""),
                    race_name=str(race.get("raceName") or race.get("name") or ""),
                    race_class=str(race.get("raceClass") or race.get("class") or ""),
                    surface=_norm_surface(str(surface), breed),
                    distance=str(race.get("distance") or ""),
                    purse=str(race.get("purse") or ""),
                    program_number=str(
                        runner.get("programNumber")
                        or runner.get("program")
                        or runner.get("pp")
                        or ""
                    ),
                    runner_name=str(
                        runner.get("horseName")
                        or runner.get("horse_name")
                        or runner.get("name")
                        or ""
                    ),
                    jockey=str(runner.get("jockey") or runner.get("jockeyName") or ""),
                    trainer=str(runner.get("trainer") or runner.get("trainerName") or ""),
                    ml_odds=str(
                        runner.get("morningLine")
                        or runner.get("morning_line")
                        or runner.get("ml")
                        or ""
                    ),
                    scratched=bool(
                        runner.get("scratched")
                        or runner.get("isScratched")
                        or False
                    ),
                    breed=breed,
                    source_name=SOURCE_NAME,
                    source_url="",
                    raw_extra=runner,
                ))

    return entries


def parse_dom_entries(html: str, track_code: str, race_date: str) -> list[RaceEntry]:
    """Fallback DOM parser when no JSON is captured."""
    cfg = TRACKS[track_code]
    track_name = cfg["name"]
    breed = cfg["breed"]
    soup = BeautifulSoup(html, "html.parser")
    entries: list[RaceEntry] = []
    race_num = 0

    for section in soup.find_all(True, class_=re.compile(r"race|entry|program|card", re.I)):
        h = section.find(re.compile(r"h[1-6]"))
        if h:
            m = re.search(r"(?:race|#)\s*(\d+)", h.get_text(), re.I)
            if m:
                race_num = int(m.group(1))

        for row in section.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all(["td"])]
            if len(cells) < 2:
                continue
            entries.append(RaceEntry(
                track_code=track_code,
                track_name=track_name,
                race_date=race_date,
                race_number=race_num,
                surface="Dirt",
                program_number=cells[0],
                runner_name=cells[1] if len(cells) > 1 else "",
                jockey=cells[2] if len(cells) > 2 else "",
                trainer=cells[3] if len(cells) > 3 else "",
                breed=breed,
                source_name=SOURCE_NAME + "_dom",
                source_url="",
            ))

    return entries


def _norm_surface(s: str, breed: str) -> str:
    if breed == "Harness":
        return "Harness"
    s = s.lower()
    if "turf" in s:
        return "Turf"
    if "dirt" in s:
        return "Dirt"
    if any(k in s for k in ["synth", "tapeta", "poly", "all weather"]):
        return "Synthetic"
    return s.title()


# ---------------------------------------------------------------------------
# Playwright fetcher
# ---------------------------------------------------------------------------

def build_url(track_code: str, race_date: str) -> str:
    cfg = TRACKS[track_code]
    parts = race_date.split("-")
    year, month, day = parts[0], parts[1], parts[2]
    return cfg["url"].format(date=race_date, year=year, month=month, day=day)


def fetch_track(track_code: str, race_date: str) -> list[RaceEntry]:
    """Fetch entries using Playwright, capturing XHR JSON or parsing DOM."""
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        logger.error(
            "Playwright not installed. Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )
        return []

    cfg = TRACKS.get(track_code)
    if not cfg:
        logger.warning("Unknown track: %s", track_code)
        return []

    url = build_url(track_code, race_date)
    keywords = cfg.get("api_keywords", ["entries"])
    logger.info("Playwright fetch [%s] %s → %s", track_code, race_date, url)

    captured_json: list[Any] = []
    final_html = ""

    def handle_response(response):
        url_lower = response.url.lower()
        if any(kw in url_lower for kw in keywords):
            ct = response.headers.get("content-type", "")
            if "json" in ct:
                try:
                    captured_json.append(response.json())
                    logger.debug("Captured JSON from %s", response.url)
                except Exception:
                    pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        page.on("response", handle_response)

        try:
            page.goto(url, wait_until="networkidle", timeout=45_000)
            page.wait_for_timeout(3_000)  # extra settle time for lazy-loaded content
            final_html = page.content()
        except PWTimeout:
            logger.warning("Page load timed out for %s — using whatever loaded", track_code)
            try:
                final_html = page.content()
            except Exception:
                pass
        except Exception as exc:
            logger.error("Playwright navigation failed for %s: %s", track_code, exc)
        finally:
            browser.close()

    # Save raw HTML for debugging / fixtures
    if final_html:
        write_raw({"html_length": len(final_html)}, SOURCE_NAME, track_code.lower(), race_date)
        raw_html_path = (
            OUTPUT_ROOT / "raw" / SOURCE_NAME / track_code.lower() / f"{race_date}.html"
        )
        raw_html_path.parent.mkdir(parents=True, exist_ok=True)
        raw_html_path.write_text(final_html, encoding="utf-8")

    # Parse captured JSON first (preferred)
    if captured_json:
        logger.info("Parsing %d captured JSON payloads for %s", len(captured_json), track_code)
        entries = parse_captured_json(captured_json, track_code, race_date)
        if entries:
            return entries

    # DOM fallback
    logger.warning("No JSON captured for %s — falling back to DOM parse", track_code)
    if final_html:
        return parse_dom_entries(final_html, track_code, race_date)

    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Playwright dynamic entries scraper")
    parser.add_argument("--date", default=today_str(), help="Race date YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS.keys()))
    args = parser.parse_args()

    all_entries: list[RaceEntry] = []

    for code in args.tracks:
        entries = fetch_track(code.upper(), args.date)
        all_entries.extend(entries)

    if all_entries:
        out = write_entries_csv(all_entries, "playwright_entries", args.date)
        logger.info("Wrote %d entries to %s", len(all_entries), out)
        master = OUTPUT_ROOT / "processed" / "us_dynamic_canonical_all.csv"
        append_entries_csv(all_entries, master)
    else:
        logger.warning("No dynamic entries collected for %s", args.date)


if __name__ == "__main__":
    main()
