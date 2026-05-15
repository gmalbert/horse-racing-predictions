"""
fetch_nyra_entries.py
Scrapes NYRA track websites for today's entries and recent results.
NYRA covers: Belmont Park (BEL), Aqueduct (AQU), Saratoga (SAR).

NYRA pages are JavaScript-rendered — uses Playwright for extraction.
Data is saved to data/raw/nyra_entries_YYYY-MM-DD.json.

Usage:
    python scripts/fetch_nyra_entries.py [--date YYYY-MM-DD] [--tracks BEL AQU SAR]

Requirements:
    playwright (pip install playwright && playwright install chromium)
"""

import asyncio
import json
import argparse
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── output directory ────────────────────────────────────────────────────────
DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

TRACKS = {
    "BEL": {"name": "Belmont Park", "base": "https://www.nyra.com/belmont"},
    "AQU": {"name": "Aqueduct",     "base": "https://www.nyra.com/aqueduct"},
    "SAR": {"name": "Saratoga",     "base": "https://www.nyra.com/saratoga"},
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

MONTH_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

SEASON_RANGE_RE = re.compile(
    r"\b(?P<start_month>[A-Z][a-z]{2})\.?\s+(?P<start_day>\d{1,2})\s*-\s*"
    r"(?P<end_month>[A-Z][a-z]{2})\.?\s+(?P<end_day>\d{1,2}),\s*(?P<year>\d{4})\b"
)


# ─────────────────────────────────────────────────────────────────────────────
# Core scraping helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_page_text(url: str, wait_ms: int = 4000) -> str:
    """Return the innerText of a JS-rendered page body."""
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            await page.wait_for_timeout(wait_ms)
            text = await page.inner_text("body")
        finally:
            await browser.close()
    return text


def _requested_date_from_url(url: str) -> date | None:
    """Extract the requested NYRA card date from a URL query string."""
    try:
        qs = parse_qs(urlparse(url).query)
        day = qs.get("day", [""])[0]
        if not day:
            return None
        return datetime.strptime(day, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _page_is_not_found(page_title: str, body_text: str) -> bool:
    """Return True when NYRA serves its branded 404 page instead of entries."""
    lowered = f"{page_title}\n{body_text[:1500]}".lower()
    return "404 - not found" in lowered or "something went wrong! let's get you back on track." in lowered


def _page_is_out_of_meet(body_text: str, requested_date: date | None) -> bool:
    """Return True when the page advertises a meet window that excludes the requested date."""
    if requested_date is None:
        return False

    match = SEASON_RANGE_RE.search(body_text[:2000])
    if not match:
        return False

    start_month = MONTH_TO_NUM.get(match.group("start_month").lower())
    end_month = MONTH_TO_NUM.get(match.group("end_month").lower())
    year = int(match.group("year"))
    if start_month is None or end_month is None:
        return False

    start_date = date(year, start_month, int(match.group("start_day")))
    end_date = date(year, end_month, int(match.group("end_day")))
    return not (start_date <= requested_date <= end_date)


def _card_signature(races: list[dict]) -> tuple:
    """Build a stable signature for a racecard so mirrored cards can be dropped."""
    signature = []
    for race in races:
        runners = tuple(
            (runner.get("horse") or "").strip().lower()
            for runner in race.get("runners", [])
            if (runner.get("horse") or "").strip()
        )
        if runners:
            signature.append((race.get("race") or "", runners))
    return tuple(signature)


async def _fetch_entries_structured(url: str) -> list[dict]:
    """
    Extract structured race/runner data from a NYRA entries page.

    NYRA uses React SSR.  Each individual race is accessible via
    ``?day=YYYY-MM-DD&limit=entries&race=N``.

    Strategy:
      1. Load the base entries page (race 1 shown by default) and detect
         the race-number tabs to determine total races on the card.
      2. Iterate races 1..N, fetching each in a new tab (reusing the browser).
      3. Extract runners using the `.space-y-3\\.5 / .flex.items-start` DOM anchors.

    Stable class anchors (discovered from live DOM inspection 2026-05):
      • `.space-y-3\\.5` — runner list container for the visible race
      • `.flex.items-start` inside that — individual runner row
      • `.order-3` — horse name / jockey / weight text block
    """
    from playwright.async_api import async_playwright

    requested_date = _requested_date_from_url(url)

    # Split off any query string - we'll add our own
    base_url = url.split("?")[0].rstrip("/")
    # Extract date from the url (caller passes it as query or we extract below)
    # We'll build URLs like: {base_url}/?day=YYYY-MM-DD&limit=entries&race=N

    async def _extract_runners(page) -> list[dict]:
        return await page.evaluate("""
            () => {
                const containers = document.querySelectorAll('.space-y-3\\\\.5');
                const container = containers[containers.length - 1];
                if (!container) return [];

                const rowEls = container.querySelectorAll('div.flex.items-start');
                const runners = [];

                for (const row of rowEls) {
                    const infoBlock = row.querySelector('.order-3');
                    if (!infoBlock) continue;

                    const lines = infoBlock.innerText
                        .split('\\n')
                        .map(s => s.trim())
                        .filter(Boolean);

                    if (!lines[0]) continue;

                    const horseName = lines[0];
                    const parts = (lines[1] || '').split('•').map(s => s.trim());
                    const jockey  = parts[0] || '';
                    const trainer = parts[1] || '';
                    const wgtParts = (lines[2] || '').split('•').map(s => s.trim());
                    const weight = wgtParts[0] || '';
                    const equip  = wgtParts[1] || '';
                    const ageSex = wgtParts[2] || '';

                    const otherText = row.innerText
                        .replace(infoBlock.innerText, '')
                        .split('\\n')
                        .map(s => s.trim())
                        .filter(Boolean);

                    const prog_no   = otherText[0] || '';
                    const live_odds = otherText[1] || '';
                    const ml_raw    = otherText.find(t => t.startsWith('ML')) || '';
                    const ml_odds   = ml_raw.replace(/^ML\\s*/, '');

                    runners.push({
                        horse:   horseName,
                        jockey:  jockey,
                        trainer: trainer,
                        weight:  weight,
                        equip:   equip,
                        age_sex: ageSex,
                        number:  prog_no,
                        odds:    live_odds,
                        ml_odds: ml_odds,
                    });
                }
                return runners;
            }
        """)

    async def _extract_race_meta(page, race_number: int) -> dict:
        """Extract race-level metadata from the currently displayed NYRA race page."""
        return await page.evaluate(
            r"""
            (raceNumber) => {
                const lines = document.body.innerText
                    .split('\n')
                    .map(s => s.trim())
                    .filter(Boolean);

                const raceLabel = `Race ${raceNumber}`;
                let lastRaceIdx = -1;
                for (let i = 0; i < lines.length; i++) {
                    if (lines[i] === raceLabel) {
                        lastRaceIdx = i;
                    }
                }

                const result = {
                    race: raceLabel,
                    race_name: '',
                    race_class: '',
                    distance: '',
                    surface: '',
                    going: '',
                    race_time: '',
                    post_time: '',
                    conditions: '',
                    prize: '',
                    course: '',
                };

                if (lastRaceIdx < 0) {
                    return result;
                }

                const windowLines = lines.slice(lastRaceIdx + 1, lastRaceIdx + 14);
                const filtered = windowLines.filter(line => {
                    if (!line) return false;
                    if (/^\d+$/.test(line)) return false;
                    if (/^Race\s+\d+$/i.test(line)) return false;
                    return true;
                });

                const distanceRe = /(?:\d+\s+\d+\/\d+|\d+\/\d+|\d+(?:\.\d+)?)\s*(?:F|M|Miles?|Yards?)\b/i;
                const surfaceRe = /^(Dirt|Turf|Synthetic|Tapeta|Inner Turf|Outer Turf)\b/i;
                const goingRe = /^(Fast|Firm|Good|Yielding|Sloppy|Muddy|Soft|Heavy|Wet Fast|Frozen)\b/i;
                const timeRe = /\b\d{1,2}:\d{2}[ap]\b/i;
                const betRe = /^(Exacta|Trifecta|Super|Double|Pick\s*\d|Daily Double)\b/i;

                for (const line of filtered) {
                    if (!result.race_name && !distanceRe.test(line) && !surfaceRe.test(line) && !goingRe.test(line) && !timeRe.test(line) && !betRe.test(line)) {
                        result.race_name = line;
                        result.race_class = line;
                        const prizeMatch = line.match(/\$[\d,]+/);
                        if (prizeMatch) {
                            result.prize = prizeMatch[0];
                        }
                        continue;
                    }

                    if (!result.distance && distanceRe.test(line)) {
                        result.distance = line;
                        continue;
                    }

                    if (!result.surface && surfaceRe.test(line)) {
                        result.surface = line;
                        continue;
                    }

                    if (!result.going && goingRe.test(line)) {
                        result.going = line;
                        continue;
                    }

                    if (!result.race_time && timeRe.test(line)) {
                        const tm = line.match(timeRe);
                        result.race_time = tm ? tm[0] : '';
                        result.post_time = result.race_time;
                        const atMatch = line.match(/at\s+(.+)$/i);
                        if (atMatch) {
                            result.course = atMatch[1].trim();
                        }
                        continue;
                    }

                    if (!result.conditions && line.length > 30 && !betRe.test(line)) {
                        result.conditions = line;
                    }
                }

                return result;
            }
            """,
            race_number,
        )

    async def _get_race_count(page) -> int:
        """Return the number of races on today's card."""
        return await page.evaluate("""
            () => {
                // Race tabs within the day's card are anchor links whose href
                // contains 'race=' — collect unique race numbers
                const links = Array.from(document.querySelectorAll('a[href*="race="]'));
                const nums = links.map(a => {
                    const m = (a.href || '').match(/race=(\\d+)/);
                    return m ? parseInt(m[1], 10) : 0;
                }).filter(n => n > 0);
                return nums.length > 0 ? Math.max(...nums) : 1;
            }
        """)


    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()

            # ── Step 1: load day page (race 1) ────────────────────────
            day_url = f"{base_url}/?limit=entries&race=1"
            # If url already has a day param, preserve it
            if "day=" in url:
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(url).query)
                day = qs.get("day", [""])[0]
                if day:
                    day_url = f"{base_url}/?day={day}&limit=entries&race=1"

            try:
                await page.goto(day_url, wait_until="load", timeout=45_000)
            except Exception:
                pass
            await page.wait_for_timeout(8_000)

            page_title = await page.title()
            body_text = await page.inner_text("body")
            if _page_is_not_found(page_title, body_text):
                return []
            if _page_is_out_of_meet(body_text, requested_date):
                return []

            # Determine how many races are on the card
            n_races = await _get_race_count(page)
            n_races = min(n_races, 20)  # safety cap

            results: list[dict] = []

            # Extract race 1 (already loaded)
            race_meta = await _extract_race_meta(page, 1)
            runners = await _extract_runners(page)
            if runners:
                results.append({**race_meta, "runners": runners})

            # ── Step 2: load each subsequent race ─────────────────────
            for rn in range(2, n_races + 1):
                race_url = day_url.replace("race=1", f"race={rn}")
                try:
                    await page.goto(race_url, wait_until="load", timeout=30_000)
                except Exception:
                    pass
                await page.wait_for_timeout(3_000)

                race_meta = await _extract_race_meta(page, rn)
                runners = await _extract_runners(page)
                if runners:
                    results.append({**race_meta, "runners": runners})

            if not results:
                return []

        finally:
            await browser.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_nyra_entries(track_code: str, date_str: str | None = None) -> list[dict]:
    """
    Fetch entries for a single NYRA track (synchronous wrapper).

    Args:
        track_code: "BEL", "AQU", or "SAR"
        date_str:   "YYYY-MM-DD" (defaults to today)

    Returns list of race dicts, each with 'race', 'runners', 'track' keys.
    """
    if track_code not in TRACKS:
        raise ValueError(f"Unknown NYRA track '{track_code}'. Valid: {list(TRACKS)}")

    if date_str is None:
        date_str = date.today().isoformat()

    base = f"{TRACKS[track_code]['base']}/racing/entries"
    url  = f"{base}/?day={date_str}&limit=entries"
    print(f"  [{track_code}] Fetching: {url}")

    races = asyncio.run(_fetch_entries_structured(url))

    for race in races:
        race["track"] = track_code
        race["track_name"] = TRACKS[track_code]["name"]

    print(f"  [{track_code}] Got {len(races)} race blocks")
    return races


def get_all_nyra_entries(
    tracks: list[str] | None = None, date_str: str | None = None
) -> dict[str, list]:
    """
    Fetch entries from all (or specified) NYRA tracks.

    Args:
        tracks:   list of track codes to fetch, defaults to all three
        date_str: "YYYY-MM-DD" (defaults to today)

    Returns dict mapping track_code → list of race dicts.
    """
    if tracks is None:
        tracks = list(TRACKS.keys())

    results: dict[str, list] = {}
    seen_cards: dict[tuple, str] = {}
    for code in tracks:
        try:
            races = get_nyra_entries(code, date_str)
            signature = _card_signature(races)
            if signature and signature in seen_cards:
                print(f"  [{code}] Duplicate card detected (matches {seen_cards[signature]}); skipping")
                results[code] = []
                continue
            results[code] = races
            if signature:
                seen_cards[signature] = code
        except Exception as exc:
            print(f"  [{code}] ERROR: {exc}")
            results[code] = []

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_nyra_entries(data: dict[str, list], date_str: str) -> Path:
    """
    Persist NYRA entries to data/raw/nyra_entries_YYYY-MM-DD.json.

    Args:
        data:     dict of {track_code: [race_dicts]}
        date_str: "YYYY-MM-DD" used in the filename

    Returns the Path of the written file.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / f"nyra_entries_{date_str}.json"
    payload = {
        "date": date_str,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "nyra",
        "tracks": data,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved -> {out_path}")
    return out_path


def load_nyra_entries(date_str: str) -> dict | None:
    """
    Load previously saved NYRA entries.

    Returns the parsed JSON dict, or None if the file doesn't exist.
    """
    p = DATA_RAW / f"nyra_entries_{date_str}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NYRA track entries (Belmont, Aqueduct, Saratoga)"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date label for the output file (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=list(TRACKS.keys()),
        default=list(TRACKS.keys()),
        help="Which NYRA tracks to fetch. Default: BEL AQU SAR",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the fetched entries to stdout",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    print("=" * 60)
    print(f"Fetching NYRA Entries for {args.date}")
    print(f"Tracks: {args.tracks}")
    print("=" * 60)

    all_data = get_all_nyra_entries(args.tracks, args.date)
    save_nyra_entries(all_data, args.date)

    total_races = sum(len(v) for v in all_data.values())
    total_runners = sum(
        sum(len(r.get("runners", [])) for r in races)
        for races in all_data.values()
    )
    print(f"\nDone — {total_races} race blocks, {total_runners} runners fetched")

    if args.show:
        for track_code, races in all_data.items():
            print(f"\n{'─'*50}")
            print(f"  {TRACKS[track_code]['name']} ({track_code})")
            print(f"{'─'*50}")
            for race in races:
                print(f"\n  {race.get('race', '?')}")
                if "raw_text" in race:
                    print(f"    [raw text preview]: {race['raw_text'][:200]}")
                else:
                    for r in race.get("runners", []):
                        num = r.get("number", "").ljust(3)
                        horse = r.get("horse", "?").ljust(30)
                        jockey = r.get("jockey", "").ljust(20)
                        ml = r.get("ml_odds", "")
                        print(f"    {num} {horse} {jockey} {ml}")


if __name__ == "__main__":
    main()
