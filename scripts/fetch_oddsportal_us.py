"""
scripts/fetch_oddsportal_us.py
Option C — OddsPortal public odds aggregator for US horse racing.

OddsPortal aggregates odds from many bookmakers for major US horse racing
events. It does NOT cover everyday card entries — coverage is limited to
high-profile races (Triple Crown series, Breeders' Cup, Graded stakes).

Use case: Historical odds comparison, value-bet validation against multiple
bookmakers for stakes races. Complements NYRA (entries) and Betfair (exchange).

Requires: playwright (pip install playwright && playwright install chromium)

Usage:
    python scripts/fetch_oddsportal_us.py --list
    python scripts/fetch_oddsportal_us.py --url https://www.oddsportal.com/horse-racing/usa/kentucky-derby-2026/
    python scripts/fetch_oddsportal_us.py --known --date 2026-05-09 --save
    python scripts/fetch_oddsportal_us.py --date 2026-05-09 --save --show
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
DATA_RAW = _REPO / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ── Known major US race URLs (keep updated each season) ───────────────────────
KNOWN_US_RACES: dict[str, str] = {
    # Triple Crown 2026
    "Kentucky Derby 2026":   "https://www.oddsportal.com/horse-racing/usa/kentucky-derby-2026/",
    "Preakness Stakes 2026": "https://www.oddsportal.com/horse-racing/usa/preakness-stakes-2026/",
    "Belmont Stakes 2026":   "https://www.oddsportal.com/horse-racing/usa/belmont-stakes-2026/",
    # Breeders' Cup
    "Breeders Cup Classic 2025": "https://www.oddsportal.com/horse-racing/usa/breeders-cup-2025/",
    # Other graded stakes (add as OddsPortal publishes them)
    "Travers Stakes 2026":   "https://www.oddsportal.com/horse-racing/usa/travers-stakes-2026/",
    "Whitney Stakes 2026":   "https://www.oddsportal.com/horse-racing/usa/whitney-stakes-2026/",
    "Haskell Stakes 2026":   "https://www.oddsportal.com/horse-racing/usa/haskell-stakes-2026/",
    "Pacific Classic 2026":  "https://www.oddsportal.com/horse-racing/usa/pacific-classic-2026/",
    "Jockey Club Gold Cup 2026": "https://www.oddsportal.com/horse-racing/usa/jockey-club-gold-cup-2026/",
}

# ── User-agent (rotate if needed) ─────────────────────────────────────────────
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ─────────────────────────────────────────────────────────────────────────────
# Async scraping helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_race_list() -> list[dict]:
    """
    Scrape the OddsPortal US horse racing index and return event links.

    Returns:
        [{"name": "Kentucky Derby 2026", "url": "https://..."}, ...]
    """
    from playwright.async_api import async_playwright

    url = "https://www.oddsportal.com/horse-racing/usa/"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=_UA)
        try:
            await page.goto(url, wait_until="load", timeout=30_000)
            await page.wait_for_timeout(4_000)

            events = await page.evaluate("""
                () => {
                    const seen = new Set();
                    return Array.from(
                        document.querySelectorAll('a[href*="/horse-racing/usa/"]')
                    )
                    .filter(a => {
                        const h = a.href;
                        // Only sub-pages (event pages have more path segments)
                        const parts = h.replace(/https?:\\/\\/[^/]+/, '').split('/').filter(Boolean);
                        return parts.length >= 3;
                    })
                    .map(a => ({ name: a.innerText.trim(), url: a.href }))
                    .filter(e => {
                        if (!e.name || seen.has(e.url)) return false;
                        seen.add(e.url);
                        return true;
                    });
                }
            """)
        finally:
            await browser.close()

    return events


async def _fetch_race_odds(race_url: str) -> dict:
    """
    Scrape bookmaker odds from a specific OddsPortal race page.

    Returns:
        {
          "race": <page title>,
          "url": <url>,
          "runners": [
              {"name": "Horse Name", "odds": {"Bet365": 8.0, "Ladbrokes": 7.5, ...}},
              ...
          ],
          "bookmakers": ["Bet365", "Ladbrokes", ...],
          "raw_rows": [[...], ...]   # For debugging
        }
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=_UA)
        try:
            await page.goto(race_url, wait_until="load", timeout=30_000)
            await page.wait_for_timeout(5_000)

            # Extract bookmaker headers and odds rows via JS evaluation
            raw = await page.evaluate("""
                () => {
                    // Try to find the odds table — OddsPortal renders a <table>
                    // with bookmaker logos in <th> and odds in <td>
                    const tables = Array.from(document.querySelectorAll('table'));
                    const biggestTable = tables.sort(
                        (a, b) => b.rows.length - a.rows.length
                    )[0];

                    if (!biggestTable) return { headers: [], rows: [] };

                    const headers = Array.from(
                        biggestTable.querySelectorAll('th')
                    ).map(h => h.innerText.trim()).filter(Boolean);

                    const rows = Array.from(biggestTable.rows).map(row =>
                        Array.from(row.cells).map(c => c.innerText.trim())
                    ).filter(r => r.length > 1 && r[0]);

                    return { headers, rows };
                }
            """)

            title = await page.title()
        finally:
            await browser.close()

    headers = raw.get("headers", [])
    rows = raw.get("rows", [])

    # Parse runners and their odds
    # OddsPortal layout (typical): [Runner Name, Bookie1, Bookie2, ...]
    runners: list[dict] = []
    bookmakers = headers[1:] if len(headers) > 1 else []

    for row in rows:
        if not row or not row[0]:
            continue
        horse_name = _clean_name(row[0])
        if not horse_name:
            continue
        odds_dict: dict[str, float | None] = {}
        for i, bm in enumerate(bookmakers, start=1):
            if i < len(row):
                odds_dict[bm] = _parse_decimal_odds(row[i])
        runners.append({"name": horse_name, "odds": odds_dict})

    return {
        "race": title,
        "url": race_url,
        "bookmakers": bookmakers,
        "runners": runners,
        "raw_rows": rows[:5],  # Keep a few rows for debugging
    }


async def _fetch_multiple(race_urls: list[str], delay: float = 5.0) -> list[dict]:
    results = []
    for url in race_urls:
        print(f"  -> scraping {url}")
        data = await _fetch_race_odds(url)
        results.append(data)
        if url != race_urls[-1]:
            await asyncio.sleep(delay)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_name(raw: str) -> str:
    """Strip extra whitespace and drop non-runner markers."""
    name = raw.strip()
    # Drop lines that look like headers or numbers
    if re.match(r"^[\d\W]+$", name):
        return ""
    return name


def _parse_decimal_odds(raw: str) -> float | None:
    """Convert OddsPortal decimal odds text to float, or None if unparseable."""
    raw = raw.strip()
    if not raw or raw in ("-", "N/A", "—"):
        return None
    try:
        val = float(raw.replace(",", "."))
        return val if val >= 1.0 else None
    except ValueError:
        return None


def _best_odds(runner: dict) -> float | None:
    """Return the highest (best back) odds across all bookmakers."""
    vals = [v for v in runner["odds"].values() if v is not None]
    return max(vals) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Public sync API
# ─────────────────────────────────────────────────────────────────────────────

def get_us_race_list() -> list[dict]:
    """Scrape the OddsPortal US horse racing index. Returns list of event dicts."""
    return asyncio.run(_fetch_race_list())


def get_race_odds(race_url: str) -> dict:
    """Scrape bookmaker odds for a single OddsPortal race URL."""
    return asyncio.run(_fetch_race_odds(race_url))


def get_known_race_odds(race_name: str) -> dict | None:
    """
    Scrape odds for a named known race (key must exist in KNOWN_US_RACES).
    Returns None if the race name is not recognised.
    """
    url = KNOWN_US_RACES.get(race_name)
    if not url:
        return None
    return get_race_odds(url)


def get_all_known_odds(delay: float = 5.0) -> list[dict]:
    """Scrape odds for all entries in KNOWN_US_RACES."""
    urls = list(KNOWN_US_RACES.values())
    return asyncio.run(_fetch_multiple(urls, delay=delay))


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_oddsportal_data(data: list[dict], date_str: str) -> Path:
    """
    Save scraped OddsPortal data to data/raw/oddsportal_us_YYYY-MM-DD.json.

    Args:
        data:     list of race dicts from get_race_odds / get_all_known_odds
        date_str: "YYYY-MM-DD" tag for the file name

    Returns Path to the saved file.
    """
    out_path = DATA_RAW / f"oddsportal_us_{date_str}.json"
    payload = {
        "date":       date_str,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source":     "oddsportal",
        "note":       "Covers major US graded stakes only — not everyday card.",
        "races":      data,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved -> {out_path}")
    return out_path


def load_oddsportal_data(date_str: str) -> dict | None:
    """
    Load a previously saved OddsPortal file for date_str.
    Returns None if no file exists.
    """
    path = DATA_RAW / f"oddsportal_us_{date_str}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def print_oddsportal_data(data: list[dict]) -> None:
    """Pretty-print scraped OddsPortal race odds to the console."""
    for race in data:
        print(f"\n{'='*70}")
        print(f"  {race['race']}")
        print(f"  URL: {race['url']}")
        bms = race.get("bookmakers", [])
        if bms:
            print(f"  Bookmakers ({len(bms)}): {', '.join(bms[:6])}{'…' if len(bms)>6 else ''}")
        print(f"  {'Runner':<35} {'Best Odds':>10}")
        print(f"  {'-'*46}")
        for runner in sorted(race.get("runners", []),
                              key=lambda r: _best_odds(r) or 9999):
            best = _best_odds(runner)
            odds_str = f"{best:.2f}" if best else "N/A"
            print(f"  {runner['name']:<35} {odds_str:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Option C: Scrape US horse racing odds from OddsPortal."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list", action="store_true",
        help="List available US horse racing events on OddsPortal (live scrape).",
    )
    group.add_argument(
        "--url", type=str, metavar="URL",
        help="Scrape odds from a specific OddsPortal race URL.",
    )
    group.add_argument(
        "--known", action="store_true",
        help="Scrape odds for all entries in the KNOWN_US_RACES dict.",
    )
    parser.add_argument(
        "--race", type=str, metavar="NAME",
        help="Name of a single known race to scrape (e.g. 'Kentucky Derby 2026').",
    )
    parser.add_argument(
        "--date", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date tag for saved files (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument("--save", action="store_true", help="Save results to data/raw/.")
    parser.add_argument("--show", action="store_true", help="Print results to console.")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.list:
        print("Fetching US horse racing event list from OddsPortal…")
        events = get_us_race_list()
        print(f"\nFound {len(events)} events:\n")
        for e in events:
            print(f"  {e['name']:<55} {e['url']}")
        return

    if args.url:
        print(f"Scraping: {args.url}")
        data = [get_race_odds(args.url)]
    elif args.race:
        if args.race not in KNOWN_US_RACES:
            print(f"ERROR: Unknown race '{args.race}'. Known races:")
            for k in KNOWN_US_RACES:
                print(f"  {k}")
            sys.exit(1)
        print(f"Scraping known race: {args.race}")
        result = get_known_race_odds(args.race)
        data = [result] if result else []
    elif args.known:
        print(f"Scraping {len(KNOWN_US_RACES)} known US races…")
        data = get_all_known_odds()
    else:
        parser.print_help()
        return

    if not data:
        print("No data returned.")
        return

    if args.show or not args.save:
        print_oddsportal_data(data)

    if args.save:
        save_oddsportal_data(data, args.date)


if __name__ == "__main__":
    main()
