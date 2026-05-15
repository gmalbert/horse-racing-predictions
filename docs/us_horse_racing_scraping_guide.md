# US Horse Racing Data — Scraping & Source Options
### Multiple approaches, Python code, trade-offs & legal notes

---

## ⚠️ Legal & Ethical Notice

Web scraping exists in a grey zone that varies by source. **Read this before running any code.**

| Source | ToS on Scraping | Risk Level |
|---|---|---|
| Equibase | **Explicitly prohibited** — bots/spiders banned in ToS | High — cease & desist possible |
| OddsPortal | No automated access clause; data is publicly visible | Medium |
| OddsChecker | Login required for some data; ToS restricts automation | Medium–High |
| The Racing API | Legitimate paid API — US data on Pro plan | None (licensed) |
| NYRA (nyra.com) | No explicit prohibition; public data | Low |
| ATR / Racing TV | Varies; public pages generally lower risk | Low–Medium |

**Best practice for all scraping:**
- Add delays between requests (2–5 seconds minimum)
- Respect `robots.txt`
- Use a realistic User-Agent string
- Do not resell or republish scraped data
- Stop immediately if you receive a legal notice

> This document is for **personal research and educational use only**. The author is not responsible for how this code is used.

---

## Table of Contents

1. [Overview of Options](#1-overview-of-options)
2. [Option A — The Racing API (Recommended, Licensed)](#2-option-a--the-racing-api-recommended-licensed)
3. [Option B — Equibase (Official US Source, ToS Restricted)](#3-option-b--equibase-official-us-source-tos-restricted)
4. [Option C — OddsPortal (Public Odds Aggregator)](#4-option-c--oddsportal-public-odds-aggregator)
5. [Option D — NYRA / Track Websites (Public Entries & Results)](#5-option-d--nyra--track-websites-public-entries--results)
6. [Option E — Betfair Exchange API (US Tracks)](#6-option-e--betfair-exchange-api-us-tracks)
7. [Combining Sources — Recommended Architecture](#7-combining-sources--recommended-architecture)
8. [Shared Utilities](#8-shared-utilities)
9. [US Track Code Reference](#9-us-track-code-reference)

---

## 1. Overview of Options

| # | Source | Data Type | Cost | Complexity | US Coverage |
|---|---|---|---|---|---|
| A | The Racing API | Entries, odds, results, form | Paid (~£30/mo Lite) | Low | ✅ Full |
| B | Equibase | Entries, results, charts | Free (public pages) | Medium | ✅ Full |
| C | OddsPortal | Historical & upcoming odds | Free (public) | High | ✅ Major races |
| D | NYRA / track sites | Entries, live results | Free (public) | Medium | 🟡 NYRA tracks only |
| E | Betfair Exchange API | Live market prices | Free with account | Medium | ✅ Major meetings |

**Recommended approach:**
- **Pre-race entries & form:** The Racing API (Option A) or Equibase (Option B)
- **Live & pre-race odds:** Betfair Exchange (Option E) — it's a genuine API with no scraping needed
- **Historical odds for analysis:** OddsPortal (Option C)
- **NYRA-specific live data:** NYRA site (Option D)

---

## 2. Option A — The Racing API (Recommended, Licensed)

[The Racing API](https://www.theracingapi.com/) provides UK, Ireland, Australia, and US coverage under a proper licence. The Lite plan (~£30/mo) covers US entries and results. This is the cleanest approach — no ToS risk, structured JSON, real API.

### Setup

```python
# requirements.txt additions
# requests>=2.31.0
# python-dotenv>=1.0.0
```

```
# .env
RACING_API_USER=your_username
RACING_API_PASS=your_password
```

### Core Client

```python
"""
racing_api_client.py
Client for The Racing API (theracingapi.com).
Uses HTTP Basic Auth.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.theracingapi.com/v1"
AUTH = (
    os.getenv("RACING_API_USER", ""),
    os.getenv("RACING_API_PASS", ""),
)


def _get(path: str, params: dict = None) -> dict:
    url = BASE_URL + path
    r = requests.get(url, auth=AUTH, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ─── US Entries ─────────────────────────────────

def get_us_racecards(date: str = None) -> dict:
    """
    Fetch today's (or a given date's) US race entries.

    Args:
        date: "YYYY-MM-DD" — defaults to today

    Returns dict with list of races, runners, distances, going, etc.
    """
    params = {"region": "usa"}
    if date:
        params["date"] = date
    return _get("/racecards/standard", params=params)


def get_us_results(date: str = None) -> dict:
    """
    Fetch US race results for a date.

    Args:
        date: "YYYY-MM-DD" — defaults to today
    """
    params = {"region": "usa"}
    if date:
        params["date"] = date
    return _get("/results", params=params)


def search_horse(name: str) -> dict:
    """Search for a horse's form and profile."""
    return _get("/horses/search", params={"name": name})


# ─── Display helpers ────────────────────────────

def print_us_card(date: str = None):
    """Print a formatted day's US racing card."""
    card = get_us_racecards(date)
    races = card.get("races", [])
    print(f"\n{'='*60}")
    print(f"  US RACING CARD — {date or 'today'} ({len(races)} races)")
    print(f"{'='*60}")
    for race in races:
        print(f"\n{race.get('course', '?')} | {race.get('off_time', '?')} | "
              f"{race.get('distance_f', '?')}f | {race.get('going', '?')}")
        for runner in race.get("runners", []):
            sp = runner.get("sp", "N/A")
            print(f"  {runner.get('number', '?'):>2}. {runner.get('horse', '?'):<30} "
                  f"SP: {sp}")


if __name__ == "__main__":
    print_us_card()
```

---

## 3. Option B — Equibase (Official US Source, ToS Restricted)

Equibase is the official supplier of US Thoroughbred data, backing FanDuel Racing, Breeders' Cup, NYRA, and TwinSpires. Their public pages contain daily entries, results, and downloadable chart PDFs.

**⚠️ Their Terms of Use explicitly state:** *"The use of any robot, spider, scraper or any other automated means to access the contents of this site is prohibited."*

This code is provided for **educational purposes** and personal reference only. Consider requesting data access via their [Marketplace](https://www.equibase.com/products/marketplace.cfm) instead, or use the licensed Equibase Free Dataset (requires registration).

### Known URL Patterns

```
# Today's entries index (static HTML, publicly accessible)
https://www.equibase.com/static/entry/index.html

# Entries for a specific track (e.g. Belmont Park = BEL, USA)
https://www.equibase.com/static/entry/BEL-entries.html

# Full chart results (PDFs)
https://www.equibase.com/static/chart/pdf/index.html

# Summary results index
https://www.equibase.com/static/chart/summary/index.html

# Results for a specific track and date
https://www.equibase.com/premium/eqbPDFChartPlus.cfm?
  RACE=A&BND=&TID=BEL&CTRY=USA&DT=05/10/2026&DAY=D&STYLE=EQB
```

### Scraper — Entries Index

```python
"""
equibase_scraper.py
Scrapes Equibase public entries and results pages.

⚠️  Equibase ToS prohibits automated access.
    This is for personal/educational use only.
    Add generous delays and use sparingly.
"""

import time
import requests
from bs4 import BeautifulSoup
from datetime import date

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.equibase.com/",
}

BASE = "https://www.equibase.com"
DELAY = 4.0  # seconds between requests — be polite


def _get_html(url: str) -> BeautifulSoup:
    """Fetch a page and return a BeautifulSoup object."""
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# ─── Entries ────────────────────────────────────

def get_entry_index() -> list[dict]:
    """
    Scrape the main entries index to find all tracks racing today.

    Returns:
        [{"track": "Belmont Park", "track_code": "BEL",
          "races": 9, "url": "..."}, ...]
    """
    soup = _get_html(f"{BASE}/static/entry/index.html")
    tracks = []

    # The index page lists tracks as links to individual entry pages
    for link in soup.select("a[href*='-entries']"):
        href = link["href"]
        name = link.get_text(strip=True)
        # Extract track code from href: e.g. "/static/entry/BEL-entries.html"
        track_code = href.split("/")[-1].replace("-entries.html", "").upper()
        tracks.append({
            "track": name,
            "track_code": track_code,
            "url": BASE + href if href.startswith("/") else href,
        })

    return tracks


def get_track_entries(track_code: str) -> list[dict]:
    """
    Scrape entries for a specific track.

    Args:
        track_code: e.g. "BEL", "CD", "SA", "KEE"

    Returns list of race dicts with runner information.
    """
    url = f"{BASE}/static/entry/{track_code}-entries.html"
    soup = _get_html(url)
    races = []

    # Each race is typically in a section with a race number heading
    # NOTE: Equibase HTML structure may change — inspect and update selectors as needed
    race_sections = soup.select("div.race-entry, section.race, div[id^='race']")

    if not race_sections:
        # Fallback: try to find table rows with horse data
        print(f"  [warn] Could not find race sections for {track_code} — "
              f"page structure may have changed")
        return _parse_entries_fallback(soup, track_code)

    for section in race_sections:
        race_num = section.select_one("h2, h3, .race-number")
        race_num_text = race_num.get_text(strip=True) if race_num else "?"

        runners = []
        for row in section.select("tr.runner, tr[class*='horse'], tr:has(td)"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) >= 3:
                runners.append({
                    "number":  cells[0] if cells else "",
                    "horse":   cells[1] if len(cells) > 1 else "",
                    "jockey":  cells[2] if len(cells) > 2 else "",
                    "trainer": cells[3] if len(cells) > 3 else "",
                    "weight":  cells[4] if len(cells) > 4 else "",
                })

        if runners:
            races.append({
                "track": track_code,
                "race": race_num_text,
                "runners": runners,
            })

    return races


def _parse_entries_fallback(soup: BeautifulSoup, track_code: str) -> list[dict]:
    """Fallback parser — attempts to extract any horse names from the page."""
    runners = []
    # Look for any table that might contain runner data
    for table in soup.select("table"):
        headers = [th.get_text(strip=True).lower()
                   for th in table.select("th")]
        if any(h in headers for h in ["horse", "runner", "name", "#"]):
            for row in table.select("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.select("td")]
                if cells:
                    runners.append(dict(zip(headers, cells)))
    return [{"track": track_code, "race": "unknown", "runners": runners}]


# ─── Results ────────────────────────────────────

def get_results_index() -> list[dict]:
    """
    Scrape the results summary index to find all tracks with results today.

    Returns list of track dicts with result URLs.
    """
    url = f"{BASE}/static/chart/summary/index.html"
    soup = _get_html(url)
    tracks = []

    for link in soup.select("a[href]"):
        href = link["href"]
        if "chart" in href.lower() or "result" in href.lower():
            tracks.append({
                "track": link.get_text(strip=True),
                "url": BASE + href if href.startswith("/") else href,
            })

    return tracks


def get_track_results(track_code: str, race_date: str = None) -> str:
    """
    Fetch the results chart PDF URL for a track and date.
    Returns the PDF URL — does NOT download the PDF automatically.

    Args:
        track_code: e.g. "BEL", "CD"
        race_date:  "MM/DD/YYYY" — defaults to today

    Returns:
        PDF URL string
    """
    if race_date is None:
        d = date.today()
        race_date = d.strftime("%m/%d/%Y")

    pdf_url = (
        f"{BASE}/premium/eqbPDFChartPlus.cfm"
        f"?RACE=A&BND=&TID={track_code}&CTRY=USA"
        f"&DT={race_date}&DAY=D&STYLE=EQB"
    )
    return pdf_url


# ─── CLI ────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        code = sys.argv[1].upper()
        print(f"\nFetching entries for {code}...")
        races = get_track_entries(code)
        for race in races:
            print(f"\n{race['race']}")
            for r in race['runners']:
                print(f"  {r.get('number','?'):>2}. {r.get('horse','?')}")
    else:
        print("Today's racing cards on Equibase:")
        for t in get_entry_index():
            print(f"  {t['track_code']:<6} {t['track']}")
```

---

## 4. Option C — OddsPortal (Public Odds Aggregator)

OddsPortal aggregates odds from many bookmakers for US horse racing's major events. It does not require login for results/historical data, and has no explicit automated-access ban in its ToS. However, it is heavily JavaScript-rendered — Playwright or Selenium are required.

This is best for **historical odds comparison** and tracking which bookmakers offered what price on a given race.

### Setup

```bash
pip install playwright
playwright install chromium
```

### Scraper

```python
"""
oddsportal_scraper.py
Scrapes US horse racing odds from OddsPortal using Playwright.

Best for: Historical odds, multi-bookmaker comparisons.
Limitation: OddsPortal's horse racing coverage focuses on major US races
(Triple Crown, Breeders' Cup, Graded stakes), not everyday card coverage.

Uses Playwright (headless Chromium) — required for JS-rendered content.
"""

import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright


US_HORSE_RACING_URL = "https://www.oddsportal.com/horse-racing/usa/"

# Known URL patterns for specific US meetings on OddsPortal:
# https://www.oddsportal.com/horse-racing/usa/kentucky-derby-2026/
# https://www.oddsportal.com/horse-racing/usa/breeders-cup-2025/
# https://www.oddsportal.com/horse-racing/usa/preakness-stakes-2026/
# https://www.oddsportal.com/horse-racing/usa/belmont-stakes-2026/


async def scrape_us_races_list() -> list[dict]:
    """
    Scrape the OddsPortal US horse racing index to find available events.

    Returns:
        [{"name": "Kentucky Derby 2026", "url": "https://..."}, ...]
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        await page.goto(US_HORSE_RACING_URL, wait_until="networkidle", timeout=30000)

        # Wait for the events list to load
        await page.wait_for_timeout(3000)

        # Extract event links from the page
        events = await page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[href*="/horse-racing/usa/"]'));
                return links
                    .filter(a => a.href !== window.location.href)
                    .map(a => ({
                        name: a.innerText.trim(),
                        url: a.href
                    }))
                    .filter(e => e.name.length > 0);
            }
        """)

        await browser.close()
        return events


async def scrape_race_odds(race_url: str) -> dict:
    """
    Scrape odds from a specific OddsPortal race page.

    Args:
        race_url: Full URL to the race on OddsPortal

    Returns dict with race info and bookmaker odds per runner.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        print(f"  Loading: {race_url}")
        await page.goto(race_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(4000)  # Let JS fully render

        # Extract the odds table
        result = await page.evaluate("""
            () => {
                const rows = Array.from(document.querySelectorAll('tr'));
                const data = [];

                for (const row of rows) {
                    const cells = Array.from(row.querySelectorAll('td'));
                    if (cells.length < 2) continue;
                    const rowData = cells.map(c => c.innerText.trim());
                    if (rowData[0]) data.push(rowData);
                }

                // Also try to get bookmaker headers
                const headers = Array.from(
                    document.querySelectorAll('th, .bookmaker-name')
                ).map(h => h.innerText.trim()).filter(Boolean);

                return { rows: data, headers };
            }
        """)

        # Get page title for race name
        title = await page.title()

        await browser.close()

        return {
            "race": title,
            "url": race_url,
            "headers": result.get("headers", []),
            "rows": result.get("rows", []),
        }


async def scrape_multiple_races(race_urls: list[str], delay: float = 5.0) -> list[dict]:
    """
    Scrape odds for multiple races with a delay between each.

    Args:
        race_urls: List of OddsPortal race URLs
        delay:     Seconds to wait between scrapes (default 5s)
    """
    results = []
    for url in race_urls:
        data = await scrape_race_odds(url)
        results.append(data)
        await asyncio.sleep(delay)
    return results


def save_to_json(data: list[dict], filename: str = "oddsportal_us.json"):
    """Save scraped data to a JSON file."""
    Path(filename).write_text(json.dumps(data, indent=2))
    print(f"Saved {len(data)} races to {filename}")


# ─── Convenience sync wrappers ──────────────────

def get_us_race_list() -> list[dict]:
    return asyncio.run(scrape_us_races_list())


def get_race_odds(url: str) -> dict:
    return asyncio.run(scrape_race_odds(url))


# ─── CLI ────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"Scraping odds from: {url}")
        data = get_race_odds(url)
        print(f"Race: {data['race']}")
        for row in data["rows"][:20]:
            print("  ", " | ".join(row))
    else:
        print("Fetching US horse racing event list from OddsPortal...")
        events = get_us_race_list()
        for e in events[:20]:
            print(f"  {e['name']:<50} {e['url']}")
```

---

## 5. Option D — NYRA / Track Websites (Public Entries & Results)

NYRA (New York Racing Association) runs Aqueduct, Belmont Park, and Saratoga. Their website publishes free entries and results with minimal anti-scraping. This option gives you real-time, same-day data for NYRA tracks without a paid API.

### NYRA Known URL Patterns

```
# Today's entries
https://www.nyra.com/belmont/racing/entries/
https://www.nyra.com/aqueduct/racing/entries/
https://www.nyra.com/saratoga/racing/entries/

# Results
https://www.nyra.com/belmont/racing/results/
```

### NYRA Scraper

```python
"""
nyra_scraper.py
Scrapes NYRA track websites for entries and results.
NYRA covers: Belmont Park (BEL), Aqueduct (AQU), Saratoga (SAR).

NYRA pages are JavaScript-rendered — Playwright required.
The pages do not appear to have an explicit anti-scraping clause.
Be respectful with delays.
"""

import asyncio
import json
from playwright.async_api import async_playwright

TRACKS = {
    "BEL": "https://www.nyra.com/belmont",
    "AQU": "https://www.nyra.com/aqueduct",
    "SAR": "https://www.nyra.com/saratoga",
}


async def get_nyra_entries(track_code: str) -> list[dict]:
    """
    Scrape entries from a NYRA track.

    Args:
        track_code: "BEL", "AQU", or "SAR"

    Returns list of race dicts.
    """
    if track_code not in TRACKS:
        raise ValueError(f"Unknown NYRA track: {track_code}. Use: {list(TRACKS)}")

    url = f"{TRACKS[track_code]}/racing/entries/"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)

        # Extract race and runner data from the rendered page
        races = await page.evaluate("""
            () => {
                const raceData = [];

                // NYRA uses accordion-style race sections
                const raceSections = document.querySelectorAll(
                    '[class*="race"], [class*="Race"], section, article'
                );

                for (const section of raceSections) {
                    const raceNum = section.querySelector(
                        '[class*="race-number"], [class*="raceNum"], h2, h3'
                    );
                    const runners = [];

                    const rows = section.querySelectorAll('tr, [class*="runner"]');
                    for (const row of rows) {
                        const cells = row.querySelectorAll('td, [class*="cell"]');
                        if (cells.length >= 2) {
                            runners.push({
                                number:  cells[0]?.innerText?.trim() || '',
                                horse:   cells[1]?.innerText?.trim() || '',
                                jockey:  cells[2]?.innerText?.trim() || '',
                                trainer: cells[3]?.innerText?.trim() || '',
                                ml_odds: cells[4]?.innerText?.trim() || '',
                            });
                        }
                    }

                    if (runners.length > 0) {
                        raceData.push({
                            race: raceNum?.innerText?.trim() || 'Unknown',
                            runners
                        });
                    }
                }

                return raceData;
            }
        """)

        # Fallback: dump all visible text structured by headings
        if not races:
            races = await page.evaluate("""
                () => {
                    return [{
                        race: "raw_dump",
                        raw_text: document.body.innerText.substring(0, 5000)
                    }]
                }
            """)

        await browser.close()

    # Tag each race with the track code
    for race in races:
        race["track"] = track_code

    return races


async def get_all_nyra_entries() -> dict[str, list]:
    """Fetch entries from all three NYRA tracks (sequential with delay)."""
    results = {}
    for code in TRACKS:
        print(f"  Fetching {code}...")
        results[code] = await get_nyra_entries(code)
        await asyncio.sleep(4)
    return results


def get_nyra(track_code: str) -> list[dict]:
    """Synchronous wrapper."""
    return asyncio.run(get_nyra_entries(track_code))


if __name__ == "__main__":
    import sys
    code = sys.argv[1].upper() if len(sys.argv) > 1 else "BEL"
    print(f"NYRA entries — {code}")
    races = get_nyra(code)
    for race in races:
        print(f"\n{race.get('race', '?')}")
        for r in race.get("runners", []):
            print(f"  {r.get('number','?'):>2}. {r.get('horse','?')}")
```

---

## 6. Option E — Betfair Exchange API (US Tracks)

Betfair's official API is free with an account and provides live market prices, back/lay odds, and trading volume for US horse racing. This is the most reliable option for **live odds** — it's a proper API with no scraping, generous rate limits, and documented endpoints.

Betfair's Exchange covers all major US meetings (Kentucky Derby, Triple Crown races, Breeders' Cup, and many graded stakes), and some everyday US card events.

### Setup

```bash
pip install betfairlightweight
```

Create a Betfair account, enable API access, and download your SSL certificates from the [Betfair Developer Programme](https://developer.betfair.com/).

```
# .env additions
BETFAIR_USERNAME=your@email.com
BETFAIR_PASSWORD=your_password
BETFAIR_APP_KEY=your_app_key
BETFAIR_CERT_PATH=./certs/betfair.crt
BETFAIR_KEY_PATH=./certs/betfair.key
```

### Client

```python
"""
betfair_client.py
Betfair Exchange API client for US horse racing.

Free official API — no scraping required.
Covers major US meetings; everyday card coverage varies by season.

Requires: pip install betfairlightweight
Requires: Betfair account + API credentials + SSL certs
"""

import os
import betfairlightweight
from betfairlightweight import filters
from dotenv import load_dotenv

load_dotenv()

# Horse Racing event type ID on Betfair
HORSE_RACING_TYPE_ID = "7"

# US country code on Betfair
US_COUNTRY = "US"


def create_client():
    """Create and return an authenticated Betfair API client."""
    return betfairlightweight.APIClient(
        username=os.getenv("BETFAIR_USERNAME"),
        password=os.getenv("BETFAIR_PASSWORD"),
        app_key=os.getenv("BETFAIR_APP_KEY"),
        certs=os.getenv("BETFAIR_CERT_PATH", "./certs"),
    )


# ─── Markets ────────────────────────────────────

def get_us_horse_racing_markets(client) -> list:
    """
    Fetch all available US horse racing markets.

    Returns list of MarketCatalogue objects.
    """
    market_filter = filters.market_filter(
        event_type_ids=[HORSE_RACING_TYPE_ID],
        market_countries=[US_COUNTRY],
        market_type_codes=["WIN"],  # Win markets only
    )

    market_catalogue_filter = filters.market_projection(
        ["COMPETITION", "EVENT", "EVENT_TYPE", "RUNNER_DESCRIPTION",
         "MARKET_START_TIME", "MARKET_DESCRIPTION"]
    )

    return client.betting.list_market_catalogue(
        filter=market_filter,
        market_projection=market_catalogue_filter,
        max_results=100,
    )


def get_market_odds(client, market_id: str) -> dict:
    """
    Fetch live back/lay odds for a specific market.

    Args:
        market_id: Betfair market ID from get_us_horse_racing_markets()

    Returns dict with runners and their best available back odds.
    """
    price_filter = filters.price_projection(
        price_data=["EX_BEST_OFFERS"],
    )

    books = client.betting.list_market_book(
        market_ids=[market_id],
        price_projection=price_filter,
    )

    if not books:
        return {}

    book = books[0]
    runners = {}
    for runner in book.runners:
        best_back = None
        best_lay = None
        if runner.ex:
            backs = runner.ex.available_to_back
            lays = runner.ex.available_to_lay
            best_back = backs[0].price if backs else None
            best_lay = lays[0].price if lays else None

        runners[runner.selection_id] = {
            "status": runner.status,
            "best_back": best_back,
            "best_lay": best_lay,
            "last_price_traded": runner.last_price_traded,
        }

    return {
        "market_id": market_id,
        "status": book.status,
        "total_matched": book.total_matched,
        "runners": runners,
    }


def get_full_us_card(client) -> list[dict]:
    """
    Fetch all US horse racing markets with their runner names and live odds.
    Combines catalogue data (names) with book data (prices).

    Returns list of race dicts ready for display or storage.
    """
    markets = get_us_horse_racing_markets(client)
    results = []

    for market in markets:
        # Build a map of selection_id → runner name
        name_map = {
            r.selection_id: r.runner_name
            for r in market.runners
        }

        market_id = market.market_id
        odds = get_market_odds(client, market_id)

        runners_out = []
        for sel_id, name in name_map.items():
            runner_odds = odds.get("runners", {}).get(sel_id, {})
            runners_out.append({
                "name":           name,
                "selection_id":   sel_id,
                "best_back":      runner_odds.get("best_back"),
                "best_lay":       runner_odds.get("best_lay"),
                "last_traded":    runner_odds.get("last_price_traded"),
                "status":         runner_odds.get("status"),
            })

        results.append({
            "market_id":    market_id,
            "market_name":  market.market_name,
            "event":        market.event.name if market.event else "",
            "competition":  market.competition.name if market.competition else "",
            "start_time":   str(market.market_start_time),
            "runners":      runners_out,
        })

    return results


# ─── Display ────────────────────────────────────

def print_us_card(client):
    """Print a formatted US horse racing card with live odds."""
    races = get_full_us_card(client)
    print(f"\n{'='*70}")
    print(f"  BETFAIR EXCHANGE — US HORSE RACING ({len(races)} markets)")
    print(f"{'='*70}")

    for race in sorted(races, key=lambda r: r["start_time"]):
        print(f"\n{race['event']:<30} {race['market_name']}")
        print(f"  Start: {race['start_time']}")
        print(f"  {'Runner':<30} {'Back':>8} {'Lay':>8} {'Last':>8}")
        print(f"  {'-'*56}")
        for r in sorted(race["runners"],
                        key=lambda x: x.get("best_back") or 999):
            back = f"{r['best_back']:.2f}" if r.get("best_back") else "  N/A"
            lay  = f"{r['best_lay']:.2f}"  if r.get("best_lay")  else "  N/A"
            last = f"{r['last_traded']:.2f}" if r.get("last_traded") else "  N/A"
            print(f"  {r['name']:<30} {back:>8} {lay:>8} {last:>8}")


# ─── CLI ────────────────────────────────────────

if __name__ == "__main__":
    client = create_client()
    client.login()
    print_us_card(client)
    client.logout()
```

---

## 7. Combining Sources — Recommended Architecture

For a complete US racing workflow, combine sources based on what each does best:

```
┌─────────────────────────────────────────────────────┐
│             US HORSE RACING DATA PIPELINE           │
├─────────────┬───────────────┬───────────────────────┤
│  Pre-Race   │  Live/Odds    │  Post-Race Results    │
├─────────────┼───────────────┼───────────────────────┤
│ Option A    │ Option E      │ Option A or B         │
│ Racing API  │ Betfair       │ Racing API / Equibase │
│ (licensed)  │ (free API)    │                       │
│             │               │                       │
│ OR          │ OR            │                       │
│             │               │                       │
│ Option B    │ Option C      │                       │
│ Equibase    │ OddsPortal    │                       │
│ (ToS risk)  │ (scrape)      │                       │
└─────────────┴───────────────┴───────────────────────┘
```

### Unified Runner (`main.py`)

```python
"""
us_racing_main.py
Unified CLI for US horse racing data.

Usage:
    python us_racing_main.py betfair             # Live Betfair markets
    python us_racing_main.py nyra BEL            # NYRA entries (Belmont)
    python us_racing_main.py nyra AQU            # NYRA entries (Aqueduct)
    python us_racing_main.py equibase BEL        # Equibase entries (⚠️ ToS risk)
    python us_racing_main.py oddsportal          # OddsPortal US race list
    python us_racing_main.py racingapi           # The Racing API card
"""

import sys


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0].lower()

    if cmd == "betfair":
        from betfair_client import create_client, print_us_card
        client = create_client()
        client.login()
        try:
            print_us_card(client)
        finally:
            client.logout()

    elif cmd == "nyra":
        from nyra_scraper import get_nyra
        track = args[1].upper() if len(args) > 1 else "BEL"
        races = get_nyra(track)
        for race in races:
            print(f"\n{race.get('race','?')} — {race.get('track','?')}")
            for r in race.get("runners", []):
                print(f"  {r.get('number',''):>2}. {r.get('horse','?')}")

    elif cmd == "equibase":
        print("⚠️  Equibase ToS restricts automated access — personal use only")
        from equibase_scraper import get_track_entries, get_entry_index
        if len(args) > 1:
            races = get_track_entries(args[1].upper())
            for race in races:
                print(f"\n{race['race']}")
                for r in race.get("runners", []):
                    print(f"  {r.get('number',''):>2}. {r.get('horse','?')}")
        else:
            for t in get_entry_index():
                print(f"  {t['track_code']:<6} {t['track']}")

    elif cmd == "oddsportal":
        from oddsportal_scraper import get_us_race_list
        events = get_us_race_list()
        print(f"\nUS horse racing on OddsPortal ({len(events)} events):")
        for e in events:
            print(f"  {e['name']}")

    elif cmd == "racingapi":
        from racing_api_client import print_us_card
        print_us_card()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
```

---

## 8. Shared Utilities

### requirements.txt (full project)

```
# Core
requests>=2.31.0
python-dotenv>=1.0.0

# HTML parsing (Equibase, static sites)
beautifulsoup4>=4.12.0
lxml>=5.0.0

# JavaScript-rendered sites (OddsPortal, NYRA)
playwright>=1.40.0

# Betfair Exchange API
betfairlightweight>=2.20.0

# Data handling (optional)
pandas>=2.0.0
```

```bash
pip install -r requirements.txt
playwright install chromium
```

### Odds Format Converters

```python
"""
odds_utils.py
Odds format conversion utilities.
"""


def decimal_to_american(decimal: float) -> str:
    """2.50 → '+150', 1.50 → '-200'"""
    if decimal >= 2.0:
        return f"+{int((decimal - 1) * 100)}"
    else:
        return f"-{int(100 / (decimal - 1))}"


def american_to_decimal(american: int) -> float:
    """'+150' → 2.50, '-200' → 1.50"""
    if american > 0:
        return round(1 + american / 100, 4)
    else:
        return round(1 + 100 / abs(american), 4)


def decimal_to_fractional(decimal: float) -> str:
    """2.50 → '3/2', 3.00 → '2/1'"""
    from fractions import Fraction
    frac = Fraction(decimal - 1).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"


def implied_probability(decimal: float) -> float:
    """Decimal odds → implied win probability as percentage."""
    return round(100 / decimal, 2)


def each_way_return(stake: float, decimal: float, place_fraction: int,
                    places: int, finishing_pos: int) -> float:
    """
    Calculate each-way return.

    Args:
        stake:          Stake per part (total bet = stake * 2)
        decimal:        Win decimal odds
        place_fraction: Denominator of place fraction (e.g. 4 = 1/4 odds)
        places:         Number of places paid (e.g. 3)
        finishing_pos:  Horse's finishing position (1-based)

    Returns total return (0 if unplaced).
    """
    win_return = 0.0
    place_return = 0.0
    place_odds = 1 + (decimal - 1) / place_fraction

    if finishing_pos == 1:
        win_return = stake * decimal
        place_return = stake * place_odds
    elif finishing_pos <= places:
        place_return = stake * place_odds

    return round(win_return + place_return, 2)
```

### Data Storage

```python
"""
storage.py
SQLite-based storage for scraped US race data.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "us_racing.db"


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS races (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at  TEXT,
            source      TEXT,
            track       TEXT,
            race_name   TEXT,
            race_date   TEXT,
            market_id   TEXT
        );

        CREATE TABLE IF NOT EXISTS runners (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id     INTEGER REFERENCES races(id),
            horse       TEXT,
            number      TEXT,
            jockey      TEXT,
            trainer     TEXT,
            ml_odds     TEXT,
            decimal_sp  TEXT
        );

        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            runner_id   INTEGER REFERENCES runners(id),
            snapped_at  TEXT,
            source      TEXT,
            back_odds   REAL,
            lay_odds    REAL
        );
    """)
    con.commit()
    con.close()


def save_race(source: str, track: str, race_name: str,
              runners: list[dict], race_date: str = None) -> int:
    """Insert a race and its runners. Returns the race ID."""
    con = sqlite3.connect(DB_PATH)
    now = datetime.utcnow().isoformat()
    cur = con.execute(
        "INSERT INTO races (fetched_at, source, track, race_name, race_date) "
        "VALUES (?,?,?,?,?)",
        (now, source, track, race_name, race_date or now[:10])
    )
    race_id = cur.lastrowid
    for r in runners:
        con.execute(
            "INSERT INTO runners (race_id, horse, number, jockey, trainer, ml_odds) "
            "VALUES (?,?,?,?,?,?)",
            (race_id,
             r.get("horse", ""),
             r.get("number", ""),
             r.get("jockey", ""),
             r.get("trainer", ""),
             r.get("ml_odds", ""))
        )
    con.commit()
    con.close()
    return race_id


def save_odds_snapshot(runner_id: int, source: str,
                       back: float = None, lay: float = None):
    """Record a point-in-time odds snapshot for a runner."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO odds_snapshots (runner_id, snapped_at, source, back_odds, lay_odds) "
        "VALUES (?,?,?,?,?)",
        (runner_id, datetime.utcnow().isoformat(), source, back, lay)
    )
    con.commit()
    con.close()
```

---

## 9. US Track Code Reference

### Major Tracks and Their Codes

| Track | Code | State | Season | Notable Events |
|---|---|---|---|---|
| Belmont Park / Big A | BEL | New York | Apr–Jun, Sep–Nov | Belmont Stakes (Jun) |
| Aqueduct | AQU | New York | Nov–Apr | Wood Memorial |
| Saratoga | SAR | New York | Late Jul–Sep | Travers Stakes |
| Churchill Downs | CD | Kentucky | Apr–Jul, Oct–Nov | Kentucky Derby (May) |
| Keeneland | KEE | Kentucky | Apr, Oct | Blue Grass Stakes |
| Pimlico | PIM | Maryland | Apr–Jun | Preakness Stakes (May) |
| Santa Anita | SA | California | Oct–Jun | Santa Anita Derby |
| Del Mar | DMR | California | Jul–Sep, Nov | Pacific Classic |
| Gulfstream Park | GP | Florida | Oct–Apr | Florida Derby |
| Oaklawn Park | OP | Arkansas | Jan–May | Arkansas Derby |
| Fair Grounds | FG | Louisiana | Nov–Mar | Louisiana Derby |
| Turfway Park | TP | Kentucky | Sep–Apr | — |
| Tampa Bay Downs | TAM | Florida | Nov–May | Tampa Bay Derby |
| Monmouth Park | MTH | New Jersey | May–Sep | Haskell S. |
| Prairie Meadows | PRM | Iowa | Apr–Oct | Iowa Derby |
| Remington Park | RP | Oklahoma | Apr–Nov | — |
| Lone Star Park | LS | Texas | Apr–Jun, Sep–Oct | — |

### Triple Crown Dates (Approximate)

| Race | Track | Typical Date |
|---|---|---|
| Kentucky Derby | Churchill Downs | First Saturday in May |
| Preakness Stakes | Pimlico | Third Saturday in May |
| Belmont Stakes | Belmont Park | First or second Saturday in June |

---

*Document compiled May 2026. Web structures change — test selectors before relying on them in production. Always check the ToS of any site before scraping.*
