# Odds Scraping Repos Analysis

> **Date**: 2026-04-13  
> **Repos reviewed**: [J-Ack0/horse-racing-ml](https://github.com/J-Ack0/horse-racing-ml) and [anupa-perera/rc-19-arb-engine](https://github.com/anupa-perera/rc-19-arb-engine)  
> **Goal**: Determine if either repo's odds-scraping methodology can be incorporated into this project to get live bookmaker odds (and any extra data).

---

## TL;DR — Recommendation

| | J-Ack0/horse-racing-ml | anupa-perera/rc-19-arb-engine |
|---|---|---|
| **Verdict** | **ADOPT — high priority** | **ADOPT selectively — odds only** |
| Source | Racing Post (racingpost.com) | At The Races (attheraces.com) |
| Method | BS4 + Selenium (Python) | Playwright + stealth plugin (Node.js) |
| Odds coverage | Single bookmaker (RP live) + 4 historical snapshots + trend | **Multiple bookmakers** via ATR odds grid |
| Extra data | Racecards, results, form, going, rating | Race menu only (no results/form) |
| Integration effort | Low — Python, same stack | Medium — TypeScript/Bun, needs a Python wrapper |
| Region | UK & Ireland | UK & Ireland (+ greyhounds) |

**Best strategy**: Use **Racing Post scraper** (J-Ack0) for racecards + single-bookie odds + results auditing. Use **ATR scraper** (rc-19-arb-engine) for **multi-bookmaker odds** which enables arbitrage detection and better value betting. Both scrape from free, public-facing sites with no API key required.

---

## Repo 1: J-Ack0/horse-racing-ml

### Overview
A full end-to-end horse racing prediction system: scraping → feature engineering → XGBoost model → Kelly Criterion betting → results auditing. ~95K historical races, AUC-ROC 0.737. Active daily use for over a year.

### Data Sources & Coverage

| Dimension | Coverage |
|---|---|
| **Region** | UK & Ireland (all Racing Post racecards) |
| **Date range** | Today/tomorrow (racecards) + any historical date (results) |
| **Odds source** | Racing Post racecard pages — one bookmaker price + 4 historical price snapshots |
| **Results** | Actual finishing positions scraped from RP results pages |
| **Extra data** | Track, going, distance, time, jockey, trainer, weight, age, rating, claims |

### Scraping Architecture

Three standalone Python scripts:

1. **`scrape_racecards.py`** — BS4 only (no Selenium needed)
   - Source: `https://www.racingpost.com/racecards/` (tomorrow by default)
   - Outputs: `Inference_Inputs/Input_YYYY-MM-DD.csv`
   - Fields: `track_name, race_date, distance, going, time, horse_name, rating, jockey, trainer, weight, age, claims`

2. **`scrape_odds.py`** — BS4 for links + Selenium for JS-rendered odds
   - Source: `https://www.racingpost.com/racecards/tomorrow/`
   - Outputs: `Inference_Odds/Odds_YYYY-MM-DD.csv`
   - Fields: `URL, horse_name, odds_live, odds_1, odds_2, odds_3, odds_4, trend`
   - Trend: `shortening`, `drifting`, `stable`, or `no_data` (based on 5% threshold)

3. **`scrape_results.py`** — BS4 only
   - Source: `https://www.racingpost.com/results/YYYY-MM-DD/` (or racecards → results URL swap)
   - Outputs: `Inference_Actuals/YYYY_MM_DD_Actual.csv` + `Inference_Summary/YYYY_MM_DD_Summary_Log.csv`
   - Fields: `track_name, time, horse_name, actual_position`

### Key CSS Selectors (current as of March 2026)

```python
# Racing Post selectors used across all three scripts
SELECTORS = {
    # Racecard page
    'race_link':          'a[data-race-is-over]',
    'course_name':        '[data-test-selector="RC-courseHeader__name"]',
    'course_date':        '[data-test-selector="RC-courseHeader__date"]',
    'course_time':        '[data-test-selector="RC-courseHeader__time"]',
    'going':              '[data-test-selector="RC-headerBox__going"]',
    'distance':           '[data-test-selector="RC-header__raceDistanceRound"]',
    'runner_row':         '.RC-runnerRow',
    'runner_name':        '[data-test-selector="RC-cardPage-runnerName"]',
    'runner_jockey':      '[data-test-selector="RC-cardPage-runnerJockey-name"]',
    'runner_trainer':     '[data-test-selector="RC-cardPage-runnerTrainer-name"]',
    'runner_weight':      '[data-test-selector="RC-cardPage-runnerWgt-carried"]',
    'runner_age':         '[data-test-selector="RC-cardPage-runnerAge"]',
    'runner_rating':      '[data-test-selector="RC-cardPage-runnerRpr"]',
    'runner_claim':       '[data-test-selector="RC-cardPage-runnerJockey-allowance"]',
    # Odds (Selenium-rendered)
    'runner_price':       '[data-test-selector="RC-cardPage-runnerPrice"]',
    'history_price_N':    '[data-test-selector="RC-historyPrices-item-{i}"]',  # i=1..4
    # Results page
    'result_row':         'tr.rp-horseTable__mainRow[data-test-selector="table-row"]',
    'result_horse_name':  'a[data-test-selector="link-horseName"]',
    'result_position':    'span[data-test-selector="text-horsePosition"]',
}
```

### Feature Engineering (20 features — different from ours)

| Feature | Description | We have it? |
|---|---|---|
| `EMA_Form` | Exponential moving average of finish positions | No — we use `recent_form_score` |
| `Recent_win_rate` | Win rate over last 5 races | Yes — `win_rate_last5` |
| `Career_win_rate` | Lifetime win % | Yes — `career_win_rate` |
| `Wilson_score` | Confidence-adjusted J/T win rate | No — worth adding |
| `jt_runs` / `jt_wins` | Jockey-trainer partnership stats | Yes — `jtcombo_*` features |
| `Track_win_rate` | Horse win rate at specific venue | Yes — `course_win_rate` |
| `Performance_on_going` | Win rate on current ground | Yes — `going_pref_*` features |
| Age bins | Young/prime/veteran | Yes — we encode age |
| `days_since_last` | Freshness | Yes |
| Odds trend | Shortening/drifting/stable | **No — high value addition** |

### Betting System

- **Value identification**: edge = model_prob − implied_prob; min edge 1%, min EV 1%
- **Stake sizing**: Quarter Kelly (`0.25 * kelly * bankroll`)
- **Odds trend adjustment**: shortening → +2% edge bonus, drifting → −2% penalty
- **Lucky 15**: Evaluates 4-leg combos from top candidates, Sharpe-filtered

---

## Repo 2: anupa-perera/rc-19-arb-engine

### Overview
A real-time arbitrage detection engine for horse racing. Scrapes live odds from **multiple bookmakers** via At The Races (ATR), identifies >100% payout arbitrage opportunities, and pushes updates over WebSockets. TypeScript/Bun/ElysiaJS/Redis/Playwright stack.

### Data Sources & Coverage

| Dimension | Coverage |
|---|---|
| **Region** | UK & Ireland (+ greyhounds via `greyhounds.attheraces.com`) |
| **Date range** | Today + tomorrow (live only, no historical) |
| **Odds source** | ATR odds grid — **multiple bookmakers** in a single scrape |
| **Bookmakers** | Varies by race but typically 6-10: Bet365, SkyBet, Ladbrokes, Coral, William Hill, Paddy Power, Betfair Sportsbook, etc. |
| **Extra data** | Venue, race time, race URL — no form/going/weight data |
| **Update frequency** | Configurable polling interval (default 10s) |

### Scraping Architecture

Playwright-based (Node.js workers spawned from Bun server):

1. **`scraper-atr-worker.js`** — Fetches daily race menu from ATR
   - Source: `https://www.attheraces.com/racecards` (today) and `/racecards/tomorrow`
   - Extracts: venue name, race times, odds page URLs
   - URL pattern: `/racecard/{Venue}/{DD-Month-YYYY}/{HHmm}/odds`

2. **`scraper-atr-odds-worker.js`** — Fetches odds grid for a single race
   - Source: Individual ATR racecard/odds pages
   - Extracts: runner names, prices from **all bookmakers** in the ATR odds comparison grid
   - CSS selectors: `.odds-grid__row-wrapper--entries`, `.card-entry`, `.odds-grid__cell--odds`

3. **`stream-odds-worker.js`** — Continuous poller for live odds updates
   - Spawns the odds worker on an interval (default 10s)
   - Pipes JSON to stdout for upstream consumption

### Key CSS Selectors (ATR)

```javascript
// ATR odds grid selectors (current as of Feb 2026)
const ATR_SELECTORS = {
    // Menu page
    racecard_link:        'a[href*="/racecard/"]',
    // Odds page
    odds_grid_entries:    '.odds-grid__row-wrapper--entries',
    card_entry:           '.card-entry',
    horse_link:           'a.horse__link',
    runner_row:           '.odds-grid__row--horse',
    bookmaker_header:     '.odds-grid__cell--bookmaker a.bookmaker-logo',
    bookmaker_name:       '.bookmaker-logo__inner',
    odds_cell:            '.odds-grid__cell--odds',
    odds_link:            'a.odds-grid-link',
    odds_value_decimal:   '.odds-value--decimal',
    odds_value:           '.odds-value',
    cookie_accept:        'button.cky-btn-accept',
};
```

### Output Schema

```json
{
  "runners": [
    {
      "name": "Horse Name",
      "prices": [
        { "bookie": "Bet365", "price": "3.50" },
        { "bookie": "SkyBet", "price": "4.00" },
        { "bookie": "Ladbrokes", "price": "3.75" }
      ]
    }
  ],
  "bookies": ["Bet365", "SkyBet", "Ladbrokes"],
  "url": "https://www.attheraces.com/racecard/...",
  "timestamp": 1713052800000
}
```

### Anti-Detection Measures
- `puppeteer-extra-plugin-stealth` (evades headless browser detection)
- Randomized viewport sizes (`1920±50 × 1080±50`)
- Human-like delays (`randomInt(1000, 3000)`)
- `en-GB` locale, `Europe/London` timezone
- Cookie consent auto-accept
- Optional residential proxy support (`SCRAPER_PROXY` env var)

---

## Comparison: Our Current Odds vs. What We'd Gain

| Feature | Current (The Odds API) | + Racing Post | + ATR Multi-Bookie |
|---|---|---|---|
| API key required | Yes (`ODDS_API_KEY`) | **No** | **No** |
| Monthly limit | 500 calls | **Unlimited** | **Unlimited** |
| Bookmaker count | Multiple (via API) | 1 (RP live) + 4 history | **6-10 per race** |
| Odds trend | No | **Yes** (shortening/drifting) | Computing from snapshots |
| Horse racing coverage | Limited availability | UK/IRE full | UK/IRE full |
| Historical odds | No | 4 price snapshots | No (real-time only) |
| Results auditing | No | **Yes** (RP results) | No |
| Requires browser | No | Yes (Selenium) | Yes (Playwright) |
| Rate limit risk | Paid, metered | Polite delays needed | Stealth + delays |

---

## Implementation Plan

### Phase 1: Racing Post Odds + Racecards + Results (Python, low effort)

This is the highest-value addition because it gives us: (a) free unlimited odds, (b) odds trend signals, (c) next-day results for model auditing, (d) racecard data that complements our Racing API data.

#### 1A. Racing Post Odds Scraper

Create `scripts/scrape_rp_odds.py`:

```python
#!/usr/bin/env python3
"""Scrape live odds + historical price snapshots from Racing Post.

Uses Selenium for JS-rendered odds, BS4 for everything else.
Outputs: data/raw/rp_odds_YYYY-MM-DD.csv

Usage:
    python scripts/scrape_rp_odds.py                   # tomorrow
    python scripts/scrape_rp_odds.py --date 2026-04-14 # specific date
"""

import argparse
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def clean_text(text):
    """Collapse whitespace and strip."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.racingpost.com/',
}


# ──────────────────────────────────────────────
#  Step 1: Get race URLs (BS4 — no Selenium)
# ──────────────────────────────────────────────

def get_race_urls(date_str=None):
    """Fetch all racecard URLs for a given date from Racing Post.

    Args:
        date_str: YYYY-MM-DD or None for tomorrow.

    Returns:
        List of full racecard URLs.
    """
    if date_str:
        url = f"https://www.racingpost.com/racecards/{date_str}/"
    else:
        url = "https://www.racingpost.com/racecards/tomorrow/"

    base_url = "https://www.racingpost.com"

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', attrs={"data-race-is-over": True})

            results = []
            for a in links:
                href = a.get('href')
                if href and "/racecards/" in href:
                    full = href if href.startswith('http') else base_url + href
                    results.append(full)

            unique = list(dict.fromkeys(results))
            print(f"Found {len(unique)} race URLs for {date_str or 'tomorrow'}")
            return unique

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    return []


# ──────────────────────────────────────────────
#  Step 2: Odds trend calculation
# ──────────────────────────────────────────────

def fractional_to_decimal(frac_str):
    """Convert fractional odds string to decimal. Returns None on failure."""
    if not frac_str or pd.isna(frac_str):
        return None
    s = str(frac_str).strip()
    if s.lower() == 'evens':
        return 2.0
    try:
        if '/' in s:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1]) + 1.0
        return float(s)
    except Exception:
        return None


def calculate_trend(odds_list):
    """Classify odds movement as shortening, drifting, stable, or no_data.

    Compares first vs last valid decimal price. >5% change = trend.
    """
    valid = []
    for o in odds_list:
        dec = fractional_to_decimal(o)
        if dec is not None:
            valid.append(dec)

    if len(valid) < 2:
        return "no_data"

    change_pct = ((valid[-1] - valid[0]) / valid[0]) * 100

    if change_pct < -5:
        return "shortening"
    elif change_pct > 5:
        return "drifting"
    else:
        return "stable"


# ──────────────────────────────────────────────
#  Step 3: Scrape odds per race (Selenium)
# ──────────────────────────────────────────────

def scrape_race_odds(driver, url):
    """Scrape odds data from a single RP racecard using Selenium.

    Returns list of dicts with horse_name, odds_live, odds_1..4, trend, track_name, time.
    """
    try:
        driver.get(url)

        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "RC-runnerRow")))
        time.sleep(2.5)  # Wait for dynamic JS odds to populate

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        main = soup.select_one('[data-site-sub-section-1="Races"]')
        if not main:
            return []

        # Extract race header
        track = clean_text(getattr(
            main.select_one('[data-test-selector="RC-courseHeader__name"]'), 'text', ''))
        race_time = clean_text(getattr(
            main.select_one('[data-test-selector="RC-courseHeader__time"]'), 'text', ''))

        rows = []
        for container in main.select('.RC-runnerRow'):
            name_el = container.select_one(
                '[data-test-selector="RC-cardPage-runnerName"]')
            name = clean_text(name_el.text) if name_el else ""
            if not name:
                continue

            # Live odds
            price_el = container.select_one(
                '[data-test-selector="RC-cardPage-runnerPrice"]')
            odds_live = clean_text(price_el.text) if price_el else ""

            # Historical price snapshots (1-4)
            odds_history = []
            for i in range(1, 5):
                el = container.select_one(
                    f'[data-test-selector="RC-historyPrices-item-{i}"]')
                odds_history.append(clean_text(el.text) if el else "")

            trend = calculate_trend([odds_live] + odds_history)

            rows.append({
                "track_name": track,
                "time": race_time,
                "horse_name": name,
                "odds_live": odds_live,
                "odds_1": odds_history[0],
                "odds_2": odds_history[1],
                "odds_3": odds_history[2],
                "odds_4": odds_history[3],
                "odds_trend": trend,
                "odds_decimal": fractional_to_decimal(odds_live),
                "source_url": url,
            })

        print(f"  Scraped {len(rows)} runners from {track} {race_time}")
        return rows

    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return []


# ──────────────────────────────────────────────
#  Step 4: Main pipeline
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape Racing Post odds")
    parser.add_argument('--date', help='YYYY-MM-DD (default: tomorrow)')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run Chrome headless (default)')
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"RACING POST ODDS SCRAPER — {target_date}")
    print(f"{'='*60}\n")

    # 1. Get race URLs
    race_urls = get_race_urls(target_date)
    if not race_urls:
        print("No race URLs found. Exiting.")
        return

    # 2. Launch Selenium
    chrome_opts = Options()
    if args.headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_opts)

    # 3. Scrape each race
    all_odds = []
    try:
        for idx, url in enumerate(race_urls, 1):
            print(f"[{idx}/{len(race_urls)}]")
            data = scrape_race_odds(driver, url)
            all_odds.extend(data)
            if idx < len(race_urls):
                time.sleep(1.5)  # Polite delay
    finally:
        driver.quit()

    if not all_odds:
        print("No odds data collected.")
        return

    # 4. Save
    df = pd.DataFrame(all_odds)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RAW_DIR / f"rp_odds_{target_date}.csv"
    df.to_csv(outfile, index=False)
    print(f"\nSaved {len(df)} rows to {outfile}")
    print(f"Columns: {', '.join(df.columns)}")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"Trend breakdown:")
    print(df['odds_trend'].value_counts().to_string())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

#### 1B. Racing Post Results Scraper (for model auditing)

Create `scripts/scrape_rp_results.py`:

```python
#!/usr/bin/env python3
"""Scrape actual race results from Racing Post for model auditing.

Usage:
    python scripts/scrape_rp_results.py --date 2026-04-13
"""

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def clean_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', re.sub(r'[\n\r\t]+', ' ', text)).strip()


HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.racingpost.com/',
}


def get_result_urls(session, date_str):
    """Get completed race result URLs for a date."""
    url = f"https://www.racingpost.com/results/{date_str}/"
    base = "https://www.racingpost.com"

    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Results page lists links to individual race results
        links = soup.find_all('a', class_='RC-meetingItem__link')
        results = []
        for a in links:
            href = a.get('href')
            if href and '/results/' in href:
                full = href if href.startswith('http') else base + href
                results.append(full)

        unique = list(dict.fromkeys(results))
        print(f"Found {len(unique)} result pages for {date_str}")
        return unique
    except Exception as e:
        print(f"Error fetching result links: {e}")
        return []


def scrape_single_result(session, url):
    """Scrape actual finishing positions from a single RP results page."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        track_el = (
            soup.select_one('[data-test-selector="RC-courseHeader__name"]')
            or soup.select_one('.rp-raceHeader__courseName')
        )
        time_el = (
            soup.select_one('[data-test-selector="RC-courseHeader__time"]')
            or soup.select_one('.rp-raceHeader__time')
        )

        track = clean_text(track_el.text) if track_el else "Unknown"
        race_time = clean_text(time_el.text) if time_el else "Unknown"

        rows = []
        for row in soup.find_all('tr', class_='rp-horseTable__mainRow',
                                  attrs={"data-test-selector": "table-row"}):
            name_el = row.find('a', {'data-test-selector': 'link-horseName'})
            pos_el = row.find('span', {'data-test-selector': 'text-horsePosition'})

            if name_el:
                name = clean_text(name_el.text)
                raw_pos = clean_text(pos_el.text) if pos_el else "N/A"
                # Extract leading digit from e.g. "1 (8)" → "1"
                match = re.search(r'\d+', str(raw_pos))
                pos = match.group(0) if match else raw_pos

                # Also try to get SP (starting price)
                sp_el = row.find('span', {'data-test-selector': 'text-horseSp'})
                sp = clean_text(sp_el.text) if sp_el else ""

                rows.append({
                    "track_name": track,
                    "time": race_time,
                    "horse_name": name,
                    "actual_position": pos,
                    "starting_price": sp,
                })

        return rows
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Scrape RP results")
    parser.add_argument('--date', help='YYYY-MM-DD (default: yesterday)')
    args = parser.parse_args()

    if args.date:
        target = args.date
    else:
        target = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\nScraping results for {target}...")
    session = requests.Session()
    urls = get_result_urls(session, target)

    all_results = []
    for url in urls:
        results = scrape_single_result(session, url)
        all_results.extend(results)
        time.sleep(1.5)

    if not all_results:
        print("No results found.")
        return

    df = pd.DataFrame(all_results)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RAW_DIR / f"rp_results_{target}.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} result rows to {outpath}")


if __name__ == "__main__":
    main()
```

### Phase 2: ATR Multi-Bookmaker Odds (Python port of the Playwright scraper)

This gives access to **multiple bookmaker prices per horse** — essential for true value betting (comparing our model's implied odds against the best available market price).

Create `scripts/scrape_atr_odds.py`:

```python
#!/usr/bin/env python3
"""Scrape multi-bookmaker odds from At The Races (ATR) odds comparison grid.

Uses Playwright (Python) to render the JS-heavy odds grid and extract
prices from all bookmakers shown. ATR typically shows 6-10 bookmakers
per race.

Outputs: data/raw/atr_odds_YYYY-MM-DD.csv

Dependencies (add to requirements.txt):
    playwright>=1.40

Setup:
    pip install playwright
    playwright install chromium

Usage:
    python scripts/scrape_atr_odds.py                    # today + tomorrow
    python scripts/scrape_atr_odds.py --date 2026-04-14  # specific date
"""

import argparse
import json
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Playwright (sync API for simplicity)
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def human_delay(min_ms=1000, max_ms=3000):
    """Sleep for a random human-like interval."""
    time.sleep(random.randint(min_ms, max_ms) / 1000)


# ──────────────────────────────────────────────
#  Step 1: Get race menu from ATR
# ──────────────────────────────────────────────

def get_atr_race_menu(page, menu_url="https://www.attheraces.com/racecards"):
    """Navigate to ATR racecards page and extract venue/race URLs.

    Returns list of dicts: {venue, race_time, odds_url}
    """
    page.goto(menu_url, wait_until="load", timeout=60000)
    human_delay(2000, 4000)

    # Accept cookies if dialog appears
    try:
        btn = page.locator('button.cky-btn-accept').first
        if btn.is_visible(timeout=3000):
            btn.click()
            page.wait_for_timeout(1000)
    except Exception:
        pass

    races = page.evaluate("""() => {
        const links = Array.from(document.querySelectorAll('a[href*="/racecard/"]'));
        const results = [];
        const seen = new Set();
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

        links.forEach(link => {
            const href = link.getAttribute('href');
            if (!href || seen.has(href)) return;
            seen.add(href);

            const match = href.match(/\\/racecard\\/([^\\/]+)\\/([\\d]{2}-[^\\/]+-[\\d]{4})\\/(\\d{4})/);
            if (!match) return;

            const venue = match[1].replace(/-/g, ' ');
            const dateStr = match[2];
            const timeStr = match[3];
            const hours = timeStr.substring(0, 2);
            const mins = timeStr.substring(2, 4);
            const dateClean = dateStr.replace(/-/g, ' ');
            const dateObj = new Date(dateClean + ' ' + hours + ':' + mins + ' UTC');

            if (isNaN(dateObj.getTime())) return;
            const raceDate = new Date(dateObj.getFullYear(), dateObj.getMonth(), dateObj.getDate());
            if (raceDate < today) return;

            results.push({
                venue: venue,
                race_time: dateObj.toISOString(),
                odds_url: 'https://www.attheraces.com' + href + '/odds'
            });
        });

        return results;
    }""")

    print(f"Found {len(races)} races from ATR menu")
    return races


# ──────────────────────────────────────────────
#  Step 2: Scrape odds grid for a single race
# ──────────────────────────────────────────────

def scrape_atr_race_odds(page, odds_url):
    """Scrape multi-bookmaker odds from a single ATR race odds page.

    Returns dict: {venue, race_time, runners: [{name, prices: [{bookie, price}]}], bookies: []}
    """
    page.goto(odds_url, wait_until="load", timeout=60000)

    try:
        page.wait_for_selector('.odds-grid__row-wrapper--entries', timeout=15000)
    except Exception:
        print(f"  Odds grid not found (may be a finished race): {odds_url}")
        return None

    page.wait_for_timeout(2000)

    data = page.evaluate("""() => {
        // 1. Extract bookmaker names from header
        const bookieHeaders = Array.from(
            document.querySelectorAll('.odds-grid__cell--bookmaker a.bookmaker-logo')
        );
        const bookies = bookieHeaders.map(a => {
            const inner = a.querySelector('.bookmaker-logo__inner');
            return inner ? inner.innerText.trim() : 'Unknown';
        });

        // 2. Build runner ID → name mapping from card entries
        const entries = Array.from(document.querySelectorAll('.card-entry'));
        const runnerMap = {};
        entries.forEach(entry => {
            const link = entry.querySelector('a.horse__link');
            if (link) {
                const name = link.innerText.trim();
                const href = link.getAttribute('href');
                const idMatch = href.match(/\\/(\\d+)(?:\\?|$)/);
                if (idMatch) runnerMap[idMatch[1]] = name;
            }
        });

        // 3. Extract odds grid rows
        const oddsRows = Array.from(document.querySelectorAll('.odds-grid__row--horse'));
        const runners = oddsRows.map(row => {
            const idAttr = row.getAttribute('id');
            const id = idAttr ? idAttr.replace('row-', '') : null;
            let name = id ? (runnerMap[id] || 'Unknown') : 'Unknown';

            // Fallback: look for name element in the row
            if (name === 'Unknown') {
                const nameEl = row.querySelector('.odds-grid__runner-name, .runner-name');
                if (nameEl) name = nameEl.innerText.trim();
            }

            const priceCells = Array.from(row.querySelectorAll('.odds-grid__cell--odds'));
            const prices = priceCells.map((cell, i) => {
                const bookie = bookies[i] || 'Bookie ' + (i + 1);
                const link = cell.querySelector('a.odds-grid-link');
                let price = null;

                if (link) {
                    const dp = link.getAttribute('data-dp') || link.getAttribute('data-odds');
                    if (dp && dp !== '0' && dp !== '-2') {
                        price = dp;
                    } else {
                        const val = link.querySelector('.odds-value--decimal') ||
                                    link.querySelector('.odds-value');
                        if (val) price = val.innerText.trim();
                    }
                }

                return { bookie, price };
            }).filter(p => p.price && p.price !== '-' && p.price !== 'SP');

            return { name, prices };
        }).filter(r => r.prices.length > 0);

        return { runners, bookies };
    }""")

    data['url'] = odds_url
    data['timestamp'] = int(time.time() * 1000)
    return data


# ──────────────────────────────────────────────
#  Step 3: Flatten to DataFrame
# ──────────────────────────────────────────────

def flatten_odds_data(race_results, races_meta):
    """Convert nested JSON to flat rows: one row per horse per bookmaker.

    Also produces a 'best odds' summary: one row per horse with the highest price.
    """
    detail_rows = []  # per-horse-per-bookie
    best_rows = []    # per-horse, best available odds

    for result, meta in zip(race_results, races_meta):
        if result is None:
            continue

        venue = meta.get('venue', 'Unknown')
        race_time = meta.get('race_time', '')

        for runner in result.get('runners', []):
            horse = runner['name']
            best_price = 0.0
            best_bookie = ""

            for price_info in runner['prices']:
                bookie = price_info['bookie']
                price_str = price_info['price']

                # Convert to decimal if fractional
                try:
                    if '/' in str(price_str):
                        parts = price_str.split('/')
                        decimal_price = float(parts[0]) / float(parts[1]) + 1.0
                    else:
                        decimal_price = float(price_str)
                except (ValueError, ZeroDivisionError):
                    decimal_price = None

                detail_rows.append({
                    'venue': venue,
                    'race_time': race_time,
                    'horse_name': horse,
                    'bookmaker': bookie,
                    'price_raw': price_str,
                    'price_decimal': decimal_price,
                })

                if decimal_price and decimal_price > best_price:
                    best_price = decimal_price
                    best_bookie = bookie

            if best_price > 0:
                best_rows.append({
                    'venue': venue,
                    'race_time': race_time,
                    'horse_name': horse,
                    'best_odds_decimal': best_price,
                    'best_bookmaker': best_bookie,
                    'num_bookmakers': len(runner['prices']),
                })

    return pd.DataFrame(detail_rows), pd.DataFrame(best_rows)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape ATR multi-bookmaker odds")
    parser.add_argument('--date', help='YYYY-MM-DD (default: today + tomorrow)')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("AT THE RACES — MULTI-BOOKMAKER ODDS SCRAPER")
    print(f"{'='*60}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-gpu',
                '--no-sandbox',
            ]
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='en-GB',
            timezone_id='Europe/London',
        )
        page = context.new_page()

        # Collect race URLs
        menu_urls = ["https://www.attheraces.com/racecards"]
        if not args.date:
            menu_urls.append("https://www.attheraces.com/racecards/tomorrow")

        all_races = []
        for menu_url in menu_urls:
            races = get_atr_race_menu(page, menu_url)
            all_races.extend(races)

        if not all_races:
            print("No races found. Exiting.")
            browser.close()
            return

        # Deduplicate by URL
        seen = set()
        unique_races = []
        for r in all_races:
            if r['odds_url'] not in seen:
                seen.add(r['odds_url'])
                unique_races.append(r)

        print(f"\nScraping odds for {len(unique_races)} races...\n")

        # Scrape each race
        odds_results = []
        for idx, race in enumerate(unique_races, 1):
            print(f"[{idx}/{len(unique_races)}] {race['venue']} — {race['odds_url']}")
            result = scrape_atr_race_odds(page, race['odds_url'])
            odds_results.append(result)
            if idx < len(unique_races):
                human_delay(1500, 3000)

        browser.close()

    # Flatten and save
    df_detail, df_best = flatten_odds_data(odds_results, unique_races)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    # Detailed: one row per horse per bookmaker
    detail_path = RAW_DIR / f"atr_odds_detail_{date_str}.csv"
    df_detail.to_csv(detail_path, index=False)
    print(f"\nSaved {len(df_detail)} detail rows to {detail_path}")

    # Summary: one row per horse, best available odds
    best_path = RAW_DIR / f"atr_odds_best_{date_str}.csv"
    df_best.to_csv(best_path, index=False)
    print(f"Saved {len(df_best)} best-odds rows to {best_path}")

    if not df_best.empty:
        print(f"\nBookmaker coverage:")
        print(df_detail['bookmaker'].value_counts().head(10).to_string())
        print(f"\nAvg bookmakers per horse: {df_best['num_bookmakers'].mean():.1f}")


if __name__ == "__main__":
    main()
```

### Phase 3: Merging Odds Into Predictions Pipeline

Create `scripts/merge_scraped_odds.py`:

```python
#!/usr/bin/env python3
"""Merge scraped odds (RP + ATR) into the predictions CSV.

Usage:
    python scripts/merge_scraped_odds.py --date 2026-04-14
"""

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def normalize_name(name):
    """Lowercase, strip, collapse whitespace for fuzzy matching."""
    if not isinstance(name, str):
        return ""
    return re.sub(r'\s+', ' ', name.strip().lower())


def merge_rp_odds(df_pred, date_str):
    """Merge Racing Post odds (trend + live fractional) into predictions."""
    rp_path = RAW_DIR / f"rp_odds_{date_str}.csv"
    if not rp_path.exists():
        print(f"No RP odds file: {rp_path}")
        return df_pred

    df_rp = pd.read_csv(rp_path)
    df_rp['_join_name'] = df_rp['horse_name'].apply(normalize_name)
    df_pred['_join_name'] = df_pred['horse_name'].apply(normalize_name)

    # Keep just the columns we need from RP
    rp_cols = ['_join_name', 'odds_live', 'odds_decimal', 'odds_trend',
               'odds_1', 'odds_2', 'odds_3', 'odds_4']
    rp_cols = [c for c in rp_cols if c in df_rp.columns]

    df_pred = df_pred.merge(
        df_rp[rp_cols].drop_duplicates(subset='_join_name'),
        on='_join_name', how='left', suffixes=('', '_rp')
    )
    df_pred.drop(columns=['_join_name'], inplace=True)

    matched = df_pred['odds_live'].notna().sum() if 'odds_live' in df_pred else 0
    print(f"RP odds matched: {matched}/{len(df_pred)} horses")
    return df_pred


def merge_atr_best_odds(df_pred, date_str):
    """Merge ATR best-available bookmaker odds into predictions."""
    atr_path = RAW_DIR / f"atr_odds_best_{date_str}.csv"
    if not atr_path.exists():
        print(f"No ATR odds file: {atr_path}")
        return df_pred

    df_atr = pd.read_csv(atr_path)
    df_atr['_join_name'] = df_atr['horse_name'].apply(normalize_name)
    df_pred['_join_name'] = df_pred['horse_name'].apply(normalize_name)

    atr_cols = ['_join_name', 'best_odds_decimal', 'best_bookmaker', 'num_bookmakers']
    atr_cols = [c for c in atr_cols if c in df_atr.columns]

    df_pred = df_pred.merge(
        df_atr[atr_cols].drop_duplicates(subset='_join_name'),
        on='_join_name', how='left', suffixes=('', '_atr')
    )
    df_pred.drop(columns=['_join_name'], inplace=True)

    matched = df_pred['best_odds_decimal'].notna().sum() if 'best_odds_decimal' in df_pred else 0
    print(f"ATR odds matched: {matched}/{len(df_pred)} horses")
    return df_pred


def calculate_value(df):
    """Add value betting columns using best available odds vs model probability."""
    if 'win_prob' not in df.columns:
        print("No win_prob column — skipping value calculation")
        return df

    # Use best available odds: ATR multi-bookie > RP single-bookie
    df['market_odds'] = df.get('best_odds_decimal', pd.Series(dtype=float))
    mask = df['market_odds'].isna() & df.get('odds_decimal', pd.Series(dtype=float)).notna()
    if mask.any():
        df.loc[mask, 'market_odds'] = df.loc[mask, 'odds_decimal']

    # Implied probability from market
    df['implied_prob'] = 1.0 / df['market_odds'].where(df['market_odds'] > 0)

    # Edge = model probability - implied probability
    df['edge'] = df['win_prob'] - df['implied_prob']

    # EV per unit staked
    df['ev_per_unit'] = df['win_prob'] * df['market_odds'] - 1.0

    # Kelly fraction (quarter Kelly)
    b = df['market_odds'] - 1.0
    df['kelly_full'] = ((df['win_prob'] * df['market_odds'] - 1.0) / b).clip(lower=0)
    df['kelly_quarter'] = df['kelly_full'] * 0.25

    # Odds trend adjustment (from RP data)
    if 'odds_trend' in df.columns:
        trend_adj = df['odds_trend'].map({
            'shortening': 0.02,
            'drifting': -0.02,
            'stable': 0.0,
            'no_data': 0.0,
        }).fillna(0.0)
        df['edge_adjusted'] = df['edge'] + trend_adj

    # Value bet flag
    df['is_value_bet'] = (df['edge'] > 0.01) & (df['ev_per_unit'] > 0.01)
    df['is_strong_value'] = (df['edge'] > 0.15) & (df['ev_per_unit'] > 0.05)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='YYYY-MM-DD')
    args = parser.parse_args()

    pred_path = PROCESSED_DIR / f"predictions_{args.date}.csv"
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        return

    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} predictions for {args.date}")

    df = merge_rp_odds(df, args.date)
    df = merge_atr_best_odds(df, args.date)
    df = calculate_value(df)

    df.to_csv(pred_path, index=False)
    print(f"\nUpdated {pred_path}")

    if 'is_value_bet' in df.columns:
        n_value = df['is_value_bet'].sum()
        n_strong = df['is_strong_value'].sum()
        print(f"Value bets: {n_value}  |  Strong value: {n_strong}")


if __name__ == "__main__":
    main()
```

---

## New Dependencies

Add to `requirements.txt`:

```
# Odds scraping (Phase 1-2)
playwright>=1.40
```

Selenium is already in use for other parts of the project. Playwright is needed for the ATR scraper (its stealth capabilities are better for ATR's anti-bot measures).

After `pip install playwright`, run:
```
playwright install chromium
```

---

## Integration Into Daily Workflow

### Updated `scripts/predict_todays_races.py` hook

After predictions are generated, add odds scraping + merging:

```python
# At the end of predict_todays_races.py, after predictions CSV is saved:
import subprocess

def enrich_with_odds(date_str):
    """Scrape odds from RP + ATR and merge into predictions."""
    scripts = [
        ['python', 'scripts/scrape_rp_odds.py', '--date', date_str],
        ['python', 'scripts/scrape_atr_odds.py', '--date', date_str],
        ['python', 'scripts/merge_scraped_odds.py', '--date', date_str],
    ]
    for cmd in scripts:
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: {cmd[1]} failed: {result.stderr[:200]}")
        else:
            print(result.stdout[-200:] if result.stdout else "OK")
```

### GitHub Actions Integration

Add to `daily_predictions.yml` (after predictions step):

```yaml
      - name: Scrape live odds (Racing Post)
        if: steps.predict.outcome == 'success'
        run: |
          playwright install chromium --with-deps
          python scripts/scrape_rp_odds.py --date ${{ env.TARGET_DATE }}
        continue-on-error: true  # Don't fail workflow if scraping is blocked

      - name: Scrape multi-bookmaker odds (ATR)
        if: steps.predict.outcome == 'success'
        run: |
          python scripts/scrape_atr_odds.py --date ${{ env.TARGET_DATE }}
        continue-on-error: true

      - name: Merge odds into predictions
        if: steps.predict.outcome == 'success'
        run: |
          python scripts/merge_scraped_odds.py --date ${{ env.TARGET_DATE }}
        continue-on-error: true
```

---

## Features We Can Borrow for the Model

### From J-Ack0 (worth implementing)

1. **Wilson Score** — confidence-adjusted win rate that handles small sample sizes better than raw win %. Good for jockey/trainer/course stats when runs < 10.

```python
def wilson_score(wins, total, z=1.96):
    """Wilson score interval lower bound for binomial proportion."""
    if total == 0:
        return 0.0
    phat = wins / total
    denom = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return (centre - margin) / denom
```

2. **EMA Form** — exponential moving average of recent positions (weights recent races more heavily than older ones).

3. **Odds Trend as Feature** — `shortening` / `drifting` / `stable` encoded as a model feature, not just a betting adjustment:

```python
# Encode odds trend for model input
df['odds_trend_encoded'] = df['odds_trend'].map({
    'shortening': 1,   # market money coming in
    'stable': 0,
    'drifting': -1,     # market moving away
    'no_data': 0,
}).fillna(0)
```

4. **Odds-implied probability as feature** — the market's view of win probability is itself a powerful predictor:

```python
df['market_implied_prob'] = 1.0 / df['best_odds_decimal']
# Normalize within race to sum to 1
df['market_implied_prob_norm'] = df.groupby('race_id')['market_implied_prob'].transform(
    lambda x: x / x.sum()
)
```

---

## Risk Considerations

| Risk | Mitigation |
|---|---|
| **Selector breakage** | Both RP and ATR change their HTML periodically. Pin selectors in a config dict; add a smoke test that verifies key selectors exist. |
| **Rate limiting / blocking** | Use 1.5-3s delays between requests. ATR scraper has stealth plugin. Both scrapers work with polite crawling. |
| **Legal / ToS** | Both sites are publicly accessible. Scraping for personal/research use is standard practice. Don't redistribute raw data. |
| **Stale odds** | Odds change rapidly (especially in the last 30 min before a race). Schedule scraping as close to race time as practical. |
| **Name matching** | RP and ATR use slightly different horse name formats. The `normalize_name()` function handles most cases; add fuzzy matching if needed. |
| **Headless Chrome in CI** | GitHub Actions runners have Chrome available. Playwright `install --with-deps` handles this. |

---

## File Summary

| New file | Purpose |
|---|---|
| `scripts/scrape_rp_odds.py` | Racing Post odds + trend (Selenium) |
| `scripts/scrape_rp_results.py` | Racing Post actual results (BS4) |
| `scripts/scrape_atr_odds.py` | ATR multi-bookmaker odds (Playwright) |
| `scripts/merge_scraped_odds.py` | Merge all odds into predictions CSV |
