#!/usr/bin/env python3
"""Scrape live Racing Post odds: fractional prices + 4-snapshot history → trend.

Uses Playwright with system Chrome (non-headless) to bypass bot detection.
Racing Post detects headless Chromium and serves an empty page; the visible
chrome channel passes cleanly.

Output: data/raw/rp_odds_YYYY-MM-DD.csv
Columns: venue, race_time, horse_name, odds_live, odds_decimal,
         odds_1, odds_2, odds_3, odds_4, odds_trend

Usage:
    python scripts/scrape_rp_odds.py                   # tomorrow (visible Chrome)
    python scripts/scrape_rp_odds.py --date 2026-04-22 # specific date
    python scripts/scrape_rp_odds.py --headless         # headless (CI, likely fails)

Prerequisites:
    pip install playwright playwright-stealth
    playwright install chromium
    System Chrome is used when available (channel='chrome').
"""

import argparse
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://www.racingpost.com"


# ─────────────────────────────────────────────
#  Odds helpers
# ─────────────────────────────────────────────

def fractional_to_decimal(s):
    """'9/2' → 5.5  |  'Evs'/'Evens' → 2.0  |  '5' → 6.0  |  failure → None"""
    if not s:
        return None
    s = str(s).strip()
    if not s or s in ("-", "N/A", "SP"):
        return None
    if s.lower() in ("evs", "evens", "1/1"):
        return 2.0
    try:
        if "/" in s:
            n, d = s.split("/", 1)
            return round(float(n) / float(d) + 1.0, 4)
        # plain integer like "4" means 4/1
        return round(float(s) + 1.0, 4)
    except (ValueError, ZeroDivisionError):
        return None


def calculate_trend(odds_list):
    """Classify movement from a list [latest, …, oldest] of fractional odds strings.

    Returns 'shortening' | 'drifting' | 'stable' | 'no_data'
    """
    dec = [fractional_to_decimal(o) for o in odds_list]
    valid = [o for o in dec if o is not None]
    if len(valid) < 2:
        return "no_data"
    change_pct = (valid[-1] - valid[0]) / valid[0] * 100
    if change_pct < -5:
        return "shortening"
    if change_pct > 5:
        return "drifting"
    return "stable"


# ─────────────────────────────────────────────
#  Browser helpers
# ─────────────────────────────────────────────

def _accept_cookies(page):
    """Dismiss cookie consent banner if present."""
    for sel in [
        'button:has-text("Accept All")',
        'button:has-text("Accept")',
        ".truste-button1",
        "#onetrust-accept-btn-handler",
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=1500):
                btn.click()
                page.wait_for_timeout(800)
                return
        except Exception:
            continue


# ─────────────────────────────────────────────
#  Step 1: Get race URLs for the target date
# ─────────────────────────────────────────────

def get_race_urls(page, date_str):
    """Return a deduplicated list of individual racecard URLs for *date_str*.

    Navigates the RP date-listing page and finds all race links in the format
    /racecards/{course_uid}/{course_key}/{date}/{race_id}.
    """
    listing_url = f"{BASE_URL}/racecards/{date_str}/"
    print(f"Fetching race list: {listing_url}")

    try:
        page.goto(listing_url, wait_until="load", timeout=35000)
        # Wait for racecard links to appear (JS-rendered via React)
        page.wait_for_selector('a[href*="/racecards/"]', timeout=12000)
        page.wait_for_timeout(2000)   # extra JS settle time
        _accept_cookies(page)

        # All racecard anchor tags
        all_links = page.eval_on_selector_all(
            'a[href*="racecards"]',
            'els => els.map(e => e.href)'
        )
        # Keep only individual race pages: /racecards/{num}/{course}/{date}/{race_id}
        # URLs without a numeric race_id at the end are "show all meeting" pages (duplicates).
        seen = set()
        race_links = []
        for href in all_links:
            # Must match: /racecards/{digit+}/{slug}/{YYYY-MM-DD}/{digit+}
            if re.search(r'/racecards/\d+/[^/]+/\d{4}-\d{2}-\d{2}/\d+', href):
                # Normalise: strip trailing slash / query string
                clean = href.rstrip('/').split('?')[0]
                if clean not in seen:
                    seen.add(clean)
                    race_links.append(clean)

        print(f"  Found {len(race_links)} individual race URLs")
        return race_links

    except Exception as exc:
        print(f"  Error fetching race list: {exc}")
        return []


# ─────────────────────────────────────────────
#  Step 2: Scrape odds from one racecard page
# ─────────────────────────────────────────────

def scrape_race_odds(page, race_url):
    """Extract runners + odds from a single RP racecard URL.

    Uses targeted data-test-selector attributes verified against the live RP page.
    Horse names and current prices are queried directly (not inside .RC-runnerRow
    which proved unreliable); history-price items are linked by position.

    Returns list of dicts or [] on failure.
    """
    try:
        page.goto(race_url, wait_until="load", timeout=35000)

        # Wait for horse names to appear (reliable presence check)
        try:
            page.wait_for_selector(
                '[data-test-selector="RC-cardPage-runnerName"]', timeout=14000
            )
        except Exception:
            print(f"  No runner names at {race_url} — skipping")
            return []

        page.wait_for_timeout(1500)   # let live-price JS finish loading

        # Parse venue + time from URL  (e.g. …/racecards/38/newmarket/2026-04-15/915460)
        venue = ""
        race_time = ""
        m = re.search(r'/racecards/\d+/([^/]+)/(\d{4}-\d{2}-\d{2})/', race_url + '/')
        if m:
            venue = m.group(1).replace('-', ' ').title()
        # Try the race-header selectors as well (more official name)
        for sel in ('[data-test-selector="RC-courseHeader__name"]',
                    '[class*="RC-courseHeader__name"]'):
            try:
                t = page.locator(sel).first.inner_text(timeout=2000).strip()
                if t:
                    venue = t
                    break
            except Exception:
                pass
        for sel in ('[data-test-selector="RC-courseHeader__time"]',
                    '[class*="RC-courseHeader__time"]',
                    'h1',
                    '[data-test-selector*="raceTime"]'):
            try:
                t = page.locator(sel).first.inner_text(timeout=2000).strip()
                m_t = re.search(r'\b(\d{1,2}:\d{2})\b', t)
                if m_t:
                    race_time = m_t.group(1)
                    break
            except Exception:
                pass
        # Fallback: extract time from URL pattern if still empty
        if not race_time:
            m_u = re.search(r'/(\d{4}-\d{2}-\d{2})/', race_url)
            if m_u:
                # Time not in URL — leave blank; merge script will match by horse name
                pass

        # --- Core extraction: names, current prices, history ---
        names = page.eval_on_selector_all(
            '[data-test-selector="RC-cardPage-runnerName"]',
            'els => els.map(e => e.innerText.trim())'
        )
        prices = page.eval_on_selector_all(
            '[data-test-selector="RC-cardPage-runnerPrice"]',
            'els => els.map(e => e.innerText.trim())'
        )
        # History prices: item-1 (oldest) … item-4 (most-recent pre-live)
        # They appear in DOM in runner order so we chunk by len(names)
        history_raw = page.eval_on_selector_all(
            '[data-test-selector^="RC-historyPrices-item"]',
            'els => els.map(e => ({sel: e.getAttribute("data-test-selector"), text: e.innerText.trim()}))'
        )

        # Build per-horse history: group into blocks of items per name count
        n_runners = len(names)
        # Each runner should have the same number of history items
        n_hist = len(history_raw) // n_runners if n_runners else 0
        runners = []
        for i, (name, price) in enumerate(zip(names, prices)):
            hist_block = history_raw[i * n_hist:(i + 1) * n_hist] if n_hist else []
            hist = [h['text'] for h in hist_block]
            # Pad to 4
            while len(hist) < 4:
                hist.append('')
            runners.append({'name': name, 'odds_live': price, 'hist': hist})

        if not runners:
            print(f"  No runners extracted from {race_url}")
            return []
        rows = []
        for r in runners:
            name = re.sub(r"\s+", " ", r["name"]).strip()
            # Strip country suffix e.g. (IRE)
            name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()
            if not name:
                continue
            odds_live = r.get("odds_live", "")
            hist = r.get("hist", ["", "", "", ""])
            rows.append({
                "venue": venue,
                "race_time": race_time,
                "horse_name": name,
                "odds_live": odds_live,
                "odds_decimal": fractional_to_decimal(odds_live),
                "odds_1": hist[0] if len(hist) > 0 else "",
                "odds_2": hist[1] if len(hist) > 1 else "",
                "odds_3": hist[2] if len(hist) > 2 else "",
                "odds_4": hist[3] if len(hist) > 3 else "",
                "odds_trend": calculate_trend([odds_live] + hist),
            })

        print(f"  {venue or '?'} {race_time or '?'}: {len(rows)} runners")
        return rows

    except Exception as exc:
        print(f"  Error scraping {race_url}: {exc}")
        return []


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape Racing Post odds")
    parser.add_argument("--date", help="YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run Chrome headless (CI mode — likely blocked by RP bot detection)")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    args = parser.parse_args()

    target_date = args.date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"RACING POST ODDS SCRAPER — {target_date}")
    print(f"{'='*60}\n")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright not installed.")
        print("  pip install playwright playwright-stealth && playwright install chromium")
        return

    try:
        from playwright_stealth import Stealth
        _stealth = Stealth()
    except ImportError:
        _stealth = None
        print("WARNING: playwright-stealth not installed (pip install playwright-stealth). Stealth mode disabled.")

    if args.headless:
        print("WARNING: Running headless — Racing Post may block this request.")
        print("         Run without --headless for reliable local scraping.")

    all_odds = []

    with sync_playwright() as p:
        # Prefer system Chrome (channel='chrome') as it is less identifiable.
        # Fall back to bundled Chromium if Chrome is not installed.
        launch_kwargs = dict(
            headless=args.headless,
            args=['--disable-blink-features=AutomationControlled'],
        )
        try:
            browser = p.chromium.launch(channel='chrome', **launch_kwargs)
        except Exception:
            browser = p.chromium.launch(**launch_kwargs)

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-GB",
            timezone_id="Europe/London",
        )
        page = context.new_page()
        if _stealth:
            _stealth.apply_stealth_sync(page)

        race_urls = get_race_urls(page, target_date)
        if not race_urls:
            print("No race URLs found — exiting.")
            browser.close()
            return

        for idx, url in enumerate(race_urls, 1):
            print(f"[{idx}/{len(race_urls)}] {url}")
            rows = scrape_race_odds(page, url)
            all_odds.extend(rows)
            if idx < len(race_urls):
                time.sleep(random.uniform(1.5, 3.0))

        browser.close()

    if not all_odds:
        print("No odds data collected.")
        return

    df = pd.DataFrame(all_odds)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"rp_odds_{target_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows -> {out_path}")

    if "odds_trend" in df.columns:
        print("\nTrend breakdown:")
        print(df["odds_trend"].value_counts().to_string())
    if "odds_decimal" in df.columns:
        with_odds = df["odds_decimal"].notna().sum()
        print(f"Horses with decimal odds: {with_odds}/{len(df)}")


if __name__ == "__main__":
    main()
