#!/usr/bin/env python3
"""Scrape live Racing Post odds: fractional prices + 4-snapshot history → trend.

Uses Playwright (headless Chromium) to handle JS-rendered odds.

Output: data/raw/rp_odds_YYYY-MM-DD.csv
Columns: venue, race_time, horse_name, odds_live, odds_decimal,
         odds_1, odds_2, odds_3, odds_4, odds_trend

Usage:
    python scripts/scrape_rp_odds.py                   # tomorrow
    python scripts/scrape_rp_odds.py --date 2026-04-22 # specific date
    python scripts/scrape_rp_odds.py --no-headless      # visible browser (debug)

Prerequisites:
    pip install playwright
    playwright install chromium
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
    """Return a deduplicated list of individual racecard URLs for *date_str*."""
    listing_url = f"{BASE_URL}/racecards/{date_str}/"
    print(f"Fetching race list: {listing_url}")

    try:
        page.goto(listing_url, wait_until="domcontentloaded", timeout=35000)
        page.wait_for_timeout(3500)   # React hydration
        _accept_cookies(page)

        links = page.evaluate("""() => {
            const BASE = 'https://www.racingpost.com';
            // Primary: links that carry the race-is-over attribute
            let anchors = Array.from(document.querySelectorAll('a[data-race-is-over]'));
            // Fallback: any racecard deep link (date/course/time/id)
            if (anchors.length === 0) {
                anchors = Array.from(
                    document.querySelectorAll('a[href*="/racecards/"]')
                ).filter(a => (a.getAttribute('href') || '').split('/').length >= 5);
            }
            const seen = new Set();
            const out = [];
            anchors.forEach(a => {
                const h = a.getAttribute('href');
                if (!h || seen.has(h)) return;
                seen.add(h);
                out.push(h.startsWith('http') ? h : BASE + h);
            });
            return out;
        }""")

        print(f"  Found {len(links)} race URLs")
        return links

    except Exception as exc:
        print(f"  Error fetching race list: {exc}")
        return []


# ─────────────────────────────────────────────
#  Step 2: Scrape odds from one racecard page
# ─────────────────────────────────────────────

def scrape_race_odds(page, race_url):
    """Extract runners + odds from a single RP racecard URL.

    Returns list of dicts or [] on failure.
    """
    try:
        page.goto(race_url, wait_until="domcontentloaded", timeout=35000)

        # Wait for the runner rows container
        try:
            page.wait_for_selector(".RC-runnerRow", timeout=14000)
        except Exception:
            print(f"  No runner rows at {race_url} — skipping")
            return []

        page.wait_for_timeout(2200)   # let odds JS finish loading

        # Read venue + time from the race header
        venue = ""
        race_time = ""
        try:
            venue = page.locator(
                '[data-test-selector="RC-courseHeader__name"]'
            ).first.inner_text(timeout=3000).strip()
            race_time = page.locator(
                '[data-test-selector="RC-courseHeader__time"]'
            ).first.inner_text(timeout=3000).strip()
        except Exception:
            pass

        runners = page.evaluate("""() => {
            return Array.from(document.querySelectorAll('.RC-runnerRow'))
                .map(row => {
                    const nameEl = row.querySelector(
                        '[data-test-selector="RC-cardPage-runnerName"]');
                    const priceEl = row.querySelector(
                        '[data-test-selector="RC-cardPage-runnerPrice"]');

                    const hist = [];
                    for (let i = 1; i <= 4; i++) {
                        const el = row.querySelector(
                            `[data-test-selector="RC-historyPrices-item-${i}"]`);
                        hist.push(el ? el.innerText.trim() : '');
                    }

                    return {
                        name: nameEl
                            ? nameEl.innerText.trim().split('\\n')[0].trim()
                            : '',
                        odds_live: priceEl ? priceEl.innerText.trim() : '',
                        hist,
                    };
                })
                .filter(r => r.name && r.name.length > 1);
        }""")

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
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run Chrome headless (default)")
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
        print("  pip install playwright && playwright install chromium")
        return

    all_odds = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-GB",
            timezone_id="Europe/London",
        )
        page = context.new_page()

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
    print(f"\nSaved {len(df)} rows → {out_path}")

    if "odds_trend" in df.columns:
        print("\nTrend breakdown:")
        print(df["odds_trend"].value_counts().to_string())
    if "odds_decimal" in df.columns:
        with_odds = df["odds_decimal"].notna().sum()
        print(f"Horses with decimal odds: {with_odds}/{len(df)}")


if __name__ == "__main__":
    main()
