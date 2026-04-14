#!/usr/bin/env python3
"""Scrape multi-bookmaker odds from At The Races (ATR) odds comparison grid.

ATR shows 6-10 bookmakers per race. This script extracts all prices from the
odds grid for today (and tomorrow if no --date given).

Outputs:
  data/raw/atr_odds_detail_YYYY-MM-DD.csv  — one row per horse per bookmaker
  data/raw/atr_odds_best_YYYY-MM-DD.csv    — one row per horse, best price only

Usage:
    python scripts/scrape_atr_odds.py                   # today + tomorrow
    python scripts/scrape_atr_odds.py --date 2026-04-22 # specific date
    python scripts/scrape_atr_odds.py --no-headless      # visible browser (debug)

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
ATR_BASE = "https://www.attheraces.com"


# ─────────────────────────────────────────────
#  Timing helpers
# ─────────────────────────────────────────────

def human_delay(min_ms: int = 1000, max_ms: int = 3000):
    time.sleep(random.randint(min_ms, max_ms) / 1000)


def _accept_cookies(page):
    """Dismiss ATR / CookieYes consent banner if present."""
    for sel in [
        "button.cky-btn-accept",
        'button:has-text("Accept All")',
        'button:has-text("Accept")',
        "#onetrust-accept-btn-handler",
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=2000):
                btn.click()
                page.wait_for_timeout(800)
                return
        except Exception:
            continue


# ─────────────────────────────────────────────
#  Step 1: Gather race menu from ATR
# ─────────────────────────────────────────────

def get_atr_race_menu(page, menu_url: str) -> list:
    """Navigate to ATR racecards page and scrape venue/race/time/odds URLs.

    Returns list of dicts: {venue, raw_date, raw_time, odds_url}
    """
    print(f"Fetching ATR menu: {menu_url}")
    try:
        page.goto(menu_url, wait_until="load", timeout=60000)
        human_delay(2500, 4500)
        _accept_cookies(page)

        races = page.evaluate("""() => {
            const ATR = 'https://www.attheraces.com';
            const links = Array.from(document.querySelectorAll('a[href*="/racecard/"]'));
            const seen = new Set();
            const results = [];

            links.forEach(link => {
                const href = link.getAttribute('href');
                if (!href || seen.has(href)) return;
                seen.add(href);

                // ATR URL format: /racecard/{Venue}/{DD-Month-YYYY}/{HHmm}
                const m = href.match(
                    /\\/racecard\\/([^\\/]+)\\/([\\d]{2}-[^\\/]+)\\/([\\d]{3,4})/
                );
                if (!m) return;

                const cleanPath = href.split('?')[0];
                results.push({
                    venue:    m[1].replace(/-/g, ' '),
                    raw_date: m[2],
                    raw_time: m[3],
                    odds_url: (href.startsWith('http') ? '' : ATR)
                              + cleanPath + '/odds',
                });
            });

            return results;
        }""")

        print(f"  Found {len(races)} races from {menu_url}")
        return races

    except Exception as exc:
        print(f"  Error fetching ATR menu ({menu_url}): {exc}")
        return []


# ─────────────────────────────────────────────
#  Step 2: Scrape the odds grid for one race
# ─────────────────────────────────────────────

def scrape_atr_race_odds(page, odds_url: str) -> dict | None:
    """Scrape the ATR multi-bookmaker odds grid for a single race URL.

    Returns dict {runners, bookies, url} or None if grid unavailable.
    """
    try:
        page.goto(odds_url, wait_until="load", timeout=60000)

        try:
            page.wait_for_selector(
                ".odds-grid__row-wrapper--entries", timeout=15000
            )
        except Exception:
            print(f"  Odds grid not found (race may be finished): {odds_url}")
            return None

        page.wait_for_timeout(2000)

        data = page.evaluate("""() => {
            // 1. Bookmaker names from column headers
            const bookieHeaders = Array.from(
                document.querySelectorAll(
                    '.odds-grid__cell--bookmaker a.bookmaker-logo'
                )
            );
            const bookies = bookieHeaders.map(a => {
                const inner = a.querySelector('.bookmaker-logo__inner');
                return inner ? inner.innerText.trim() : 'Unknown';
            });

            // 2. Map runner horse-profile IDs → horse names
            const runnerMap = {};
            document.querySelectorAll('.card-entry').forEach(entry => {
                const link = entry.querySelector('a.horse__link');
                if (!link) return;
                const name = link.innerText.trim();
                const href = link.getAttribute('href') || '';
                const m = href.match(/\\/(\\d+)(?:\\?|$)/);
                if (m) runnerMap[m[1]] = name;
            });

            // 3. Extract odds rows
            const rows = Array.from(
                document.querySelectorAll('.odds-grid__row--horse')
            );

            const runners = rows.map(row => {
                const rowId = (row.getAttribute('id') || '').replace('row-', '');
                let name = rowId ? (runnerMap[rowId] || '') : '';
                if (!name) {
                    const el = row.querySelector(
                        '.odds-grid__runner-name, .runner-name'
                    );
                    if (el) name = el.innerText.trim();
                }
                if (!name) return null;

                const cells = Array.from(
                    row.querySelectorAll('.odds-grid__cell--odds')
                );
                const prices = cells
                    .map((cell, i) => {
                        const bookie = bookies[i] || ('Bookie ' + (i + 1));
                        const link = cell.querySelector('a.odds-grid-link');
                        if (!link) return null;
                        // Prefer numeric data attribute
                        const dp = link.getAttribute('data-dp')
                                || link.getAttribute('data-odds');
                        let price = (dp && dp !== '0' && dp !== '-2')
                            ? dp
                            : null;
                        if (!price) {
                            const val = link.querySelector('.odds-value--decimal')
                                     || link.querySelector('.odds-value');
                            if (val) price = val.innerText.trim();
                        }
                        if (!price || price === '-' || price === 'SP') return null;
                        return { bookie, price };
                    })
                    .filter(Boolean);

                return prices.length > 0 ? { name, prices } : null;
            }).filter(Boolean);

            return { runners, bookies };
        }""")

        if not data or not data.get("runners"):
            return None

        data["url"] = odds_url
        return data

    except Exception as exc:
        print(f"  Error scraping {odds_url}: {exc}")
        return None


# ─────────────────────────────────────────────
#  Step 3: Flatten nested JSON → DataFrames
# ─────────────────────────────────────────────

def _parse_decimal(price_str: str) -> float | None:
    """Convert a price string ('4.50', '9/2', '7') to decimal float or None."""
    s = str(price_str).strip()
    try:
        if "/" in s:
            n, d = s.split("/", 1)
            return round(float(n) / float(d) + 1.0, 4)
        v = float(s)
        # ATR sometimes gives plain numerator (e.g. '9' meaning '9/1')
        # Only add 1 if it looks fractional (< 100 and no decimal point)
        return round(v, 4)
    except (ValueError, ZeroDivisionError):
        return None


def flatten_odds(race_results: list, races_meta: list) -> tuple:
    """Convert list of race result dicts to (detail_df, best_df).

    detail_df: one row per horse per bookmaker
    best_df:   one row per horse, best (highest) decimal price
    """
    detail_rows = []
    best_rows = []

    for result, meta in zip(race_results, races_meta):
        if result is None:
            continue

        venue = meta.get("venue", "")
        raw_time = str(meta.get("raw_time", ""))

        # Normalise raw_time (e.g. '1430' → '14:30', '130' → '01:30')
        if len(raw_time) == 4:
            race_time = f"{raw_time[:2]}:{raw_time[2:]}"
        elif len(raw_time) == 3:
            race_time = f"0{raw_time[0]}:{raw_time[1:]}"
        else:
            race_time = raw_time

        for runner in result.get("runners", []):
            horse = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", runner["name"]).strip()
            best_dec = 0.0
            best_bookie = ""

            for price_info in runner["prices"]:
                bookie = price_info["bookie"]
                price_str = str(price_info["price"])
                dec = _parse_decimal(price_str)

                detail_rows.append({
                    "venue": venue,
                    "race_time": race_time,
                    "horse_name": horse,
                    "bookmaker": bookie,
                    "price_raw": price_str,
                    "price_decimal": dec,
                })

                if dec is not None and dec > best_dec:
                    best_dec = dec
                    best_bookie = bookie

            if best_dec > 0:
                best_rows.append({
                    "venue": venue,
                    "race_time": race_time,
                    "horse_name": horse,
                    "best_odds_decimal": round(best_dec, 4),
                    "best_bookmaker": best_bookie,
                    "num_bookmakers": len(runner["prices"]),
                })

    return pd.DataFrame(detail_rows), pd.DataFrame(best_rows)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape ATR multi-bookmaker odds"
    )
    parser.add_argument(
        "--date", help="YYYY-MM-DD (default: today + tomorrow)"
    )
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run Chromium headless (default)")
    parser.add_argument("--no-headless", dest="headless",
                        action="store_false")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print("AT THE RACES — MULTI-BOOKMAKER ODDS SCRAPER")
    print(f"Target date: {date_str}")
    print(f"{'='*60}\n")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright not installed.")
        print("  pip install playwright && playwright install chromium")
        return

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

        # Collect from today (and tomorrow if no specific date given)
        menu_urls = [f"{ATR_BASE}/racecards"]
        if not args.date:
            menu_urls.append(f"{ATR_BASE}/racecards/tomorrow")

        all_races: list = []
        for mu in menu_urls:
            all_races.extend(get_atr_race_menu(page, mu))

        # Deduplicate by odds_url
        seen_urls: set = set()
        unique_races = []
        for r in all_races:
            if r["odds_url"] not in seen_urls:
                seen_urls.add(r["odds_url"])
                unique_races.append(r)

        if not unique_races:
            print("No races found — exiting.")
            browser.close()
            return

        print(f"\nScraping odds for {len(unique_races)} races...\n")

        odds_results = []
        for idx, race in enumerate(unique_races, 1):
            print(
                f"[{idx}/{len(unique_races)}] "
                f"{race.get('venue', '')} — {race['odds_url']}"
            )
            result = scrape_atr_race_odds(page, race["odds_url"])
            odds_results.append(result)
            if idx < len(unique_races):
                human_delay(1500, 3000)

        browser.close()

    df_detail, df_best = flatten_odds(odds_results, unique_races)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = RAW_DIR / f"atr_odds_detail_{date_str}.csv"
    df_detail.to_csv(detail_path, index=False)
    print(f"\nSaved {len(df_detail)} detail rows → {detail_path}")

    best_path = RAW_DIR / f"atr_odds_best_{date_str}.csv"
    df_best.to_csv(best_path, index=False)
    print(f"Saved {len(df_best)} best-odds rows → {best_path}")

    if not df_best.empty:
        print(f"\nAvg bookmakers per horse: {df_best['num_bookmakers'].mean():.1f}")
    if not df_detail.empty:
        print("\nTop bookmakers by coverage:")
        print(df_detail["bookmaker"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
