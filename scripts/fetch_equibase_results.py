"""
fetch_equibase_results.py — Scrape Equibase chart/result pages for completed US races.

For each track+date, parses the Equibase results page and extracts finishing
positions, winning margins, and runner details.  The results are then joined
against existing us_predictions_*.csv files to produce an accuracy report.

Output:
    data/raw/equibase_results_YYYY-MM-DD.json    — raw scraped results
    data/processed/equibase_results_YYYY-MM-DD.csv — flat CSV
    data/processed/us_accuracy_YYYY-MM-DD.csv    — predictions joined to results

Usage:
    python scripts/fetch_equibase_results.py --date 2026-05-09
    python scripts/fetch_equibase_results.py --date 2026-05-09 --force
    python scripts/fetch_equibase_results.py --days 7          # last 7 days
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"
PROC_DIR  = REPO_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] fetch_equibase_results: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fetch_equibase_results")

# Equibase results URL pattern: chart page shows finishing order
# e.g. https://www.equibase.com/static/chart/summary/CD052426USA.htm
#   format: TRACKMMDDYYcountry.htm
# Results index listing all tracks for a given date:
# https://www.equibase.com/static/chart/summary/index.htm?DateRace=05%2F24%2F2026
RESULTS_INDEX_URL = "https://www.equibase.com/static/chart/summary/index.htm"
CHART_BASE_URL    = "https://www.equibase.com/static/chart/summary/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# --------------------------------------------------------------------------- #
# HTML parsing helpers
# --------------------------------------------------------------------------- #

def _parse_results_index(html: str, date_str: str) -> list[str]:
    """
    Parse the Equibase chart-summary index page and return a list of
    chart page URLs for the given date.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        # Equibase relative links like '/static/chart/summary/CD052426USA.htm'
        if re.search(r'/static/chart/summary/[A-Z]+\d{6}[A-Z]+\.htm', href, re.I):
            full = href if href.startswith("http") else f"https://www.equibase.com{href}"
            links.append(full)
    # Remove duplicates preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique


def _parse_chart_page(html: str, chart_url: str) -> list[dict]:
    """
    Parse one Equibase chart page and return a list of result dicts,
    one per runner per race.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Derive track + date from URL filename like 'CD052426USA.htm'
    fname = chart_url.split("/")[-1].replace(".htm", "").replace(".HTML", "")
    track_code = re.match(r'^([A-Z]+)', fname)
    track_code = track_code.group(1) if track_code else "UNK"

    # Parse date from filename: MMDDYY
    date_m = re.search(r'([A-Z]+)(\d{2})(\d{2})(\d{2})([A-Z]+)', fname, re.I)
    race_date = ""
    if date_m:
        mm, dd, yy = date_m.group(2), date_m.group(3), date_m.group(4)
        year = int(yy) + (2000 if int(yy) < 50 else 1900)
        race_date = f"{year}-{mm}-{dd}"

    results: list[dict] = []

    # Each race result block has a heading like "RACE 1" and a table of runners
    race_sections = soup.find_all("div", class_=re.compile(r"race", re.I)) or []
    if not race_sections:
        # Fallback: try table-based parsing
        race_sections = [soup]

    current_race_num = 0
    current_race_name = ""

    for section in soup.find_all(True):
        tag_text = section.get_text(separator=" ", strip=True)
        # Detect race header
        race_header = re.match(r'RACE\s+(\d+)', tag_text[:30], re.I)
        if race_header and section.name in ("h2", "h3", "h4", "div", "span"):
            current_race_num = int(race_header.group(1))
            current_race_name = tag_text.strip()[:80]
            continue

        # Try to parse runner rows from tables inside race sections
        if section.name == "table":
            rows = section.find_all("tr")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) < 3:
                    continue
                # Heuristic: first cell is finish position (digit or PP)
                pos_raw = cells[0]
                if not re.match(r'^\d+$', pos_raw):
                    continue
                position = int(pos_raw)
                # Runner name is typically in cell index 1 or 2
                horse = cells[1] if len(cells) > 1 else ""
                jockey = cells[2] if len(cells) > 2 else ""
                margin = cells[3] if len(cells) > 3 else ""  # beaten lengths
                results.append({
                    "race_date":   race_date,
                    "track_code":  track_code,
                    "race_number": current_race_num,
                    "race_name":   current_race_name,
                    "position":    position,
                    "horse":       horse,
                    "jockey":      jockey,
                    "margin_str":  margin,
                    "source_url":  chart_url,
                })

    return results


# --------------------------------------------------------------------------- #
# Playwright fallback
# --------------------------------------------------------------------------- #

def _fetch_html_playwright(url: str, timeout_ms: int = 30_000) -> str:
    """Fetch a JS-rendered page via Playwright (headless Chromium)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("playwright not installed: pip install playwright && playwright install chromium")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=_HEADERS["User-Agent"])
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(2_000)
        html = page.content()
        browser.close()
    return html


# --------------------------------------------------------------------------- #
# Fetch & save
# --------------------------------------------------------------------------- #

def fetch_results_for_date(race_date: str, force: bool = False) -> list[dict]:
    """
    Fetch and parse Equibase results for the given date.
    Returns flat list of result dicts (one per runner per race).
    """
    raw_path  = RAW_DIR  / f"equibase_results_{race_date}.json"
    csv_path  = PROC_DIR / f"equibase_results_{race_date}.csv"

    if raw_path.exists() and not force:
        logger.info("Loading cached results for %s", race_date)
        return json.loads(raw_path.read_text(encoding="utf-8")).get("results", [])

    logger.info("Fetching Equibase results index for %s", race_date)

    # Build index URL with date query param (MM/DD/YYYY)
    dt = datetime.strptime(race_date, "%Y-%m-%d")
    date_param = dt.strftime("%m%%2F%d%%2F%Y")
    index_url = f"{RESULTS_INDEX_URL}?DateRace={date_param}"

    # Try requests first, fall back to Playwright if needed
    try:
        resp = requests.get(index_url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        index_html = resp.text
    except Exception as exc:
        logger.warning("requests failed for index (%s) — trying Playwright", exc)
        try:
            index_html = _fetch_html_playwright(index_url)
        except Exception as exc2:
            logger.error("Playwright also failed: %s", exc2)
            return []

    chart_urls = _parse_results_index(index_html, race_date)
    logger.info("Found %d chart page(s) for %s", len(chart_urls), race_date)

    all_results: list[dict] = []

    for chart_url in chart_urls:
        logger.info("  Fetching chart: %s", chart_url)
        try:
            resp = requests.get(chart_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            chart_html = resp.text
        except Exception as exc:
            logger.warning("  Failed to fetch %s: %s — trying Playwright", chart_url, exc)
            try:
                chart_html = _fetch_html_playwright(chart_url)
            except Exception as exc2:
                logger.warning("  Playwright failed for %s: %s", chart_url, exc2)
                continue

        results = _parse_chart_page(chart_html, chart_url)
        logger.info("  → Parsed %d runner results", len(results))
        all_results.extend(results)

    # Save raw JSON
    payload = {
        "race_date":   race_date,
        "fetched_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chart_urls":  chart_urls,
        "count":       len(all_results),
        "results":     all_results,
    }
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d results to %s", len(all_results), raw_path)

    # Save CSV
    if all_results:
        cols = ["race_date", "track_code", "race_number", "race_name",
                "position", "horse", "jockey", "margin_str", "source_url"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("CSV saved to %s", csv_path)

    return all_results


# --------------------------------------------------------------------------- #
# Accuracy join
# --------------------------------------------------------------------------- #

def _normalise_horse(name: str) -> str:
    """Strip suffixes and lowercase for fuzzy join."""
    s = re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', str(name), flags=re.I).strip().lower()
    return re.sub(r'\s+', ' ', s)


def build_accuracy_report(race_date: str) -> None:
    """
    Join us_predictions_*.csv with equibase_results_*.csv and save an accuracy CSV.

    Accuracy CSV columns:
        date, course, race_time, race_name, horse,
        predicted_rank, actual_position,
        top1_correct, top3_correct,
        model_win_prob, model_win_odds
    """
    import pandas as pd

    pred_path   = PROC_DIR / f"us_predictions_{race_date}.csv"
    res_path    = PROC_DIR / f"equibase_results_{race_date}.csv"
    out_path    = PROC_DIR / f"us_accuracy_{race_date}.csv"

    if not pred_path.exists():
        logger.warning("No predictions file for %s — skipping accuracy join", race_date)
        return
    if not res_path.exists():
        logger.warning("No results file for %s — skipping accuracy join", race_date)
        return

    preds   = pd.read_csv(pred_path)
    results = pd.read_csv(res_path)

    # Normalise horse names
    preds["horse_key"]   = preds["horse"].apply(_normalise_horse)
    results["horse_key"] = results["horse"].apply(_normalise_horse)

    # Rank horses per race (prediction rank)
    preds["predicted_rank"] = (
        preds.groupby(["course", "race_time"])["win_probability"]
             .rank(method="first", ascending=False)
             .astype(int)
    )

    # Join on horse_key (track_code ≈ course — best-effort fuzzy join)
    merged = preds.merge(
        results[["horse_key", "track_code", "race_number", "position", "margin_str"]],
        on="horse_key",
        how="left",
    )
    merged["actual_position"] = pd.to_numeric(merged["position"], errors="coerce")
    merged["top1_correct"]    = ((merged["predicted_rank"] == 1) & (merged["actual_position"] == 1)).astype(int)
    merged["top3_correct"]    = ((merged["predicted_rank"] <= 3) & (merged["actual_position"] <= 3)).astype(int)

    keep_cols = [c for c in [
        "date", "course", "race_time", "race_name", "horse",
        "predicted_rank", "actual_position", "top1_correct", "top3_correct",
        "win_probability", "win_odds_fractional", "margin_str",
    ] if c in merged.columns]

    merged[keep_cols].sort_values(["course", "race_time", "predicted_rank"]).to_csv(
        out_path, index=False
    )
    logger.info("Accuracy report saved to %s", out_path)

    # Print summary
    top1_acc = merged["top1_correct"].sum() / max((merged["predicted_rank"] == 1).sum(), 1)
    top3_acc = merged[merged["predicted_rank"] <= 3]["top3_correct"].mean()
    races_n  = merged.groupby(["course", "race_time"]).ngroups
    logger.info(
        "Accuracy summary for %s: %d races | Top-1: %.1f%% | Top-3: %.1f%%",
        race_date, races_n, top1_acc * 100, (top3_acc or 0) * 100,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Fetch Equibase race results and build accuracy report")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date",  type=str, help="Race date YYYY-MM-DD")
    group.add_argument("--days",  type=int, default=1,
                       help="Fetch results for the last N days (default: 1 = yesterday)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--no-accuracy", action="store_true",
                        help="Skip building accuracy report (just fetch results)")
    args = parser.parse_args()

    dates: list[str] = []
    if args.date:
        dates = [args.date]
    else:
        today = datetime.now(timezone.utc).date()
        dates = [
            (today - timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range(args.days)
        ]

    for date_str in dates:
        logger.info("=== %s ===", date_str)
        results = fetch_results_for_date(date_str, force=args.force)
        if results and not args.no_accuracy:
            build_accuracy_report(date_str)


if __name__ == "__main__":
    main()
