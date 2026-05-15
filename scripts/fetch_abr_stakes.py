"""
fetch_abr_stakes.py — scrape the America's Best Racing graded-stakes calendar.

Source: https://www.americasbestracing.net/horse-racing-events/stakes-schedule
Coverage: Full-year G1/G2/G3 (and selected listed) US thoroughbred stakes.
Output:  data/raw/abr_stakes_YYYY.json   (cached per year)
         data/processed/abr_stakes_YYYY.csv

Usage:
    python scripts/fetch_abr_stakes.py              # current year
    python scripts/fetch_abr_stakes.py --year 2027  # specific year
    python scripts/fetch_abr_stakes.py --force       # re-fetch even if cached
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] fetch_abr_stakes: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fetch_abr_stakes")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"
PROC_DIR  = REPO_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ABR_URL = "https://www.americasbestracing.net/races"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

GRADE_MAP = {
    "grade i": "G1", "grade 1": "G1", "g1": "G1", "gi": "G1",
    "grade ii": "G2", "grade 2": "G2", "g2": "G2", "gii": "G2",
    "grade iii": "G3", "grade 3": "G3", "g3": "G3", "giii": "G3",
    "listed": "Listed",
}

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def _parse_grade(text: str) -> str:
    """Extract grade label from any string containing it."""
    lower = text.lower()
    for key, val in GRADE_MAP.items():
        if key in lower:
            return val
    return ""


def _parse_purse(text: str) -> int:
    """Return integer purse value from strings like '$750,000', '750000', etc."""
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else 0


def _parse_dot_date(raw: str, year: int) -> str:
    """
    Parse dates like '5.15' (month.day) or '5.09' as used on the ABR /races page.
    Returns YYYY-MM-DD or '' on failure.
    """
    m = re.match(r"^(\d{1,2})\.(\d{1,2})$", raw.strip())
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def _parse_date(raw_date: str, year: int) -> str:
    """
    Parse date strings: 'May 17', 'June 5-7', '5.15', 'March 15'.
    Returns YYYY-MM-DD of the first date found, or '' if not parseable.
    """
    raw = _clean(raw_date)
    # Dot format: "5.15"
    dot = _parse_dot_date(raw, year)
    if dot:
        return dot
    # "Month Day" or "Month Day-Day"
    m = re.match(r"(\w+)\s+(\d+)", raw, re.I)
    if m:
        month_name = m.group(1).lower()
        day = int(m.group(2))
        month_num = MONTH_MAP.get(month_name)
        if month_num:
            try:
                return datetime(year, month_num, day).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return ""


# ---------------------------------------------------------------------------
# ABR /races page parser
# ---------------------------------------------------------------------------

# Known US graded stakes for grade enrichment (supplement what appears in race names)
KNOWN_GRADES: dict[str, str] = {
    "kentucky derby": "G1", "preakness": "G1", "belmont stakes": "G1",
    "breeders' cup classic": "G1", "breeders cup classic": "G1",
    "travers": "G1", "santa anita derby": "G1", "blue grass": "G1",
    "florida derby": "G1", "wood memorial": "G1", "arkansas derby": "G1",
    "santa anita handicap": "G1", "pacific classic": "G1",
    "whitney": "G1", "woodward": "G1", "jockey club gold cup": "G1",
    "beldame": "G1", "distaff": "G1", "turf": "G1", "sprint": "G1",
    "juvenile": "G1", "juvenile fillies": "G1", "mile": "G1",
    "filly and mare turf": "G1", "filly and mare sprint": "G1",
    "dirt mile": "G1", "marathon": "G1",
    "peter pan": "G2", "ohio derby": "G2", "west virginia derby": "G2",
    "ruffian": "G2", "black-eyed susan": "G2", "pimlico special": "G2",
    "brooklyn": "G2", "monmouth stakes": "G2", "san antonio": "G2",
    "san pasqual": "G2", "strub": "G2", "charles whittingham": "G2",
    "san bernardino": "G2", "senorita": "G2",
}


def _enrich_grade(race_name: str, grade_from_page: str) -> str:
    """Return grade from page text, or look it up in KNOWN_GRADES, or infer from name."""
    if grade_from_page:
        return grade_from_page
    lower = race_name.lower()
    for key, val in KNOWN_GRADES.items():
        if key in lower:
            return val
    # Grade often in parentheses: "Santa Anita Derby (G1)"
    m = re.search(r"\(G([123])\)", race_name, re.I)
    if m:
        return f"G{m.group(1)}"
    return ""


def parse_stakes_page(html: str, year: int) -> list[dict]:
    """
    Parse the ABR /races page (Playwright-rendered).

    DOM structure (as of 2026):
      <div class="race align-middle expandable">
        <a class="image-link" href="https://…/races/2026-race-slug">  ← empty text
        <div class="name">
          <strong class="serif"><a href="https://…/races/2026-slug">Race Name</a></strong>
          <a class="sub" href="https://…/tracks/track-slug">Track Name</a>
        </div>
        <div class="post-time text-right">
          <strong>5:17 PM EDT</strong>
          <span class="inactive sub">FanDuel TV</span>
        </div>
      </div>

    Date comes from the parent <div class="row collapse"> text prefix, e.g. "5.15".
    """
    soup = BeautifulSoup(html, "html.parser")
    stakes: list[dict] = []

    # ── Strategy 1: rendered JS structure (div.race.expandable) ─────────────
    race_items = soup.find_all(
        "div",
        class_=lambda c: c and "race" in c and "expandable" in c
    )

    for item in race_items:
        # Race name: inside div.name > strong > a
        name_div = item.find("div", class_="name")
        race_name = ""
        race_url = ""
        track = ""
        if name_div:
            name_link = name_div.find("a")
            if name_link:
                race_name = _clean(name_link.get_text())
                race_url = name_link.get("href", "")
            track_link = name_div.find("a", class_="sub")
            if track_link:
                track = _clean(track_link.get_text())

        if not race_name:
            # Fallback: extract from any race link in this item
            for a in item.find_all("a", href=True):
                href = a.get("href", "")
                if "/races/20" in href and _clean(a.get_text()):
                    race_name = _clean(a.get_text())
                    race_url = href
                    break

        if not race_name:
            continue

        # Extract year from URL slug
        slug_year_m = re.search(r"/races/(\d{4})-", race_url)
        race_year = int(slug_year_m.group(1)) if slug_year_m else year

        # Date extraction — three strategies (inner-first):
        # 1. span.date in the race's own "row collapse" (Upcoming section layout)
        # 2. Walk back through preceding sibling "row collapse" divs to find one with span.date
        #    (monthly calendar: a header "row collapse" contains the date for following race rows)
        # 3. First span.date in the containing group div
        date_str = ""
        row_div = item.find_parent("div", class_=re.compile(r"\bcollapse\b"))

        if row_div:
            # Strategy 1: date span in same row
            date_span = row_div.find("span", class_="date")
            if date_span:
                date_str = _parse_dot_date(_clean(date_span.get_text()), race_year)

            # Strategy 2: walk back to previous sibling row that has a date span
            if not date_str:
                for sib in row_div.previous_siblings:
                    if not isinstance(sib, Tag):
                        continue
                    sib_date = sib.find("span", class_="date")
                    if sib_date:
                        candidate = _parse_dot_date(_clean(sib_date.get_text()), race_year)
                        if candidate:
                            date_str = candidate
                            break

            # Strategy 3: first span.date in the containing group div
            if not date_str:
                group_div = row_div.parent
                if group_div:
                    all_date_spans = group_div.find_all("span", class_="date")
                    if all_date_spans:
                        date_str = _parse_dot_date(
                            _clean(all_date_spans[0].get_text()), race_year
                        )

        # Grade: infer from race name (ABR page doesn't show grade inline)
        grade = _enrich_grade(race_name, _parse_grade(race_name))

        stakes.append({
            "date": date_str,
            "race_name": race_name,
            "track": track,
            "grade": grade,
            "purse": 0,
            "purse_raw": "",
            "distance": "",
            "surface": "",
            "year": race_year,
            "source": "ABR",
            "abr_url": race_url if race_url.startswith("http") else (
                f"https://www.americasbestracing.net{race_url}" if race_url else ""
            ),
        })

    if stakes:
        return [s for s in stakes if s.get("race_name")]

    # ── Strategy 2: table-based layout (older years) ─────────────────────────
    tables = soup.find_all("table")
    for table in tables:
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [_clean(th.get_text()).lower() for th in header_row.find_all(["th", "td"])]
        col = {}
        for i, h in enumerate(headers):
            if "date" in h:
                col.setdefault("date", i)
            elif "race" in h or "name" in h or "stakes" in h:
                col.setdefault("race", i)
            elif "track" in h or "venue" in h:
                col.setdefault("track", i)
            elif "grade" in h:
                col.setdefault("grade", i)
            elif "purse" in h:
                col.setdefault("purse", i)
            elif "dist" in h:
                col.setdefault("distance", i)
            elif "surface" in h or "turf" in h or "dirt" in h:
                col.setdefault("surface", i)
        if not col:
            continue
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            texts = [_clean(c.get_text()) for c in cells]
            def _get(key, texts=texts, col=col):
                idx = col.get(key)
                return texts[idx] if idx is not None and idx < len(texts) else ""
            race_name = _get("race")
            if not race_name:
                continue
            grade_raw = _get("grade")
            grade = _enrich_grade(race_name, _parse_grade(grade_raw))
            stakes.append({
                "date": _parse_date(_get("date"), year),
                "race_name": race_name,
                "track": _get("track"),
                "grade": grade,
                "purse": _parse_purse(_get("purse")),
                "purse_raw": _get("purse"),
                "distance": _get("distance"),
                "surface": _normalize_surface(_get("surface")),
                "year": year,
                "source": "ABR",
                "abr_url": "",
            })

    return [s for s in stakes if s.get("race_name")]


def _parse_div_rows(soup: BeautifulSoup, year: int) -> list[dict]:
    """Fallback parser for generic div/list layouts."""
    stakes = []
    candidates = soup.find_all(
        lambda tag: tag.name in ("li", "div", "p") and
        re.search(r"\bstakes\b", tag.get_text(), re.I) and
        re.search(r"\b(january|february|march|april|may|june|july|august|"
                  r"september|october|november|december)\b",
                  tag.get_text(), re.I)
    )
    for el in candidates:
        text = _clean(el.get_text())
        date_str = _parse_date(text, year)
        grade = _parse_grade(text)
        parts = re.split(r"[|—\-]{1,3}", text)
        race_name = _clean(parts[1]) if len(parts) > 1 else text[:80]
        track = _clean(parts[2]) if len(parts) > 2 else ""
        purse_m = re.search(r"\$[\d,]+", text)
        purse_raw = purse_m.group() if purse_m else ""
        stakes.append({
            "date": date_str,
            "race_name": race_name,
            "track": track,
            "grade": grade,
            "purse": _parse_purse(purse_raw),
            "purse_raw": purse_raw,
            "distance": "",
            "surface": "",
            "year": year,
            "source": "ABR",
        })
    return stakes


def _normalize_surface(raw: str) -> str:
    raw = raw.lower()
    if "turf" in raw or "grass" in raw:
        return "Turf"
    if "synth" in raw or "tapeta" in raw or "poly" in raw:
        return "Synthetic"
    if "dirt" in raw:
        return "Dirt"
    return raw.title() if raw else ""


# ---------------------------------------------------------------------------
# Playwright fetch — iterates through all month tabs to get full year
# ---------------------------------------------------------------------------


def _fetch_html_playwright(url: str, year: int, timeout_ms: int = 45_000) -> list[str]:
    """
    Fetch the ABR /races page using Playwright, selecting each month from the dropdown.

    The ABR filter bar has a <select> with relative month offsets:
      "none" = All Months, "0" = current month, "1" = next month, etc.

    Returns a list of HTML snapshots (one per month that was successfully loaded).
    Falls back to a single page snapshot if month navigation is unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        logger.warning("Playwright not installed; cannot render JS pages.")
        return []

    html_pages: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
            )
            page = context.new_page()

            # Initial load
            try:
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                page.wait_for_timeout(2_000)
            except PWTimeout:
                logger.warning("Initial page load timed out — using partial content")
            except Exception as exc:
                logger.warning("Playwright navigation error: %s", exc)

            # Capture the initial view (Upcoming + current month)
            try:
                html_pages.append(page.content())
                logger.debug("Captured initial page")
            except Exception:
                pass

            # Find the month <select> and iterate through all options
            # ABR uses relative offset values: "0"=current month, "1"=next, etc.
            try:
                select_el = page.locator("li.month-filter select").first
                options = select_el.locator("option").all()
                option_values = []
                for opt in options:
                    val = opt.get_attribute("value")
                    label = (opt.text_content() or "").strip()
                    if val and val != "none":
                        option_values.append((val, label))
                logger.info("Found %d month options: %s",
                            len(option_values),
                            [lbl for _, lbl in option_values])
            except Exception as exc:
                logger.warning("Could not read month select options: %s", exc)
                option_values = []

            for val, label in option_values:
                try:
                    select_el.select_option(val)
                    page.wait_for_timeout(1_500)
                    html_pages.append(page.content())
                    logger.info("Captured month: %s (value=%s)", label, val)
                except PWTimeout:
                    logger.warning("Timeout selecting month %s — skipping", label)
                except Exception as exc:
                    logger.warning("Could not select month %s: %s", label, exc)

            browser.close()
    except Exception as exc:
        logger.error("Playwright launch failed: %s", exc)

    logger.info("Playwright captured %d HTML snapshots", len(html_pages))
    return html_pages


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_abr_stakes(year: int, force: bool = False) -> list[dict]:
    """Fetch ABR stakes calendar for the given year, using cached file if available."""
    cache_path = RAW_DIR / f"abr_stakes_{year}.json"

    if cache_path.exists() and not force:
        logger.info("Loading cached ABR stakes for %d from %s", year, cache_path)
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("stakes", [])

    logger.info("Fetching ABR stakes calendar for %d from %s", year, ABR_URL)

    # Step 1: Try fast requests fetch first
    html_pages: list[str] = []
    try:
        resp = requests.get(ABR_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        static_html = resp.text
    except requests.RequestException as exc:
        logger.warning("requests fetch failed: %s — will try Playwright", exc)
        static_html = ""

    # Step 2: Parse and check for race data
    stakes = parse_stakes_page(static_html, year) if static_html else []

    # Step 3: If no races found (JS-rendered site), use Playwright with month iteration
    if not stakes:
        logger.info("Static HTML has no race data — fetching with Playwright (iterating months)…")
        html_pages = _fetch_html_playwright(ABR_URL, year)
        if html_pages:
            # Merge races from all monthly snapshots, deduplicating by race name + date
            seen: set[tuple[str, str]] = set()
            for html in html_pages:
                for s in parse_stakes_page(html, year):
                    key = (s["race_name"], s.get("date", ""))
                    if key not in seen:
                        seen.add(key)
                        stakes.append(s)

    # Save representative raw HTML for debugging (first non-empty page)
    raw_html_path = RAW_DIR / f"abr_stakes_{year}_raw.html"
    for html in html_pages or ([static_html] if static_html else []):
        if html:
            raw_html_path.write_text(html, encoding="utf-8")
            logger.info("Saved raw HTML sample to %s (%d bytes)", raw_html_path, len(html))
            break

    logger.info("Parsed %d stakes entries for %d", len(stakes), year)

    if not stakes:
        logger.warning(
            "No stakes parsed after Playwright render — ABR may have changed layout. "
            "Check %s for the raw HTML.", raw_html_path
        )
        return []

    # Sort by date, then grade priority (G1 first), then purse
    grade_order = {"G1": 0, "G2": 1, "G3": 2, "Listed": 3, "": 4}
    stakes.sort(key=lambda s: (
        s.get("date") or "9999",
        grade_order.get(s.get("grade", ""), 4),
        -(s.get("purse") or 0),
    ))

    # Save structured JSON
    payload = {
        "year": year,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(stakes),
        "stakes": stakes,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d stakes to %s", len(stakes), cache_path)

    # Save CSV
    csv_path = PROC_DIR / f"abr_stakes_{year}.csv"
    _write_csv(stakes, csv_path)

    return stakes


def _write_csv(stakes: list[dict], path: Path) -> None:
    import csv
    if not stakes:
        return
    fieldnames = ["date", "race_name", "track", "grade", "purse", "purse_raw",
                  "distance", "surface", "year", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stakes)
    logger.info("CSV saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch ABR graded-stakes calendar")
    parser.add_argument("--year", type=int, default=datetime.now().year,
                        help="Year to fetch (default: current year)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cached")
    args = parser.parse_args()

    stakes = fetch_abr_stakes(year=args.year, force=args.force)
    if stakes:
        grades = {}
        for s in stakes:
            g = s.get("grade") or "Ungraded"
            grades[g] = grades.get(g, 0) + 1
        logger.info("Summary: %s", " | ".join(f"{k}: {v}" for k, v in sorted(grades.items())))
    else:
        logger.warning("No stakes retrieved. Check ABR page structure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
