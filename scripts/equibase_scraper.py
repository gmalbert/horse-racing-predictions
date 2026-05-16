"""
Equibase Scraper
================
Scrapes race cards, results, entries, and horse stats from equibase.com.

Uses Playwright headless Chromium to bypass Imperva bot-protection
(cloudscraper only handles Cloudflare; Equibase uses Imperva).

URL patterns (discovered from live Equibase HTML, May 2026):
  Entries index:   /static/entry/index.html
  Track entry idx: /static/entry/RaceCardIndex{TRACK}{MMDDYY}{COUNTRY}-EQB.html
  Per-race entry:  /static/entry/{TRACK}{MMDDYY}{COUNTRY}{RACE}-EQB.html

  Results index:   /static/chart/summary/index.html
  Track result idx:/static/chart/summary/RaceCardIndex{TRACK}{MMDDYY}{COUNTRY}-EQB.html
  Per-race result: /static/chart/summary/{TRACK}{MMDDYY}{COUNTRY}{RACE}-EQB.html
  Full race chart: /static/chart/gps/{TRACK}{MMDDYY}{COUNTRY}{RACE}.html  (all finishers, GPS times)

  Horse profile:   /profiles/Results.cfm?type=Horse&refno={ID}&registry=T&rbt=TB

Usage:
    python equibase_scraper.py --mode entries --date 2025-05-15
    python equibase_scraper.py --mode results --date 2025-05-14
    python equibase_scraper.py --mode horse --name "Justify"
    python equibase_scraper.py --mode chart --track AQU --date 2025-05-14 --race 3

Requirements:
    pip install playwright beautifulsoup4 lxml
    playwright install chromium
"""

import argparse
import csv
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("equibase")

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://www.equibase.com"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class Entry:
    track: str = ""
    race_number: str = ""
    race_date: str = ""
    post_position: str = ""
    horse_name: str = ""
    jockey: str = ""
    trainer: str = ""
    morning_line_odds: str = ""
    weight: str = ""
    equipment: str = ""
    horse_profile_url: str = ""  # /profiles/Results.cfm?type=Horse&refno=...


@dataclass
class RaceResult:
    track: str = ""
    race_number: str = ""
    race_date: str = ""
    finish_position: str = ""
    program_number: str = ""
    horse_name: str = ""
    jockey: str = ""
    trainer: str = ""       # not available on free static pages
    final_odds: str = ""    # win payout from summary page (W/P/S only)
    place_payout: str = ""
    show_payout: str = ""
    margin: str = ""        # not available on free static pages
    final_time: str = ""    # per-horse GPS time


@dataclass
class HorseStat:
    horse_name: str = ""
    sire: str = ""
    dam: str = ""
    owner: str = ""
    trainer: str = ""
    starts: str = ""
    wins: str = ""
    places: str = ""
    shows: str = ""
    earnings: str = ""


@dataclass
class ChartRunner:
    finish: str = ""
    program_number: str = ""
    post: str = ""
    horse: str = ""
    jockey: str = ""
    weight: str = ""           # not available on GPS page
    odds: str = ""             # win payout from summary (W/P/S only)
    place_payout: str = ""
    show_payout: str = ""
    final_time: str = ""       # per-horse GPS final time
    running_positions: list = field(default_factory=list)   # position at each call
    fraction_splits: list = field(default_factory=list)     # horse's time at each call
    fractional_times: list = field(default_factory=list)    # leader time at each call (from header)
    comment: str = ""


@dataclass
class CombinedRecord:
    """Entry fields merged with result fields for a single horse in a single race."""
    track: str = ""
    race_number: str = ""
    race_date: str = ""
    # Entry data
    post_position: str = ""
    program_number: str = ""   # from results
    horse_name: str = ""
    jockey: str = ""
    trainer: str = ""          # from entries
    morning_line_odds: str = ""
    weight: str = ""
    equipment: str = ""
    horse_profile_url: str = ""
    # Result data
    finish_position: str = ""
    final_odds: str = ""       # win payout
    place_payout: str = ""
    show_payout: str = ""
    margin: str = ""           # not available on static pages
    final_time: str = ""


# ── Session ───────────────────────────────────────────────────────────────────
class EquibaseSession:
    """
    Playwright-backed session that bypasses Imperva bot protection.
    Lazy-initialises the browser on first use.  Use as a context manager
    (``with EquibaseSession() as s: ...``) or call ``close()`` when done.
    """

    def __init__(self, delay: float = 2.0, jitter: float = 1.0, headless: bool = True):
        self.delay = delay
        self.jitter = jitter
        self.headless = headless
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False
        self._last_request: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def _ensure_started(self):
        if self._initialized:
            return
        from playwright.sync_api import sync_playwright  # local import keeps startup fast
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=_USER_AGENT,
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        self._page = self._context.new_page()
        # Warm-up: load homepage then entries index so Imperva session score builds up
        log.info("Starting Playwright browser…")
        self._page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(3)
        self._page.goto(f"{BASE_URL}/static/entry/index.html", wait_until="networkidle", timeout=30_000)
        time.sleep(2)
        self._initialized = True

    def close(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._initialized = False

    def __enter__(self):
        self._ensure_started()
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Request ───────────────────────────────────────────────────────────────
    def get(self, url: str, params: Optional[dict] = None) -> BeautifulSoup:
        self._ensure_started()
        elapsed = time.time() - self._last_request
        wait = self.delay + random.uniform(0, self.jitter) - elapsed
        if wait > 0:
            time.sleep(wait)

        full_url = f"{url}?{urlencode(params)}" if params else url
        log.info("GET %s", full_url)
        try:
            self._page.goto(full_url, wait_until="networkidle", timeout=30_000)
        except Exception as exc:
            raise RuntimeError(f"Playwright navigation error on {full_url}: {exc}") from exc
        time.sleep(0.5)
        self._last_request = time.time()

        content = self._page.content()
        if "Pardon Our Interruption" in content:
            raise RuntimeError(f"Imperva block on {full_url}")

        return BeautifulSoup(content, "lxml")


# ── Scrapers ──────────────────────────────────────────────────────────────────
class EquibaseScraper:
    """
    High-level scraper with methods for each data type.
    All public methods return a list of dataclass instances.
    """

    def __init__(self, session: Optional[EquibaseSession] = None):
        self.session = session or EquibaseSession()

    # ── URL helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _date_code(race_date: date) -> str:
        """Return MMDDYY string, e.g. '051526' for 2026-05-15."""
        return race_date.strftime("%m%d%y")

    @staticmethod
    def _fix_href(href: str) -> str:
        """Equibase sometimes uses backslashes in href; normalise them."""
        return href.replace("\\", "/")

    def _full_url(self, path: str) -> str:
        path = self._fix_href(path)
        return path if path.startswith("http") else BASE_URL + path

    # ── Entries ──────────────────────────────────────────────────────────────
    def get_entries(self, race_date: date, track: Optional[str] = None) -> list[Entry]:
        """
        Fetch race entries for a given date.
        track: 3-letter track code e.g. "CD".  None = all tracks.
        """
        if track:
            return self._get_entries_for_track(race_date, track)

        # All tracks: parse main index to find links for this date
        date_code = self._date_code(race_date)
        soup = self.session.get(f"{BASE_URL}/static/entry/index.html")
        all_entries: list[Entry] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = self._fix_href(a["href"])
            if "RaceCardIndex" not in href or date_code not in href or "entry" not in href:
                continue
            track_code, country = self._parse_track_country(href, date_code)
            if not track_code:
                continue
            key = f"{track_code}-{country}"
            if key in seen:
                continue
            seen.add(key)
            all_entries.extend(self._get_entries_for_track(race_date, track_code, country))

        return all_entries

    def _get_entries_for_track(
        self, race_date: date, track: str, country: str = "USA"
    ) -> list[Entry]:
        date_code = self._date_code(race_date)
        index_url = (
            f"{BASE_URL}/static/entry/RaceCardIndex{track}{date_code}{country}-EQB.html"
        )
        try:
            index_soup = self.session.get(index_url)
        except RuntimeError as exc:
            log.warning("Skipping %s: %s", index_url, exc)
            return []

        entries: list[Entry] = []
        table = index_soup.find("table")
        if not table:
            log.warning("No race table at %s", index_url)
            return entries

        # Columns: Race, Purse, Race Type, Distance, Surface, Starters, Est. Post, Free Tools
        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells:
                continue
            race_info = {
                "race_number": cells[0],
                "purse":       cells[1] if len(cells) > 1 else "",
                "race_type":   cells[2] if len(cells) > 2 else "",
                "distance":    cells[3] if len(cells) > 3 else "",
                "surface":     cells[4] if len(cells) > 4 else "",
            }

            # Find per-race entry href (contains TRACK+date_code but not RaceCardIndex)
            entry_href = None
            for a in row.find_all("a", href=True):
                h = a["href"]
                if (
                    "entry" in h.lower()
                    and f"{track}{date_code}" in h
                    and "RaceCardIndex" not in h
                ):
                    entry_href = h
                    break
            if not entry_href:
                continue

            try:
                entry_soup = self.session.get(self._full_url(entry_href))
            except RuntimeError as exc:
                log.warning("Skipping race %s: %s", race_info["race_number"], exc)
                continue

            entries.extend(
                self._parse_entry_table(entry_soup, track, race_date.isoformat(), race_info)
            )

        return entries

    def _parse_entry_table(
        self,
        soup: BeautifulSoup,
        track: str,
        race_date: str,
        race_info: dict,
    ) -> list[Entry]:
        """
        Parse a per-race entry page.
        Expected columns: P#, PP, Horse, VS, A/S, Med, Claim $, Jockey, Wgt, Trainer, M/L, LiveOdds
        """
        entries: list[Entry] = []
        table = soup.find("table")
        if not table:
            return entries

        headers = [" ".join(th.get_text(strip=True).split()).lower() for th in table.find_all("th")]
        col = {h: i for i, h in enumerate(headers)}

        for row in table.find_all("tr"):
            cells = [" ".join(td.get_text(strip=True).split()) for td in row.find_all("td")]
            if len(cells) < 3:
                continue

            def c(key: str, fallback: str = "") -> str:
                return cells[col[key]] if key in col and col[key] < len(cells) else fallback

            # Strip breeding state "(KY)" from horse name
            horse_raw = c("horse") or (cells[2] if len(cells) > 2 else "")
            horse_name = horse_raw.split("(")[0].strip()

            # Extract horse profile URL (contains refno) from the horse cell's <a> tag
            horse_profile_url = ""
            horse_col = col.get("horse", 2)
            horse_td_list = row.find_all("td")
            if horse_col < len(horse_td_list):
                link = horse_td_list[horse_col].find("a", href=True)
                if link:
                    horse_profile_url = self._fix_href(link["href"])

            entries.append(Entry(
                track=track,
                race_number=race_info.get("race_number", ""),
                race_date=race_date,
                post_position=c("pp"),
                horse_name=horse_name,
                jockey=c("jockey"),
                trainer=c("trainer"),
                morning_line_odds=c("m/l"),
                weight=c("wgt"),
                equipment=c("med"),
                horse_profile_url=horse_profile_url,
            ))

        log.info(
            "Parsed %d entries — %s R%s",
            len(entries), track, race_info.get("race_number", "?"),
        )
        return entries

    # ── Results ──────────────────────────────────────────────────────────────
    def get_results(self, race_date: date, track: Optional[str] = None) -> list[RaceResult]:
        """Fetch race results for a given date (and optional track).
        All finishers are returned via the GPS chart page; Win/Place/Show payouts
        are merged from the summary page.  Trainer and margin are not available
        on Equibase's free static pages.
        """
        if track:
            return self._get_results_for_track(race_date, track)

        date_code = self._date_code(race_date)
        soup = self.session.get(f"{BASE_URL}/static/chart/summary/index.html")
        all_results: list[RaceResult] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = self._fix_href(a["href"])
            if "RaceCardIndex" not in href or date_code not in href or "summary" not in href:
                continue
            track_code, country = self._parse_track_country(href, date_code)
            if not track_code:
                continue
            key = f"{track_code}-{country}"
            if key in seen:
                continue
            seen.add(key)
            all_results.extend(self._get_results_for_track(race_date, track_code, country))

        return all_results

    def _get_results_for_track(
        self, race_date: date, track: str, country: str = "USA"
    ) -> list[RaceResult]:
        date_code = self._date_code(race_date)
        index_url = (
            f"{BASE_URL}/static/chart/summary/RaceCardIndex{track}{date_code}{country}-EQB.html"
        )
        try:
            index_soup = self.session.get(index_url)
        except RuntimeError as exc:
            log.warning("Skipping %s: %s", index_url, exc)
            return []

        results: list[RaceResult] = []
        table = index_soup.find("table")
        if not table:
            return results

        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells:
                continue
            race_number = cells[0]
            if not race_number.isdigit():
                continue

            # GPS page: all finishers
            gps_url = (
                f"{BASE_URL}/static/chart/gps/"
                f"{track}{date_code}{country}{race_number}.html"
            )
            try:
                gps_soup = self.session.get(gps_url)
            except RuntimeError as exc:
                log.warning("GPS page unavailable for %s R%s: %s", track, race_number, exc)
                continue

            runners = self._parse_gps_table(gps_soup)
            if not runners:
                log.warning("No GPS runners parsed — %s R%s", track, race_number)
                continue

            # Summary page: Win/Place/Show payouts
            summary_url = (
                f"{BASE_URL}/static/chart/summary/"
                f"{track}{date_code}{country}{race_number}-EQB.html"
            )
            payouts: dict[str, tuple[str, str, str]] = {}
            try:
                summary_soup = self.session.get(summary_url)
                payouts = self._parse_summary_payouts(summary_soup)
            except RuntimeError:
                pass

            for finish_pos, runner in enumerate(runners, 1):
                horse_key = runner.horse.lower()
                win_p, place_p, show_p = payouts.get(horse_key, ("", "", ""))
                results.append(RaceResult(
                    track=track,
                    race_number=race_number,
                    race_date=race_date.isoformat(),
                    finish_position=str(finish_pos),
                    program_number=runner.program_number,
                    horse_name=runner.horse,
                    jockey=runner.jockey,
                    final_odds=win_p,
                    place_payout=place_p,
                    show_payout=show_p,
                    final_time=runner.final_time,
                ))

            log.info("Parsed %d results — %s R%s", len(runners), track, race_number)

        return results

    # ── Horse Stats ───────────────────────────────────────────────────────────
    def get_horse_stats(self, horse_name: str, refno: Optional[str] = None) -> list[HorseStat]:
        """
        Fetch a horse's profile and return its stats.

        ``refno`` is the Equibase numeric horse ID found in entry/result links
        (e.g. ``11153956`` from ``/profiles/Results.cfm?type=Horse&refno=11153956``).
        If omitted, a name-search page is attempted but may yield limited data.
        """
        if refno:
            url = f"{BASE_URL}/profiles/Results.cfm"
            params = {"type": "Horse", "refno": refno, "registry": "T", "rbt": "TB"}
        else:
            # Equibase search by name (returns a disambiguation/search results page)
            url = f"{BASE_URL}/profiles/Results.cfm"
            params = {"type": "Horse", "searchFor": horse_name, "searchType": "horseName"}
        soup = self.session.get(url, params=params)
        return self._parse_horse(soup, horse_name)

    def _parse_horse(self, soup: BeautifulSoup, horse_name: str) -> list[HorseStat]:
        """
        Parse the Equibase horse profile page.
        HTML landmarks (stable as of 2026):
        - h2.horse-name-header  → horse name with breeding state
        - h5.horse-profile-top-bar-headings (2nd occurrence) → "(Sire - Dam, by GranSire)"
        - p.horse-profile-top-bar-para → "Jockey: ... Trainer: ... Owner: ... Breeder: ..."
        - First career stats table: headers Starts/Firsts/Seconds/Thirds/Earnings
        """
        row_data = HorseStat(horse_name=horse_name)

        # Horse name
        name_h2 = soup.select_one("h2.horse-name-header")
        if name_h2:
            row_data.horse_name = " ".join(name_h2.get_text(strip=True).split())
            # Strip age suffix and "(KY)" breeding state
            row_data.horse_name = row_data.horse_name.split("(")[0].strip()

        # Sire / Dam
        pedigree_h5s = soup.select("h5.horse-profile-top-bar-headings")
        for h5 in pedigree_h5s:
            txt = h5.get_text(" ", strip=True)
            if txt.startswith("(") and " - " in txt:
                # Format: "( Sire - Dam , by GranSire )"
                inner = txt.strip("()").strip()
                parts = inner.split(" - ", 1)
                row_data.sire = parts[0].strip()
                if len(parts) > 1:
                    dam_part = parts[1].split(",")[0].strip()
                    row_data.dam = dam_part
                break

        # Connections (owner, trainer)
        para = soup.select_one("p.horse-profile-top-bar-para")
        if para:
            txt = para.get_text(" ", strip=True)
            for field, attr in [("Trainer:", "trainer"), ("Owner:", "owner")]:
                if field in txt:
                    after = txt.split(field, 1)[1].strip()
                    # Value ends at next field name or end of string
                    value = after.split("Trainer:")[0] if attr == "owner" else after
                    value = value.split("Owner:")[0] if attr == "trainer" else value
                    value = value.split("Breeder:")[0].strip()
                    setattr(row_data, attr, " ".join(value.split()))

        # Career stats: search all tables, prefer the one with a "Career" row
        career_cells: list[str] = []
        career_col: dict[str, int] = {}
        fallback_cells: list[str] = []
        fallback_col: dict[str, int] = {}

        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in table.select("th")]
            if "starts" not in headers:
                continue
            col = {h: i for i, h in enumerate(headers)}
            for row in table.select("tr"):
                cells = [" ".join(td.get_text(strip=True).split()) for td in row.select("td")]
                if not cells:
                    continue
                if cells[0].lower() == "career":
                    career_cells, career_col = cells, col
                    break
                # Keep the last valid numeric row as fallback
                starts_idx = col.get("starts", 0)
                if len(cells) >= 4 and starts_idx < len(cells) and cells[starts_idx].isdigit():
                    fallback_cells, fallback_col = cells, col

        def _apply_stats(cells: list[str], col: dict[str, int]) -> None:
            def _get(key: str, alt: str, default: int) -> str:
                idx = col.get(key, col.get(alt, default))
                return cells[idx] if idx < len(cells) else ""
            row_data.starts   = _get("starts", "starts", 0)
            row_data.wins     = _get("firsts", "wins", 1)
            row_data.places   = _get("seconds", "2nd", 2)
            row_data.shows    = _get("thirds", "3rd", 3)
            earn_idx = col.get("earnings", len(cells) - 1)
            row_data.earnings = cells[earn_idx] if earn_idx < len(cells) else ""

        if career_cells:
            _apply_stats(career_cells, career_col)
        elif fallback_cells:
            _apply_stats(fallback_cells, fallback_col)

        return [row_data]

    # ── GPS / Chart Parsing ───────────────────────────────────────────────────
    @staticmethod
    def _split_horse_jockey(combined: str) -> tuple[str, str]:
        """
        Split the concatenated HorseJockey string from GPS chart pages.
        Format: horse name immediately followed by jockey "LastName, FirstName"
        e.g. "Carcar ExpressCurtis, Ben" → ("Carcar Express", "Curtis, Ben")
        Uses a lookbehind for a lowercase letter immediately before an uppercase
        word followed by a comma (start of jockey's last name).
        """
        m = re.search(r"(?<=[a-z])([A-Z][a-zA-Z'\-]+),\s+([A-Za-z]+)", combined)
        if not m:
            # Fallback: split on last uppercase word + comma pattern
            m = re.search(r"([A-Z][a-zA-Z'\-]+),\s+([A-Za-z]+)$", combined)
        if m:
            return combined[: m.start()].strip(), combined[m.start() :].strip()
        return combined, ""

    @staticmethod
    def _parse_time_pos(cell: str) -> tuple[str, str]:
        """
        Parse a GPS time+position cell such as "1:48.161" → ("1:48.16", "1")
        or "24.711" → ("24.71", "1").  The final digit(s) after the 2-decimal
        time portion are the running/finish position.
        """
        m = re.match(r"^(.*\.\d{2})(\d+)$", cell)
        if m:
            return m.group(1), m.group(2)
        return cell, ""

    def _parse_gps_table(self, soup: BeautifulSoup) -> list[ChartRunner]:
        """
        Parse the Equibase GPS chart page.
        Table structure (2026): Result | No.(Post) | HorseJockey | frac_cols...
        Each data row is sorted by finish position (1st row = winner).
        Header cells encode the leader's fractional time, e.g. "1/424.71".
        """
        runners: list[ChartRunner] = []
        target_table = None

        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            h_lower = [h.lower() for h in headers]
            if "result" not in h_lower:
                continue
            if not any("horse" in h for h in h_lower):
                continue
            # Skip tables with no data rows (can occur in nested-table HTML)
            has_data = any(
                len(row.find_all("td")) >= 3
                for row in table.find_all("tr")
            )
            if not has_data:
                continue
            target_table = table
            break

        if not target_table:
            log.warning("GPS chart table not found")
            return runners

        headers = [th.get_text(strip=True) for th in target_table.find_all("th")]
        # Extract leader fractional times from header names, e.g. "1/424.71" → "24.71"
        leader_times: list[str] = []
        frac_names: list[str] = []
        for h in headers[3:]:  # skip Result, No.(Post), HorseJockey
            # Header format: "{fractionLabel}{leaderTime}", e.g. "1/424.71" or "Stretch1:41.21"
            # Time can be "M:SS.ff" (single-digit minutes) or "SS.ff" (exactly 2 leading digits)
            tm = re.search(r"([1-9]:\d{2}\.\d+|\d{2}\.\d+)$", h)
            frac_names.append(h)
            leader_times.append(tm.group(1) if tm else "")

        for row in target_table.find_all("tr"):
            cells_raw = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells_raw) < 3:
                continue

            # Result column: "1:48.161" → final_time="1:48.16", finish="1"
            final_time, finish = self._parse_time_pos(cells_raw[0])

            # No.(Post) column: "3(3)" → pgm="3", post="3"
            pgm_post = cells_raw[1]
            pp_m = re.match(r"^(\d+)\((\d+)\)$", pgm_post)
            pgm = pp_m.group(1) if pp_m else pgm_post
            post = pp_m.group(2) if pp_m else ""

            # HorseJockey column: split combined string
            horse, jockey = self._split_horse_jockey(cells_raw[2])

            # Fractional columns: each is "time+position"
            running_positions: list[str] = []
            fraction_splits: list[str] = []
            for cell in cells_raw[3:]:
                ft, rp = self._parse_time_pos(cell)
                fraction_splits.append(ft)
                running_positions.append(rp)

            runners.append(ChartRunner(
                finish=finish,
                program_number=pgm,
                post=post,
                horse=horse,
                jockey=jockey,
                final_time=final_time,
                running_positions=running_positions,
                fraction_splits=fraction_splits,
                fractional_times=leader_times,
            ))

        log.info("Parsed %d GPS chart runners", len(runners))
        return runners

    def _parse_summary_payouts(self, soup: BeautifulSoup) -> dict[str, tuple[str, str, str]]:
        """
        Parse the per-race summary page for Win/Place/Show dollar payouts.
        Returns {lowercase_horse_name: (win, place, show)} for W/P/S horses.
        """
        payouts: dict[str, tuple[str, str, str]] = {}
        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            if "horse" not in headers or "win" not in headers:
                continue
            col = {h: i for i, h in enumerate(headers)}
            for row in table.find_all("tr"):
                cells = [" ".join(td.get_text(strip=True).split()) for td in row.find_all("td")]
                if len(cells) < 2:
                    continue
                horse_key = cells[col.get("horse", 1)].lower()
                if not horse_key:
                    continue
                payouts[horse_key] = (
                    cells[col["win"]] if col.get("win", 99) < len(cells) else "",
                    cells[col["place"]] if col.get("place", 99) < len(cells) else "",
                    cells[col["show"]] if col.get("show", 99) < len(cells) else "",
                )
            break
        return payouts

    # ── Race Chart ────────────────────────────────────────────────────────────
    def get_chart(self, track: str, race_date: date, race_number: int) -> list[ChartRunner]:
        """
        Fetch the full race chart for a specific race.
        Uses the GPS page for all finishers with running positions and per-horse
        GPS times.  Win/Place/Show payouts are merged from the summary page.
        NOTE: GPS pages are only available after the race has been run.
        """
        date_code = self._date_code(race_date)
        for country in ("USA", "CAN"):
            gps_url = (
                f"{BASE_URL}/static/chart/gps/"
                f"{track}{date_code}{country}{race_number}.html"
            )
            try:
                gps_soup = self.session.get(gps_url)
                runners = self._parse_gps_table(gps_soup)
                if runners:
                    break
            except RuntimeError:
                continue
        else:
            log.warning("GPS chart not found for %s R%s %s", track, race_number, race_date)
            return []

        # Merge Win/Place/Show payouts from summary page
        summary_url = (
            f"{BASE_URL}/static/chart/summary/"
            f"{track}{date_code}USA{race_number}-EQB.html"
        )
        try:
            summary_soup = self.session.get(summary_url)
            payouts = self._parse_summary_payouts(summary_soup)
            for r in runners:
                win_p, place_p, show_p = payouts.get(r.horse.lower(), ("", "", ""))
                r.odds = win_p
                r.place_payout = place_p
                r.show_payout = show_p
        except RuntimeError:
            pass

        return runners

    # ── All races for a date ─────────────────────────────────────────────────
    def get_all_races(
        self,
        race_date: date,
        include_entries: bool = True,
    ) -> list[CombinedRecord]:
        """
        Fetch every race across every US track for *race_date*.

        Returns a list of CombinedRecord rows — one per horse per race.

        include_entries (default True):
            When True, also fetches the entry pages for pre-race fields
            (trainer, morning-line odds, weight, equipment).  Set False for
            historical dates where entry pages are likely gone — the result
            fields (finish, GPS time, W/P/S) are still returned, with entry
            fields left blank.

        NOTE: For past dates, entry pages *may* still exist as static files;
        the scraper will attempt them but fall back gracefully if they 404 or
        return a block.  Use include_entries=False to skip the attempt entirely
        and roughly halve the number of HTTP requests needed.
        """
        results = self.get_results(race_date, track=None)

        if not include_entries:
            # Wrap results as CombinedRecord with no entry data
            return [
                CombinedRecord(
                    track=r.track,
                    race_number=r.race_number,
                    race_date=r.race_date,
                    program_number=r.program_number,
                    horse_name=r.horse_name,
                    jockey=r.jockey,
                    finish_position=r.finish_position,
                    final_odds=r.final_odds,
                    place_payout=r.place_payout,
                    show_payout=r.show_payout,
                    margin=r.margin,
                    final_time=r.final_time,
                )
                for r in results
            ]

        # Merge entries for pre-race fields
        entries = self.get_entries(race_date, track=None)
        result_map = {
            (r.race_number, r.horse_name.lower()): r
            for r in results
        }
        combined: list[CombinedRecord] = []

        # Start with entries as the spine (covers scratches)
        seen: set[tuple[str, str]] = set()
        for e in entries:
            key = (e.race_number, e.horse_name.lower())
            seen.add(key)
            r = result_map.get(key)
            combined.append(CombinedRecord(
                track=e.track,
                race_number=e.race_number,
                race_date=e.race_date,
                post_position=e.post_position,
                program_number=r.program_number if r else "",
                horse_name=e.horse_name,
                jockey=e.jockey,
                trainer=e.trainer,
                morning_line_odds=e.morning_line_odds,
                weight=e.weight,
                equipment=e.equipment,
                horse_profile_url=e.horse_profile_url,
                finish_position=r.finish_position if r else "",
                final_odds=r.final_odds if r else "",
                place_payout=r.place_payout if r else "",
                show_payout=r.show_payout if r else "",
                margin=r.margin if r else "",
                final_time=r.final_time if r else "",
            ))

        # Append any finishers without a matching entry row (edge case)
        for r in results:
            key = (r.race_number, r.horse_name.lower())
            if key not in seen:
                combined.append(CombinedRecord(
                    track=r.track,
                    race_number=r.race_number,
                    race_date=r.race_date,
                    program_number=r.program_number,
                    horse_name=r.horse_name,
                    jockey=r.jockey,
                    finish_position=r.finish_position,
                    final_odds=r.final_odds,
                    place_payout=r.place_payout,
                    show_payout=r.show_payout,
                    margin=r.margin,
                    final_time=r.final_time,
                ))

        log.info(
            "get_all_races %s: %d entries + %d results → %d combined rows",
            race_date, len(entries), len(results), len(combined),
        )
        return combined

    # ── Combined (entries + results) ─────────────────────────────────────────
    def get_combined(self, race_date: date, track: str | None = None) -> list[CombinedRecord]:
        """
        Fetch entries and results for the given date (and optional track),
        then merge them by (race_number, horse_name) so each row has both
        pre-race entry data and post-race result data.
        Horses that scratched will still appear with empty result fields.
        """
        entries = self.get_entries(race_date, track)
        results = self.get_results(race_date, track)

        result_map: dict[tuple[str, str], RaceResult] = {
            (r.race_number, r.horse_name.lower()): r
            for r in results
        }

        combined: list[CombinedRecord] = []
        for e in entries:
            r = result_map.get((e.race_number, e.horse_name.lower()))
            combined.append(CombinedRecord(
                track=e.track,
                race_number=e.race_number,
                race_date=e.race_date,
                post_position=e.post_position,
                program_number=r.program_number if r else "",
                horse_name=e.horse_name,
                jockey=e.jockey,
                trainer=e.trainer,
                morning_line_odds=e.morning_line_odds,
                weight=e.weight,
                equipment=e.equipment,
                horse_profile_url=e.horse_profile_url,
                finish_position=r.finish_position if r else "",
                final_odds=r.final_odds if r else "",
                place_payout=r.place_payout if r else "",
                show_payout=r.show_payout if r else "",
                margin=r.margin if r else "",
                final_time=r.final_time if r else "",
            ))

        log.info("Combined %d entries with %d results → %d records", len(entries), len(results), len(combined))
        return combined

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_track_country(href: str, date_code: str) -> tuple[str, str]:
        """
        Extract track code and country from a RaceCardIndex href.
        E.g. '/static/entry/RaceCardIndexCD051526USA-EQB.html' → ('CD', 'USA')
        """
        fname = href.split("/")[-1]
        inner = fname.replace("RaceCardIndex", "").replace("-EQB.html", "")
        pos = inner.find(date_code)
        if pos < 0:
            return "", ""
        return inner[:pos], inner[pos + len(date_code):]


# ── Output Helpers ────────────────────────────────────────────────────────────
def save_csv(records: list, path: Path):
    if not records:
        log.warning("No records to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(records[0]).keys())
        writer.writeheader()
        for r in records:
            row = asdict(r)
            # Flatten list fields
            for k, v in row.items():
                if isinstance(v, list):
                    row[k] = "|".join(v)
            writer.writerow(row)
    log.info("Saved %d rows → %s", len(records), path)


def save_json(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    log.info("Saved %d records → %s", len(records), path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Equibase scraper")
    parser.add_argument(
        "--mode",
        choices=["entries", "results", "horse", "chart", "combined", "all-races"],
        required=True,
        help="What to scrape",
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    parser.add_argument("--track", default=None, help="3-letter track code, e.g. AQU")
    parser.add_argument("--race", type=int, default=1, help="Race number (for chart mode)")
    parser.add_argument("--name", default=None, help="Horse name (for horse mode)")
    parser.add_argument("--refno", default=None, help="Equibase horse ID for direct profile lookup (horse mode)")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--format", choices=["csv", "json", "both"], default="csv")
    parser.add_argument("--delay", type=float, default=2.0, help="Base delay between requests (sec)")
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="all-races mode: skip entry pages (faster; use for historical dates)",
    )
    args = parser.parse_args()

    race_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    session = EquibaseSession(delay=args.delay)
    scraper = EquibaseScraper(session)
    out_dir = Path(args.out)

    try:
        records: list = []
        stem = "unknown"

        if args.mode == "entries":
            records = scraper.get_entries(race_date, args.track)
            stem = f"entries_{args.date}_{args.track or 'ALL'}"

        elif args.mode == "results":
            records = scraper.get_results(race_date, args.track)
            stem = f"results_{args.date}_{args.track or 'ALL'}"

        elif args.mode == "horse":
            if not args.name and not args.refno:
                parser.error("--name or --refno is required for horse mode")
            name = args.name or args.refno
            records = scraper.get_horse_stats(name, refno=args.refno)
            stem = f"horse_{name.replace(' ', '_')}"

        elif args.mode == "chart":
            if not args.track:
                parser.error("--track is required for chart mode")
            records = scraper.get_chart(args.track, race_date, args.race)
            stem = f"chart_{args.track}_{args.date}_R{args.race}"

        elif args.mode == "combined":
            records = scraper.get_combined(race_date, args.track)
            stem = f"combined_{args.date}_{args.track or 'ALL'}"

        elif args.mode == "all-races":
            include_entries = not args.results_only
            records = scraper.get_all_races(race_date, include_entries=include_entries)
            stem = f"all_races_{args.date}"

        if args.format in ("csv", "both"):
            save_csv(records, out_dir / f"{stem}.csv")
        if args.format in ("json", "both"):
            save_json(records, out_dir / f"{stem}.json")

        print(f"\n✓ Done — {len(records)} records scraped.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
