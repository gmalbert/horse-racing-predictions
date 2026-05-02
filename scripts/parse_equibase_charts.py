"""
parse_equibase_charts.py
========================
Equibase result-chart PDF scraper + parser for the US horse racing prediction system.

What this file does
-------------------
1.  DISCOVER   – Finds chart PDF URLs on Equibase for any track/date combination.
2.  DOWNLOAD   – Downloads PDFs with polite rate-limiting and caches them locally
                 under  data/raw/us/charts/<YYYY>/<track_code>/
3.  PARSE      – Extracts every data field from the PDF text that is useful for
                 building a US betting model:
                    • Race header    (track, date, race#, distance, surface,
                                      conditions, purse, grade, restriction)
                    • Past-post odds / morning-line odds
                    • Full result    (finish position, horse name, jockey,
                                      trainer, owner, weight, medication/equip,
                                      Beyer Speed Figure, fractional times,
                                      final time, lengths beaten, payouts)
                    • Pace / sectional splits
                    • Payoffs        (Win/Place/Show + exotic payouts)
                    • Race footnotes (trip notes, track variant, weather/going)
4.  STORE      – Saves structured data to:
                    • data/processed/us_race_results.parquet  (full history)
                    • data/processed/us_pace_splits.parquet   (pace data)
                    • data/processed/us_payoffs.parquet       (payouts)
                    • data/processed/us_horses.parquet        (per-horse stats)

Usage
-----
    # Scrape all tracks for today
    python scripts/us/parse_equibase_charts.py --date today

    # Specific date
    python scripts/us/parse_equibase_charts.py --date 2025-05-03

    # Specific track
    python scripts/us/parse_equibase_charts.py --date 2025-05-03 --track CD

    # Parse already-downloaded PDFs only (no network)
    python scripts/us/parse_equibase_charts.py --parse-only --pdf-dir data/raw/us/charts

    # Historical backfill (slow – be polite to Equibase)
    python scripts/us/parse_equibase_charts.py --start 2020-01-01 --end 2024-12-31 --track SAR

Requirements (add to requirements.txt if missing)
--------------------------------------------------
    pdfplumber>=0.9.0
    requests>=2.31.0
    beautifulsoup4>=4.12.0
    pandas>=2.0.0
    pytz>=2023.3
    tenacity>=8.2.0       # retry logic
    tqdm>=4.66.0          # progress bars
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("equibase_scraper")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root
RAW_DIR = ROOT / "data" / "raw" / "us" / "charts"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Track reference table
# ---------------------------------------------------------------------------
# Maps Equibase 2-3 letter codes → human names + timezone
# Extend this dict as you add more tracks.
TRACK_CODES: dict[str, dict] = {
    # Triple Crown venues
    "CD":  {"name": "Churchill Downs",   "state": "KY", "tz": "America/Chicago"},
    "PIM": {"name": "Pimlico",           "state": "MD", "tz": "America/New_York"},
    "BEL": {"name": "Belmont Park",      "state": "NY", "tz": "America/New_York"},
    # Major US tracks
    "SAR": {"name": "Saratoga",          "state": "NY", "tz": "America/New_York"},
    "KEE": {"name": "Keeneland",         "state": "KY", "tz": "America/Chicago"},
    "SA":  {"name": "Santa Anita",       "state": "CA", "tz": "America/Los_Angeles"},
    "DMR": {"name": "Del Mar",           "state": "CA", "tz": "America/Los_Angeles"},
    "GP":  {"name": "Gulfstream Park",   "state": "FL", "tz": "America/New_York"},
    "AQU": {"name": "Aqueduct",          "state": "NY", "tz": "America/New_York"},
    "MTH": {"name": "Monmouth Park",     "state": "NJ", "tz": "America/New_York"},
    "OP":  {"name": "Oaklawn Park",      "state": "AR", "tz": "America/Chicago"},
    "FG":  {"name": "Fair Grounds",      "state": "LA", "tz": "America/Chicago"},
    "TMP": {"name": "Tampa Bay Downs",   "state": "FL", "tz": "America/New_York"},
    "TUP": {"name": "Turf Paradise",     "state": "AZ", "tz": "America/Phoenix"},
    "LRC": {"name": "Los Alamitos",      "state": "CA", "tz": "America/Los_Angeles"},
    "HAW": {"name": "Hawthorne",         "state": "IL", "tz": "America/Chicago"},
    "IND": {"name": "Indiana Grand",     "state": "IN", "tz": "America/Chicago"},
    "LAD": {"name": "Lone Star Park",    "state": "TX", "tz": "America/Chicago"},
    "LAP": {"name": "Louisiana Downs",   "state": "LA", "tz": "America/Chicago"},
    "TDN": {"name": "Thistledown",       "state": "OH", "tz": "America/New_York"},
    "PEN": {"name": "Penn National",     "state": "PA", "tz": "America/New_York"},
    "PRX": {"name": "Parx Racing",       "state": "PA", "tz": "America/New_York"},
    "CTR": {"name": "Canterbury Park",   "state": "MN", "tz": "America/Chicago"},
    "WO":  {"name": "Woodbine",          "state": "ON", "tz": "America/Toronto"},  # Canada
}

# ---------------------------------------------------------------------------
# HTTP session helpers
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


# ---------------------------------------------------------------------------
# Data classes – one per level of granularity
# ---------------------------------------------------------------------------

@dataclass
class RaceResult:
    """One row per horse per race."""
    # Identifiers
    track_code:          str = ""
    track_name:          str = ""
    race_date:           str = ""   # ISO YYYY-MM-DD
    race_number:         int = 0

    # Race conditions
    race_name:           str = ""   # e.g. "Kentucky Derby"
    grade:               Optional[int] = None   # 1 / 2 / 3 / None
    race_type:           str = ""   # Maiden / Claiming / Allowance / Stakes
    claiming_price:      Optional[float] = None # USD
    purse:               Optional[float] = None # USD
    distance_furlongs:   Optional[float] = None
    surface:             str = ""   # Dirt / Turf / Synthetic
    course:              str = ""   # Main / Inner turf / etc.
    conditions:          str = ""   # Full text condition line
    restrictions:        str = ""   # Age/sex restrictions
    weather:             str = ""
    track_condition:     str = ""   # Fast / Good / Sloppy / Yielding etc.
    track_variant:       Optional[int] = None   # Equibase track variant
    field_size:          int = 0

    # Fractional / final times  (seconds)
    fraction_1:          Optional[float] = None  # 1/4 mile split
    fraction_2:          Optional[float] = None  # 1/2 mile split
    fraction_3:          Optional[float] = None  # 3/4 mile split
    final_time:          Optional[float] = None  # Total race time
    pace_fig:            Optional[int]   = None

    # Per-horse result
    finish_position:     Optional[int]   = None  # Official finish; None = DNF/DQ
    finish_position_raw: str = ""                # e.g. "DQ", "DNF", "1"
    program_number:      str = ""
    horse_name:          str = ""
    jockey:              str = ""
    trainer:             str = ""
    owner:               str = ""
    breeder:             str = ""
    weight:              Optional[float] = None  # lbs
    medication:          str = ""   # L (Lasix), B (Bute), etc.
    equipment:           str = ""   # b (blinkers), f (front wraps), etc.
    is_lasix:            bool = False
    is_first_time_lasix: bool = False   # FTL flag (very predictive)
    age:                 Optional[int] = None
    sex:                 str = ""
    sire:                str = ""
    dam:                 str = ""
    dam_sire:            str = ""
    color:               str = ""
    foaling_date:        str = ""

    # Race-day odds
    morning_line:        Optional[float] = None  # Decimal ML odds
    post_time_odds:      Optional[float] = None  # Decimal odds at post
    morning_line_frac:   str = ""  # e.g. "5-2"
    post_time_odds_frac: str = ""

    # Running positions (by call)
    pos_start:           Optional[int] = None
    pos_first_call:      Optional[int] = None
    pos_second_call:     Optional[int] = None
    pos_stretch:         Optional[int] = None
    pos_finish:          Optional[int] = None
    lengths_back_start:  Optional[float] = None
    lengths_back_first:  Optional[float] = None
    lengths_back_second: Optional[float] = None
    lengths_back_stretch:Optional[float] = None
    lengths_back_finish: Optional[float] = None

    # Speed figure (Beyer)
    beyer_speed_fig:     Optional[int] = None
    equibase_speed_fig:  Optional[int] = None

    # Post position
    post_position:       Optional[int] = None

    # Claim info
    claimed:             bool = False
    claimed_by:          str = ""

    # Footnotes
    footnote:            str = ""   # Trip notes for this horse
    race_footnote:       str = ""   # General race footnote


@dataclass
class RacePayoff:
    """Payouts for a single race."""
    track_code:   str = ""
    race_date:    str = ""
    race_number:  int = 0
    bet_type:     str = ""   # WIN / PLACE / SHOW / EXACTA / TRIFECTA etc.
    combination:  str = ""   # e.g. "4" or "4-7" or "4-7-2"
    payout:       float = 0.0   # Per $2 base
    pool_total:   Optional[float] = None


@dataclass
class PaceSplit:
    """Fractional timing detail per race."""
    track_code:       str = ""
    race_date:        str = ""
    race_number:      int = 0
    horse_name:       str = ""
    split_name:       str = ""   # "1/4", "1/2", "3/4", "str", "fin"
    split_time:       Optional[float] = None
    split_position:   Optional[int]   = None
    lengths_back:     Optional[float] = None


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def build_chart_index_url(track_code: str, race_date: date) -> str:
    """
    Equibase free results page for a given track/date.
    Returns an HTML page that lists individual race chart PDF links.
    """
    params = {
        "TRACK_CODE":    track_code,
        "RACE_DATE":     race_date.strftime("%m/%d/%Y"),
        "RACE_TYPE":     "X",
        "CHART_OPTION":  "FINISH",
    }
    return f"https://www.equibase.com/premium/eqbPDFChartPlus.cfm?{urlencode(params)}"


def build_chart_pdf_url(track_code: str, race_date: date, race_number: int) -> str:
    """
    Direct URL for a single race chart PDF.
    Equibase free charts use this pattern.
    """
    params = {
        "TRACK_CODE":   track_code,
        "RACE_DATE":    race_date.strftime("%m/%d/%Y"),
        "RACE_NUMBER":  race_number,
        "RACE_TYPE":    "X",
        "CHART_OPTION": "FINISH",
    }
    return f"https://www.equibase.com/premium/chartPlus.cfm?{urlencode(params)}"


def build_entries_url(track_code: str, race_date: date) -> str:
    """Equibase entries page for upcoming races (pre-race data)."""
    params = {
        "TRACK":     track_code,
        "RACEDATE":  race_date.strftime("%m/%d/%Y"),
    }
    return f"https://www.equibase.com/static/entry/{track_code}{race_date.strftime('%m%d%Y')}USA.pdf"


# ---------------------------------------------------------------------------
# Discovery: find how many races ran on a given track/date
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=15))
def discover_races(track_code: str, race_date: date) -> list[int]:
    """
    Scrape the Equibase results index page to discover race numbers that have
    charts available for the given track/date.

    Returns a sorted list of race numbers, e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9].
    """
    url = build_chart_index_url(track_code, race_date)
    log.debug("Discovering races: %s", url)

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response.status_code == 404:
            log.info("No races found for %s on %s", track_code, race_date)
            return []
        raise

    soup = BeautifulSoup(resp.text, "lxml")
    race_nums: set[int] = set()

    # Method 1: look for PDF links with race numbers
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "chartPlus.cfm" in href or "eqbPDFChart" in href:
            m = re.search(r"RACE_NUMBER[=_](\d+)", href, re.IGNORECASE)
            if m:
                race_nums.add(int(m.group(1)))

    # Method 2: look for text patterns like "Race 1", "Race 2"
    if not race_nums:
        for tag in soup.find_all(string=re.compile(r"Race\s+\d+", re.IGNORECASE)):
            m = re.search(r"Race\s+(\d+)", tag, re.IGNORECASE)
            if m:
                race_nums.add(int(m.group(1)))

    # Method 3: fallback – assume up to 12 races and return range
    if not race_nums:
        log.warning(
            "Could not discover race count for %s %s – defaulting to 12",
            track_code, race_date
        )
        return list(range(1, 13))

    return sorted(race_nums)


# ---------------------------------------------------------------------------
# Download individual PDF chart
# ---------------------------------------------------------------------------

def pdf_cache_path(track_code: str, race_date: date, race_number: int) -> Path:
    """Local path where we cache a downloaded PDF."""
    year_dir = RAW_DIR / str(race_date.year) / track_code
    year_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{track_code}_{race_date.strftime('%Y%m%d')}_R{race_number:02d}.pdf"
    return year_dir / fname


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=3, min=3, max=30))
def download_pdf(track_code: str, race_date: date, race_number: int,
                 force_refresh: bool = False) -> Optional[Path]:
    """
    Download a single Equibase chart PDF to the local cache.
    Returns the local file path on success, None if the chart doesn't exist.
    """
    cache = pdf_cache_path(track_code, race_date, race_number)
    if cache.exists() and not force_refresh:
        log.debug("Cache hit: %s", cache)
        return cache

    url = build_chart_pdf_url(track_code, race_date, race_number)
    log.debug("Downloading: %s", url)

    try:
        resp = session.get(url, timeout=60, stream=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response.status_code in (403, 404):
            return None
        raise

    # Verify we got a PDF
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not resp.content[:4] == b"%PDF":
        log.warning("Response for %s R%d is not a PDF (content-type: %s)",
                    track_code, race_number, content_type)
        return None

    with open(cache, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    log.info("Saved: %s", cache)
    return cache


# ---------------------------------------------------------------------------
# PDF text extraction helpers
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber (preserves layout)."""
    pages_text: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2)
            if txt:
                pages_text.append(txt)
    return "\n".join(pages_text)


def extract_pdf_tables(pdf_path: Path) -> list[list[list]]:
    """
    Extract structured tables from the PDF.
    Returns a list of tables; each table is a list of rows; each row is a list of cells.
    """
    all_tables: list[list[list]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 3,
                }
            )
            if tables:
                all_tables.extend(tables)
    return all_tables


# ---------------------------------------------------------------------------
# Regex patterns for Equibase chart text
# ---------------------------------------------------------------------------

# --- Race header -----------------------------------------------------------------
RE_RACE_HEADER  = re.compile(
    r"RACE\s+(?P<race_num>\d+)",
    re.IGNORECASE
)
RE_RACE_NAME    = re.compile(
    r"(?P<race_name>(?:THE\s+)?[A-Z][A-Z\s'\-]+(?:STAKES|DERBY|CUP|CLASSIC|HANDICAP|FUTURITY|OAKS|INVITATIONAL|MEMORIAL|SPRINT|TURF)(?:\s+PRESENTED.*?)?)\s*(?:\(Grade\s+(?P<grade>[I]+)\))?",
    re.IGNORECASE
)
RE_GRADE        = re.compile(r"Grade\s+(?P<grade>I{1,3}|[123])", re.IGNORECASE)
RE_DISTANCE     = re.compile(
    r"(?P<distance>[\d\s/]+)\s+"
    r"(?:Furlongs?|Miles?|Yards?)"
    r"(?:\s+on\s+the\s+(?P<course>[A-Za-z\s]+))?",
    re.IGNORECASE
)
RE_PURSE        = re.compile(r"Purse:\s*\$(?P<purse>[\d,]+)", re.IGNORECASE)
RE_CLAIMING     = re.compile(r"Claiming\s+Price[:\s]*\$(?P<price>[\d,]+)", re.IGNORECASE)
RE_SURFACE      = re.compile(r"\b(Dirt|Turf|Synthetic|Tapeta|Polytrack|All\s+Weather)\b", re.IGNORECASE)
RE_CONDITIONS   = re.compile(r"(Maiden|Claiming|Allowance|Stakes|Handicap|Optional\s+Claiming)", re.IGNORECASE)
RE_RESTRICTIONS = re.compile(r"(For\s+(?:[\w\s,]+)(?:Fillies?|Colts?|Geldings?|Mares?|Horses?)[^\n]*)", re.IGNORECASE)
RE_WEATHER      = re.compile(r"Weather:\s*(?P<weather>[^\n]+)", re.IGNORECASE)
RE_TRACK_COND   = re.compile(r"Track:\s*(?P<condition>[A-Za-z\s]+?)(?:\s+\d|\.|$)", re.IGNORECASE)
RE_TRACK_VAR    = re.compile(r"Track\s+Variant[:\s]*(?P<variant>[\+\-]?\d+)", re.IGNORECASE)
RE_FIELD_SIZE   = re.compile(r"(\d+)\s+(?:Starters?|Runners?)", re.IGNORECASE)

# --- Times -----------------------------------------------------------------------
RE_TIME_FORMAT  = re.compile(r"(?P<mins>\d+)?:?(?P<secs>\d+)\.(?P<frac>\d+)")
RE_FRACTIONS    = re.compile(
    r"(?:Fractions?|Splits?)[\s:]+(?P<fracs>[\d:\.\s,/]+)",
    re.IGNORECASE
)
RE_FINAL_TIME   = re.compile(
    r"(?:Final\s+Time|Time)[:\s]+(?P<time>\d+:\d+\.\d+|\d+\.\d+)",
    re.IGNORECASE
)

# --- Per-horse result line -------------------------------------------------------
# Equibase chart layout (simplified):
# PP  Horse Name            Jockey   Wgt  ME  Start  1/4   1/2   Str   Fin  Beyer
RE_RESULT_LINE  = re.compile(
    r"^\s*(?P<pp>\d+)\s+"                           # program number
    r"(?P<horse>[A-Z][A-Za-z\s'\(\)]+?)\s+"         # horse name
    r"(?P<jockey>[A-Z][a-zA-Z\s,\.]+?)\s+"          # jockey
    r"(?P<weight>\d{3})\s*"                          # weight
    r"(?P<med>[LBfb ]{0,4})\s*"                     # medication/equipment
    r"(?P<rest>.*?)$",                               # rest of line
    re.MULTILINE
)
RE_POSITION     = re.compile(r"(\d+)(?:\s*hd|\s*nk|\s*ns)?\s*([\d½¾¼]+)?")
RE_ODDS         = re.compile(r"(?P<odds>\d+(?:\.\d+)?)-(?:\d+)|(?P<ml>\d+(?:/\d+)?)")
RE_BEYER        = re.compile(r"\b(?P<beyer>\d{2,3})\s*$")
RE_FINISH_POS   = re.compile(r"^(?P<pos>\d+|DQ|DNF|NS|SCR|PU|RO|UR|BD|CO|SU|REF)$", re.IGNORECASE)
RE_LASIX        = re.compile(r"\b(?:L|Lasix|Furosemide)\b", re.IGNORECASE)
RE_FIRST_LASIX  = re.compile(r"\bFTL\b|\bFirst\s+Time\s+Lasix\b", re.IGNORECASE)

# --- Breeding / ownership --------------------------------------------------------
RE_BREEDING     = re.compile(
    r"(?P<horse>[A-Z][A-Za-z\s']+?)\s*[,;]\s*"
    r"(?P<color>[a-z\s]+?),\s*"
    r"(?P<age>\d+)\s+(?P<sex>c|f|g|m|h|r)\s*,?\s*"
    r"(?:by\s+(?P<sire>[A-Z][A-Za-z\s']+?)(?:\s+out\s+of\s+(?P<dam>[A-Z][A-Za-z\s']+?)(?:\s*,\s*by\s+(?P<dam_sire>[A-Z][A-Za-z\s']+))?)?)?",
    re.IGNORECASE
)
RE_TRAINER      = re.compile(r"Trainer[:\s]+(?P<trainer>[A-Z][A-Za-z\s,\.]+?)(?:\n|Owner)", re.IGNORECASE)
RE_OWNER        = re.compile(r"Owner[:\s]+(?P<owner>[A-Z][A-Za-z\s,\.&]+?)(?:\n|Breeder)", re.IGNORECASE)
RE_BREEDER      = re.compile(r"Breeder[:\s]+(?P<breeder>[A-Z][A-Za-z\s,\.&\(\)]+?)(?:\n|$)", re.IGNORECASE)

# --- Payoffs ---------------------------------------------------------------------
RE_WIN_PAYOFF   = re.compile(r"(?P<num>\d+)\s+Win\s+\$(?P<payout>[\d,\.]+)", re.IGNORECASE)
RE_PLACE_PAYOFF = re.compile(r"(?P<num>\d+)\s+Place\s+\$(?P<payout>[\d,\.]+)", re.IGNORECASE)
RE_SHOW_PAYOFF  = re.compile(r"(?P<num>\d+)\s+Show\s+\$(?P<payout>[\d,\.]+)", re.IGNORECASE)
RE_EXOTIC_LINE  = re.compile(
    r"(?P<bet_type>Daily\s+Double|Exacta|Trifecta|Superfecta|Pick\s+\d|"
    r"Pick\s+Three|Pick\s+Four|Pick\s+Five|Pick\s+Six|"
    r"Super\s+Hi\-Five|Hi\-Five|Omni|Place\s+Pick\s+All)\s+"
    r"(?P<combo>[\d\-]+)\s+\$(?P<payout>[\d,\.]+)",
    re.IGNORECASE
)
RE_POOL_TOTAL   = re.compile(r"Pool:\s*\$(?P<pool>[\d,]+)", re.IGNORECASE)
RE_FOOTNOTE     = re.compile(r"(?P<horse>[A-Z][A-Za-z\s']+?)--(?P<note>[^;]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def time_to_seconds(time_str: str) -> Optional[float]:
    """Convert '1:12.40' or '45.20' to total seconds as a float."""
    if not time_str:
        return None
    time_str = time_str.strip()
    m = RE_TIME_FORMAT.match(time_str)
    if not m:
        return None
    mins  = int(m.group("mins")) if m.group("mins") else 0
    secs  = int(m.group("secs"))
    frac  = float(f"0.{m.group('frac')}")
    return mins * 60 + secs + frac


def frac_odds_to_decimal(frac_str: str) -> Optional[float]:
    """
    Convert fractional ML odds string to decimal.
    '5-2' or '5/2' → 3.5    '6-1' → 7.0    'Even' → 2.0
    """
    frac_str = frac_str.strip().lower()
    if frac_str in ("even", "evens", "1-1", "1/1"):
        return 2.0
    m = re.match(r"(\d+)[\-/](\d+)", frac_str)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return round((num / den) + 1, 3)
    return None


def parse_money(s: str) -> Optional[float]:
    """'$1,234,567' or '1234567' → 1234567.0"""
    cleaned = re.sub(r"[\$,]", "", s or "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade_roman_to_int(grade_str: str) -> Optional[int]:
    """'I' → 1, 'II' → 2, 'III' → 3, '1' → 1, etc."""
    mapping = {"I": 1, "II": 2, "III": 3, "1": 1, "2": 2, "3": 3}
    return mapping.get(grade_str.upper().strip())


def extract_distance_furlongs(text: str) -> Optional[float]:
    """
    Parse distance strings into furlongs.
    Handles: '6 Furlongs', '1 1/16 Miles', '1 Mile 70 Yards', '5 1/2 Furlongs'
    """
    text = text.strip()
    # Miles with fraction
    m = re.search(r"(\d+)\s+(\d+)/(\d+)\s+Mile", text, re.IGNORECASE)
    if m:
        whole = int(m.group(1)); num = int(m.group(2)); den = int(m.group(3))
        return round((whole + num / den) * 8, 4)
    # Plain miles
    m = re.search(r"(\d+)\s+Mile", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 8.0
    # Furlongs with fraction
    m = re.search(r"(\d+)\s+(\d+)/(\d+)\s+Furlong", text, re.IGNORECASE)
    if m:
        whole = int(m.group(1)); num = int(m.group(2)); den = int(m.group(3))
        return whole + num / den
    # Plain furlongs
    m = re.search(r"(\d+(?:\.\d+)?)\s+Furlong", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Yards → furlongs (1 furlong = 220 yards)
    m = re.search(r"(\d+)\s+Yard", text, re.IGNORECASE)
    if m:
        return round(int(m.group(1)) / 220, 4)
    return None


# ---------------------------------------------------------------------------
# Core parser – takes raw text from one PDF page/chart, returns structured data
# ---------------------------------------------------------------------------

def parse_chart_text(
    text: str,
    track_code: str,
    race_date: date,
    race_number: int,
) -> tuple[list[RaceResult], list[RacePayoff], list[PaceSplit]]:
    """
    Parse the full text of one Equibase chart into structured records.

    Returns
    -------
    results : list[RaceResult]  – one per horse
    payoffs : list[RacePayoff]  – one per bet type
    splits  : list[PaceSplit]   – one per horse per time call
    """
    results: list[RaceResult] = []
    payoffs: list[RacePayoff] = []
    splits:  list[PaceSplit]  = []

    lines = text.splitlines()

    # ------------------------------------------------------------------ #
    # 1.  Race header fields (shared across all horses in this race)      #
    # ------------------------------------------------------------------ #
    race_date_str = race_date.isoformat()
    track_name    = TRACK_CODES.get(track_code, {}).get("name", track_code)

    header = {
        "track_code":   track_code,
        "track_name":   track_name,
        "race_date":    race_date_str,
        "race_number":  race_number,
    }

    # Race name + grade
    for line in lines[:30]:
        m = RE_RACE_NAME.search(line)
        if m:
            header["race_name"] = m.group("race_name").strip().title()
            if m.group("grade"):
                header["grade"] = grade_roman_to_int(m.group("grade"))
        m2 = RE_GRADE.search(line)
        if m2 and "grade" not in header:
            header["grade"] = grade_roman_to_int(m2.group("grade"))

    # Distance + surface + course
    dist_text = "\n".join(lines[:40])
    header["distance_furlongs"] = extract_distance_furlongs(dist_text)

    surf_m = RE_SURFACE.search(dist_text)
    header["surface"] = surf_m.group(1).title() if surf_m else ""

    dist_m = RE_DISTANCE.search(dist_text)
    if dist_m and dist_m.group("course"):
        header["course"] = dist_m.group("course").strip().title()

    # Purse
    purse_m = RE_PURSE.search(text)
    header["purse"] = parse_money(purse_m.group("purse")) if purse_m else None

    # Claiming price
    claim_m = RE_CLAIMING.search(text)
    header["claiming_price"] = parse_money(claim_m.group("price")) if claim_m else None

    # Race type (conditions)
    cond_m = RE_CONDITIONS.search(text[:500])
    header["race_type"] = cond_m.group(1).strip().title() if cond_m else ""
    header["conditions"] = " ".join(lines[1:4])  # first few lines after race#

    # Restrictions
    restr_m = RE_RESTRICTIONS.search(text[:500])
    header["restrictions"] = restr_m.group(1).strip() if restr_m else ""

    # Weather / track condition
    wx_m = RE_WEATHER.search(text)
    header["weather"] = wx_m.group("weather").strip() if wx_m else ""

    tc_m = RE_TRACK_COND.search(text)
    header["track_condition"] = tc_m.group("condition").strip() if tc_m else ""

    var_m = RE_TRACK_VAR.search(text)
    header["track_variant"] = int(var_m.group("variant")) if var_m else None

    fs_m = RE_FIELD_SIZE.search(text)
    header["field_size"] = int(fs_m.group(1)) if fs_m else 0

    # ------------------------------------------------------------------ #
    # 2.  Fractional / final times (race-level)                           #
    # ------------------------------------------------------------------ #
    frac_m = RE_FRACTIONS.search(text)
    if frac_m:
        frac_strs = re.findall(r"\d+:\d+\.\d+|\d+\.\d+", frac_m.group("fracs"))
        frac_secs = [time_to_seconds(f) for f in frac_strs]
        if len(frac_secs) >= 1: header["fraction_1"] = frac_secs[0]
        if len(frac_secs) >= 2: header["fraction_2"] = frac_secs[1]
        if len(frac_secs) >= 3: header["fraction_3"] = frac_secs[2]

    ft_m = RE_FINAL_TIME.search(text)
    header["final_time"] = time_to_seconds(ft_m.group("time")) if ft_m else None

    # ------------------------------------------------------------------ #
    # 3.  Per-horse result lines                                          #
    # ------------------------------------------------------------------ #
    # Equibase charts have a tabular result section.
    # We try a structured table parse first, then fall back to regex.
    horse_blocks = _split_horse_blocks(text)

    for block in horse_blocks:
        result = RaceResult(**{k: v for k, v in header.items() if hasattr(RaceResult, k)})
        _parse_horse_block(block, result)

        # Beyer
        beyer_m = RE_BEYER.search(block)
        if beyer_m:
            result.beyer_speed_fig = int(beyer_m.group("beyer"))

        # Lasix flag
        result.is_lasix = bool(RE_LASIX.search(block))
        result.is_first_time_lasix = bool(RE_FIRST_LASIX.search(block))

        if result.horse_name:
            results.append(result)

    # field_size fallback
    if not header["field_size"] and results:
        header["field_size"] = len(results)
        for r in results:
            r.field_size = len(results)

    # ------------------------------------------------------------------ #
    # 4.  Payoffs section                                                 #
    # ------------------------------------------------------------------ #
    payoffs.extend(_parse_payoffs(text, track_code, race_date_str, race_number))

    # ------------------------------------------------------------------ #
    # 5.  Pace splits (per-horse running lines)                           #
    # ------------------------------------------------------------------ #
    splits.extend(_parse_pace_splits(text, track_code, race_date_str, race_number, results))

    return results, payoffs, splits


def _split_horse_blocks(text: str) -> list[str]:
    """
    Attempt to split the chart text into per-horse blocks.
    Equibase charts typically have a result table followed by
    breeding/ownership paragraph per horse.
    """
    blocks: list[str] = []
    # Each horse block often starts with an integer (program number) on a line
    # followed by the horse name. Split on numbered lines.
    pattern = re.compile(r"(?:^|\n)(\s*\d{1,2}\s+[A-Z][A-Za-z\s'\-\(\)]{3,40})", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end])
    return blocks if blocks else [text]


def _parse_horse_block(block: str, result: RaceResult) -> None:
    """Fill in per-horse fields from a text block."""
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if not lines:
        return

    # First line: PP  HorseName  ...
    first = lines[0]
    pp_m = re.match(r"(\d{1,2})\s+(.+)", first)
    if pp_m:
        result.program_number = pp_m.group(1)
        rest = pp_m.group(2)
        # Extract weight (3 digits) from the line
        wt_m = re.search(r"\b(\d{3})\b", rest)
        if wt_m:
            result.weight = float(wt_m.group(1))
        # Horse name is everything before the weight or jockey
        result.horse_name = re.split(r"\s{2,}", rest)[0].strip().title()

    # Post position often matches program number unless there's a scratch
    result.post_position = int(result.program_number) if result.program_number.isdigit() else None

    # Running positions: numbers like "3  4  5  1" across the line
    pos_nums = re.findall(r"\b(\d{1,2})(?:hd|nk|ns)?\b", block)
    if len(pos_nums) >= 4:
        try:
            result.pos_first_call   = int(pos_nums[0])
            result.pos_second_call  = int(pos_nums[1])
            result.pos_stretch      = int(pos_nums[2])
            result.pos_finish       = int(pos_nums[3])
        except (ValueError, IndexError):
            pass

    # Finish position (first standalone integer on line often = official finish)
    fin_m = re.search(r"^\s*(\d{1,2})\s", first)
    if fin_m:
        result.finish_position = int(fin_m.group(1))
        result.finish_position_raw = fin_m.group(1)

    # DQ / DNF / SCR
    for token in ("DQ", "DNF", "SCR", "NS", "PU", "RO", "UR", "BD"):
        if re.search(rf"\b{token}\b", block, re.IGNORECASE):
            result.finish_position_raw = token
            result.finish_position = None
            break

    # Medication / equipment flags (e.g. "L", "Lb", "b")
    med_m = re.search(r"\b([LBfbso]{1,4})\b\s+\d{3}", block)
    if med_m:
        flags = med_m.group(1)
        result.medication = flags
        result.equipment  = flags
        result.is_lasix   = "L" in flags.upper()

    # Morning-line odds (format "5-2" or "5/2" early in block)
    ml_m = re.search(r"\b(\d{1,3}[-/]\d{1,2})\b", block)
    if ml_m:
        result.morning_line_frac = ml_m.group(1)
        result.morning_line = frac_odds_to_decimal(ml_m.group(1))

    # Breeding line (usually towards end of block)
    breed_m = RE_BREEDING.search(block)
    if breed_m:
        result.color    = breed_m.group("color").strip() if breed_m.group("color") else ""
        try:
            result.age  = int(breed_m.group("age")) if breed_m.group("age") else None
        except ValueError:
            pass
        result.sex      = breed_m.group("sex").lower() if breed_m.group("sex") else ""
        result.sire     = breed_m.group("sire").strip().title() if breed_m.group("sire") else ""
        result.dam      = breed_m.group("dam").strip().title() if breed_m.group("dam") else ""
        result.dam_sire = breed_m.group("dam_sire").strip().title() if breed_m.group("dam_sire") else ""

    # Trainer
    tr_m = RE_TRAINER.search(block)
    if tr_m:
        result.trainer = tr_m.group("trainer").strip().title()

    # Owner
    ow_m = RE_OWNER.search(block)
    if ow_m:
        result.owner = ow_m.group("owner").strip().title()

    # Breeder
    br_m = RE_BREEDER.search(block)
    if br_m:
        result.breeder = br_m.group("breeder").strip().title()

    # Claimed flag
    if re.search(r"\bclaimed\b", block, re.IGNORECASE):
        result.claimed = True
        cl_m = re.search(r"claimed\s+by\s+(.+?)(?:\n|$)", block, re.IGNORECASE)
        if cl_m:
            result.claimed_by = cl_m.group(1).strip().title()

    # Trip footnote for this horse
    fn_m = RE_FOOTNOTE.search(block)
    if fn_m and fn_m.group("horse").lower() in result.horse_name.lower():
        result.footnote = fn_m.group("note").strip()


def _parse_payoffs(
    text: str,
    track_code: str,
    race_date_str: str,
    race_number: int,
) -> list[RacePayoff]:
    payoffs: list[RacePayoff] = []

    # Win / Place / Show
    for m in RE_WIN_PAYOFF.finditer(text):
        payoffs.append(RacePayoff(
            track_code=track_code, race_date=race_date_str,
            race_number=race_number, bet_type="WIN",
            combination=m.group("num"),
            payout=parse_money(m.group("payout")) or 0.0,
        ))
    for m in RE_PLACE_PAYOFF.finditer(text):
        payoffs.append(RacePayoff(
            track_code=track_code, race_date=race_date_str,
            race_number=race_number, bet_type="PLACE",
            combination=m.group("num"),
            payout=parse_money(m.group("payout")) or 0.0,
        ))
    for m in RE_SHOW_PAYOFF.finditer(text):
        payoffs.append(RacePayoff(
            track_code=track_code, race_date=race_date_str,
            race_number=race_number, bet_type="SHOW",
            combination=m.group("num"),
            payout=parse_money(m.group("payout")) or 0.0,
        ))

    # Exotics
    for m in RE_EXOTIC_LINE.finditer(text):
        payoffs.append(RacePayoff(
            track_code=track_code, race_date=race_date_str,
            race_number=race_number,
            bet_type=m.group("bet_type").upper().replace(" ", "_"),
            combination=m.group("combo"),
            payout=parse_money(m.group("payout")) or 0.0,
        ))

    return payoffs


def _parse_pace_splits(
    text: str,
    track_code: str,
    race_date_str: str,
    race_number: int,
    results: list[RaceResult],
) -> list[PaceSplit]:
    splits: list[PaceSplit] = []
    call_names = ["1/4", "1/2", "3/4", "Str", "Fin"]

    # Build a lookup for horses already parsed
    horse_map = {r.horse_name.lower(): r for r in results}

    # Look for a pace table: lines with multiple time/position values
    split_section = re.search(
        r"(?:Running\s+Positions?|Splits?|Fractional\s+Times?)[^\n]*\n(.*?)(?:\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not split_section:
        return splits

    for line in split_section.group(1).splitlines():
        # Attempt to match horse name + sequence of position/length pairs
        name_m = re.match(r"([A-Z][A-Za-z\s']+?)\s{2,}(.+)", line)
        if not name_m:
            continue
        horse_name = name_m.group(1).strip().title()
        values_str = name_m.group(2)
        values = re.findall(r"(\d+(?:½|¾|¼)?)\s*([\d\.]+)?", values_str)
        for i, (pos_str, lbs_str) in enumerate(values[:5]):
            if i >= len(call_names):
                break
            try:
                pos = int(re.sub(r"[½¾¼]", "", pos_str))
            except ValueError:
                pos = None
            try:
                lbs = float(lbs_str) if lbs_str else None
            except ValueError:
                lbs = None

            splits.append(PaceSplit(
                track_code=track_code,
                race_date=race_date_str,
                race_number=race_number,
                horse_name=horse_name,
                split_name=call_names[i],
                split_position=pos,
                lengths_back=lbs,
            ))

    return splits


# ---------------------------------------------------------------------------
# Orchestration: scrape + parse a full track/date
# ---------------------------------------------------------------------------

def scrape_track_date(
    track_code: str,
    race_date: date,
    force_refresh: bool = False,
    delay_seconds: float = 3.0,
) -> tuple[list[RaceResult], list[RacePayoff], list[PaceSplit]]:
    """
    Full pipeline for one track/date:
    1. Discover race numbers
    2. Download each PDF
    3. Parse each PDF
    4. Return combined results
    """
    all_results: list[RaceResult] = []
    all_payoffs: list[RacePayoff] = []
    all_splits:  list[PaceSplit]  = []

    log.info("Scraping %s on %s", track_code, race_date)
    race_numbers = discover_races(track_code, race_date)

    if not race_numbers:
        log.info("No races found for %s on %s", track_code, race_date)
        return all_results, all_payoffs, all_splits

    for rn in tqdm(race_numbers, desc=f"{track_code} {race_date}", unit="race"):
        pdf_path = download_pdf(track_code, race_date, rn, force_refresh)
        if pdf_path is None:
            log.warning("Skipping R%d – PDF not available", rn)
            continue

        try:
            text = extract_pdf_text(pdf_path)
            if not text.strip():
                log.warning("Empty text from %s", pdf_path)
                continue
            r, p, s = parse_chart_text(text, track_code, race_date, rn)
            all_results.extend(r)
            all_payoffs.extend(p)
            all_splits.extend(s)
        except Exception as exc:
            log.exception("Failed to parse %s: %s", pdf_path, exc)

        time.sleep(delay_seconds)

    log.info(
        "Finished %s %s: %d horse results, %d payoffs, %d splits",
        track_code, race_date, len(all_results), len(all_payoffs), len(all_splits)
    )
    return all_results, all_payoffs, all_splits


def scrape_date_all_tracks(
    race_date: date,
    track_codes: Optional[list[str]] = None,
    delay_between_tracks: float = 5.0,
    **kwargs,
) -> tuple[list[RaceResult], list[RacePayoff], list[PaceSplit]]:
    """Scrape all (or a subset of) US tracks for a given date."""
    tracks = track_codes or list(TRACK_CODES.keys())
    all_r, all_p, all_s = [], [], []
    for tc in tracks:
        r, p, s = scrape_track_date(tc, race_date, **kwargs)
        all_r.extend(r); all_p.extend(p); all_s.extend(s)
        time.sleep(delay_between_tracks)
    return all_r, all_p, all_s


# ---------------------------------------------------------------------------
# Local PDF parsing (no network)
# ---------------------------------------------------------------------------

def parse_local_pdfs(pdf_dir: Path) -> tuple[list[RaceResult], list[RacePayoff], list[PaceSplit]]:
    """
    Parse all .pdf files found recursively under pdf_dir.
    Infers track_code and race_date from filename convention:
        <TRACK>_<YYYYMMDD>_R<NN>.pdf
    """
    all_r, all_p, all_s = [], [], []
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    log.info("Found %d PDFs under %s", len(pdfs), pdf_dir)

    for pdf_path in tqdm(pdfs, desc="Parsing PDFs", unit="file"):
        m = re.match(r"([A-Z]{2,4})_(\d{8})_R(\d{2})\.pdf", pdf_path.name, re.IGNORECASE)
        if not m:
            log.warning("Skipping unrecognized filename: %s", pdf_path.name)
            continue
        track_code = m.group(1).upper()
        race_date  = datetime.strptime(m.group(2), "%Y%m%d").date()
        race_num   = int(m.group(3))

        try:
            text = extract_pdf_text(pdf_path)
            r, p, s = parse_chart_text(text, track_code, race_date, race_num)
            all_r.extend(r); all_p.extend(p); all_s.extend(s)
        except Exception as exc:
            log.exception("Error parsing %s: %s", pdf_path, exc)

    return all_r, all_p, all_s


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_to_parquet(
    results: list[RaceResult],
    payoffs: list[RacePayoff],
    splits:  list[PaceSplit],
    output_dir: Path = PROCESSED_DIR,
    append: bool = True,
) -> None:
    """
    Persist the three record types to parquet files.
    If append=True, merge with existing data (dedup on natural key).
    """
    def _merge_and_save(new_rows: list, path: Path, key_cols: list[str]) -> None:
        if not new_rows:
            log.info("No data to save for %s", path.name)
            return
        new_df = pd.DataFrame([asdict(r) for r in new_rows])
        if append and path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=key_cols, keep="last", inplace=True)
        else:
            combined = new_df
        combined.to_parquet(path, index=False)
        log.info("Saved %d rows → %s", len(combined), path)

    _merge_and_save(
        results, output_dir / "us_race_results.parquet",
        key_cols=["track_code", "race_date", "race_number", "horse_name"]
    )
    _merge_and_save(
        payoffs, output_dir / "us_payoffs.parquet",
        key_cols=["track_code", "race_date", "race_number", "bet_type", "combination"]
    )
    _merge_and_save(
        splits, output_dir / "us_pace_splits.parquet",
        key_cols=["track_code", "race_date", "race_number", "horse_name", "split_name"]
    )


def save_to_json(
    results: list[RaceResult],
    payoffs: list[RacePayoff],
    splits:  list[PaceSplit],
    output_path: Path,
) -> None:
    """Save all records to a single JSON file (useful for debugging)."""
    data = {
        "results": [asdict(r) for r in results],
        "payoffs": [asdict(p) for p in payoffs],
        "splits":  [asdict(s) for s in splits],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("JSON debug dump → %s", output_path)


# ---------------------------------------------------------------------------
# Derived / computed features
# (run after building the full parquet to add model-ready columns)
# ---------------------------------------------------------------------------

def compute_model_features(results_path: Path = PROCESSED_DIR / "us_race_results.parquet") -> pd.DataFrame:
    """
    Read us_race_results.parquet and engineer the key features needed
    by the US prediction model.  Returns an enriched DataFrame.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"No results file found at {results_path}")

    df = pd.read_parquet(results_path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df.sort_values(["horse_name", "race_date", "race_number"], inplace=True)

    grp = df.groupby("horse_name")

    # --- Rolling / historical features (no lookahead) ---
    df["career_runs"]        = grp.cumcount()
    df["career_wins"]        = grp["finish_position"].apply(
        lambda x: (x == 1).shift(1).fillna(False).cumsum()
    ).reset_index(level=0, drop=True)
    df["career_win_rate"]    = df["career_wins"] / df["career_runs"].clip(lower=1)

    # Top-3 place rate
    df["is_top3"]            = df["finish_position"].le(3)
    df["career_top3"]        = grp["is_top3"].apply(
        lambda x: x.shift(1).fillna(False).cumsum()
    ).reset_index(level=0, drop=True)
    df["career_place_rate"]  = df["career_top3"] / df["career_runs"].clip(lower=1)

    # Last 3 average finishing position
    df["avg_last_3_pos"]     = (
        grp["finish_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Beyer figure trend
    df["avg_beyer_last_3"]   = (
        grp["beyer_speed_fig"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["beyer_change"]       = df["beyer_speed_fig"] - grp["beyer_speed_fig"].shift(1)

    # Days since last run
    df["prev_race_date"]     = grp["race_date"].shift(1)
    df["days_since_last_run"] = (df["race_date"] - df["prev_race_date"]).dt.days

    # Class step (claiming price change)
    df["prev_claiming"]      = grp["claiming_price"].shift(1)
    df["claiming_drop_pct"]  = (
        (df["prev_claiming"] - df["claiming_price"]) / df["prev_claiming"].clip(lower=1)
    ).fillna(0)

    # Surface switch flag
    df["prev_surface"]       = grp["surface"].shift(1)
    df["surface_switch"]     = (df["surface"] != df["prev_surface"]) & df["prev_surface"].notna()

    # Age / veteran flags
    df["is_veteran"]         = df["age"].ge(8)
    df["is_3yo"]             = df["age"].eq(3)
    df["age_vs_avg"]         = df["age"] - df.groupby(["race_date", "track_code", "race_number"])["age"].transform("mean")

    # Pace style (early vs. late)
    # Positive → front-runner; negative → closer
    df["pace_style_score"]   = (
        (df["pos_second_call"].fillna(6) - df["pos_finish"].fillna(6))
    )  # positive = came from behind

    # Grade encoding
    df["grade_numeric"]      = df["grade"].fillna(4).astype(int)

    # Purse log (avoids skew)
    df["purse_log"]          = df["purse"].clip(lower=1).apply(lambda x: pd.np.log(x) if pd.notna(x) else None)

    # Post position relative to field size
    df["post_pct"]           = df["post_position"] / df["field_size"].clip(lower=1)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Equibase chart PDF scraper & parser for US horse racing data"
    )
    p.add_argument("--date",      type=str, default="today",
                   help="Race date YYYY-MM-DD or 'today' (default: today)")
    p.add_argument("--start",     type=str, default=None,
                   help="Start date for historical backfill (YYYY-MM-DD)")
    p.add_argument("--end",       type=str, default=None,
                   help="End date for historical backfill (YYYY-MM-DD)")
    p.add_argument("--track",     type=str, default=None,
                   help="Track code(s), comma-separated, e.g. 'CD,SAR' (default: all)")
    p.add_argument("--parse-only", action="store_true",
                   help="Parse already-downloaded PDFs, skip network")
    p.add_argument("--pdf-dir",   type=str, default=str(RAW_DIR),
                   help="Directory of PDFs when using --parse-only")
    p.add_argument("--force",     action="store_true",
                   help="Re-download even if cached PDF exists")
    p.add_argument("--delay",     type=float, default=3.0,
                   help="Seconds to wait between PDF downloads (default: 3)")
    p.add_argument("--no-append", action="store_true",
                   help="Overwrite parquet files instead of merging")
    p.add_argument("--json-out",  type=str, default=None,
                   help="Also dump results to JSON file (for debugging)")
    p.add_argument("--features",  action="store_true",
                   help="Re-compute model features after scraping and save enriched parquet")
    return p.parse_args()


def resolve_date(date_str: str) -> date:
    if date_str.lower() == "today":
        return date.today()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> list[date]:
    delta = (end - start).days
    return [start + timedelta(days=i) for i in range(delta + 1)]


def main() -> None:
    args = parse_args()

    # Track filter
    tracks: Optional[list[str]] = (
        [t.strip().upper() for t in args.track.split(",")]
        if args.track else None
    )

    all_r, all_p, all_s = [], [], []

    if args.parse_only:
        # ── Offline mode ──────────────────────────────────────────────────
        all_r, all_p, all_s = parse_local_pdfs(Path(args.pdf_dir))

    elif args.start and args.end:
        # ── Historical backfill ───────────────────────────────────────────
        start_d = resolve_date(args.start)
        end_d   = resolve_date(args.end)
        dates   = date_range(start_d, end_d)
        log.info("Backfilling %d days from %s to %s", len(dates), start_d, end_d)
        for d in tqdm(dates, desc="Dates", unit="day"):
            r, p, s = scrape_date_all_tracks(
                d, tracks,
                delay_between_tracks=args.delay * 2,
                delay_seconds=args.delay,
                force_refresh=args.force,
            )
            all_r.extend(r); all_p.extend(p); all_s.extend(s)
            # Save incrementally to avoid losing data on interruption
            save_to_parquet(r, p, s, append=not args.no_append)
            time.sleep(args.delay * 3)   # extra pause between days

    else:
        # ── Single date ───────────────────────────────────────────────────
        target_date = resolve_date(args.date)
        if tracks:
            for tc in tracks:
                r, p, s = scrape_track_date(
                    tc, target_date,
                    force_refresh=args.force,
                    delay_seconds=args.delay,
                )
                all_r.extend(r); all_p.extend(p); all_s.extend(s)
        else:
            all_r, all_p, all_s = scrape_date_all_tracks(
                target_date, tracks,
                delay_between_tracks=args.delay * 2,
                delay_seconds=args.delay,
                force_refresh=args.force,
            )

    # ── Save ──────────────────────────────────────────────────────────────
    if not args.start or not args.end:   # backfill saves incrementally above
        save_to_parquet(all_r, all_p, all_s, append=not args.no_append)

    if args.json_out:
        save_to_json(all_r, all_p, all_s, Path(args.json_out))

    # ── Feature engineering pass ──────────────────────────────────────────
    if args.features:
        log.info("Computing model features…")
        enriched = compute_model_features()
        out_path  = PROCESSED_DIR / "us_race_features.parquet"
        enriched.to_parquet(out_path, index=False)
        log.info("Feature-enriched data → %s  (%d rows)", out_path, len(enriched))

    log.info(
        "Done. Totals: %d horse results | %d payoffs | %d pace splits",
        len(all_r), len(all_p), len(all_s)
    )


if __name__ == "__main__":
    main()
