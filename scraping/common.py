"""
common.py — shared utilities for US horse racing scrapers.

Provides:
  - Canonical race entry/result schema (dataclasses)
  - HTTP session with retry + exponential backoff + jitter
  - Polite request pacing
  - JSON / CSV / Parquet output writers
  - Simple logging setup
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

PARSER_VERSION = "1.0.0"


@dataclass
class RaceEntry:
    """One runner (entry) in a race."""
    track_code: str
    track_name: str
    race_date: str            # YYYY-MM-DD
    race_number: int
    race_time: str = ""       # HH:MM local or empty
    race_name: str = ""
    race_class: str = ""
    surface: str = ""         # Dirt | Turf | Synthetic | Harness
    distance: str = ""        # e.g. "6f", "1 1/16m"
    purse: str = ""           # dollar string or empty
    program_number: str = ""
    runner_name: str = ""
    jockey: str = ""          # or driver for harness
    trainer: str = ""
    ml_odds: str = ""         # morning line e.g. "5/2"
    scratched: bool = False
    breed: str = "Thoroughbred"  # Thoroughbred | Quarter Horse | Harness
    source_name: str = ""
    source_url: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parser_version: str = PARSER_VERSION
    raw_extra: dict = field(default_factory=dict)  # extra source fields

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_extra", None)  # keep canonical output clean
        return d


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

JSON_HEADERS = {
    **DEFAULT_HEADERS,
    "Accept": "application/json, text/plain, */*",
}


def build_session(
    retries: int = 3,
    backoff_factor: float = 1.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
    timeout: int = 20,
) -> requests.Session:
    """Build a requests.Session with retry + backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    session._timeout = timeout  # stored for callers
    return session


def polite_get(
    session: requests.Session,
    url: str,
    min_delay: float = 2.0,
    max_delay: float = 5.0,
    **kwargs,
) -> requests.Response:
    """GET with random pacing to be polite to servers."""
    time.sleep(random.uniform(min_delay, max_delay))
    timeout = getattr(session, "_timeout", 20)
    return session.get(url, timeout=timeout, **kwargs)


def polite_get_json(
    session: requests.Session,
    url: str,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
    **kwargs,
) -> Any:
    """GET JSON endpoint with pacing. Returns parsed dict or raises."""
    session.headers.update({"Accept": "application/json, text/plain, */*"})
    resp = polite_get(session, url, min_delay=min_delay, max_delay=max_delay, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path(__file__).parent.parent / "output"


def raw_path(source: str, track: str, race_date: str) -> Path:
    p = OUTPUT_ROOT / "raw" / source / track / f"{race_date}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def processed_csv_path(label: str, race_date: str) -> Path:
    p = OUTPUT_ROOT / "processed" / f"{label}_{race_date}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_raw(data: Any, source: str, track: str, race_date: str) -> Path:
    path = raw_path(source, track, race_date)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def write_entries_csv(entries: list[RaceEntry], label: str, race_date: str) -> Path:
    if not entries:
        return processed_csv_path(label, race_date)
    path = processed_csv_path(label, race_date)
    rows = [e.to_dict() for e in entries]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_entries_csv(entries: list[RaceEntry], master_csv: Path) -> None:
    """Append entries to a rolling master CSV (creates if missing)."""
    if not entries:
        return
    rows = [e.to_dict() for e in entries]
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not master_csv.exists()
    with open(master_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def today_str() -> str:
    return date.today().isoformat()


def format_date_for_url(race_date: str, fmt: str = "%m/%d/%Y") -> str:
    """Convert YYYY-MM-DD to a different format string."""
    return datetime.strptime(race_date, "%Y-%m-%d").strftime(fmt)
