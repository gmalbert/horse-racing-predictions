# PulseScore API — Horse Racing Odds Guide
### Bet365 & Paddy Power | UK/Ireland + US Markets

---

## Table of Contents

1. [Overview & Account Limits](#1-overview--account-limits)
2. [Authentication](#2-authentication)
3. [API Architecture](#3-api-architecture)
4. [Endpoint Reference — Bet365](#4-endpoint-reference--bet365)
5. [Endpoint Reference — Paddy Power](#5-endpoint-reference--paddy-power)
6. [Request Budget Strategy (500/month)](#6-request-budget-strategy-500month)
7. [Python Implementation](#7-python-implementation)
   - [Project Setup](#71-project-setup)
   - [Core Client Module](#72-core-client-module)
   - [Horse Racing Fetcher](#73-horse-racing-fetcher)
   - [Scheduler / Budget Manager](#74-scheduler--budget-manager)
   - [CLI Runner](#75-cli-runner)
8. [Response Structure & Parsing](#8-response-structure--parsing)
9. [US Markets — Coverage Notes](#9-us-markets--coverage-notes)
10. [Data Storage & Caching](#10-data-storage--caching)
11. [Error Handling Reference](#11-error-handling-reference)
12. [Quick-Reference Cheat Sheet](#12-quick-reference-cheat-sheet)

---

## 1. Overview & Account Limits

| Item | Detail |
|---|---|
| API provider | [PulseScore](https://pulsescore.net) |
| Documentation | https://pulsescore.net/docs |
| Base URL (Bet365) | `https://api.pulsescore.net/api/v2/bet365` |
| Base URL (Paddy Power) | `https://api.pulsescore.net/api/paddypower` |
| Auth header | `X-Secret: YOUR_API_KEY` |
| Plan | BASIC (Free) |
| Monthly limit | **500 requests total** (across all bookmakers) |
| WebSocket access | Not available on BASIC plan — REST only |
| Rate limit | 1 request/second per bookmaker |

> **Key constraint:** 500 requests/month ≈ **16–17 requests/day** if used every day.  
> Budget management is critical — see Section 6.

---

## 2. Authentication

Every request requires your API key in the `X-Secret` header. Generate it from your [dashboard](https://pulsescore.net/dashboard).

```bash
# Test your key with a minimal curl call
curl -X GET "https://api.pulsescore.net/api/v2/bet365/horse-racing/leagues" \
  -H "X-Secret: YOUR_API_KEY"
```

**Never hard-code your key in source files.** Store it as an environment variable:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PULSESCORE_API_KEY="your_key_here"
```

Or use a `.env` file (with `python-dotenv`):

```
# .env
PULSESCORE_API_KEY=your_key_here
```

---

## 3. API Architecture

PulseScore separates data into two categories:

### Live Events
Returns all currently in-progress events across a sport. One call gets everything live right now for that bookmaker + sport.

```
GET /live-events?sport=horse-racing
```

### Pre-Match (by Sport → League → Events)
Pre-match data is hierarchical. You first retrieve the league list, then drill into specific leagues for odds. This is the primary flow for horse racing, since most races are pre-match.

```
GET /horse-racing/leagues          → list of meetings/leagues
GET /horse-racing/events?league=X  → all events + odds for that meeting
GET /horse-racing/events/:fi       → single event by fixture ID
```

**For Paddy Power**, the base URL differs but the path structure mirrors Bet365:

```
https://api.pulsescore.net/api/paddypower/horse-racing/leagues
https://api.pulsescore.net/api/paddypower/horse-racing/events?league=X
```

---

## 4. Endpoint Reference — Bet365

**Base URL:** `https://api.pulsescore.net/api/v2/bet365`

| Endpoint | Method | Description | Params |
|---|---|---|---|
| `/live-events?sport=horse-racing` | GET | All live horse racing events + odds | `sport=horse-racing` |
| `/horse-racing/leagues` | GET | All available meetings (pre-match) | — |
| `/horse-racing/events` | GET | All events for a meeting + full odds | `league=<name>` |
| `/horse-racing/events/:fi` | GET | Single event by fixture ID | path param |
| `/live-events/sports` | GET | Check which sports have live events | — |

### Sample Response — `/horse-racing/leagues`

```json
[
  {
    "league": "Cheltenham",
    "sport": "Horse Racing",
    "live": 0,
    "events": [
      { "home": "14:30 Cheltenham", "fi": "12345678" },
      { "home": "15:05 Cheltenham", "fi": "12345679" }
    ]
  },
  {
    "league": "Kempton",
    "sport": "Horse Racing",
    "live": 0,
    "events": [
      { "home": "18:00 Kempton", "fi": "12345680" }
    ]
  }
]
```

### Sample Response — `/horse-racing/events?league=Cheltenham`

```json
[
  {
    "fi": "12345678",
    "sport": "Horse Racing",
    "league": "Cheltenham",
    "home": "14:30 Cheltenham",
    "live": 0,
    "mg": [
      {
        "name": "Win",
        "ma": [
          { "name": "Appreciate It", "pa": [{ "decimal": "2.50" }] },
          { "name": "Honeysuckle",   "pa": [{ "decimal": "3.00" }] },
          { "name": "Energumene",    "pa": [{ "decimal": "4.50" }] }
        ]
      },
      {
        "name": "Each Way",
        "ma": [
          { "name": "Appreciate It", "pa": [{ "decimal": "2.50" }] }
        ]
      }
    ]
  }
]
```

### Key Field Reference

| Field | Meaning |
|---|---|
| `fi` | Fixture ID — use to fetch a single event |
| `league` | Meeting name (e.g. "Cheltenham", "Ascot") |
| `home` | Race name/time (horse racing has no home/away teams) |
| `live` | `1` = in-running, `0` = pre-match |
| `mg[].name` | Market name (e.g. "Win", "Each Way", "To Be Placed") |
| `ma[].name` | Selection name (horse name) |
| `ma[].pa[0].decimal` | Decimal odds |

---

## 5. Endpoint Reference — Paddy Power

**Base URL:** `https://api.pulsescore.net/api/paddypower`

The endpoint structure mirrors Bet365. Paddy Power supports horse-racing in both the UK/Ireland and selected US markets.

| Endpoint | Method | Description |
|---|---|---|
| `/live-events?sport=horse-racing` | GET | All live horse racing |
| `/horse-racing/leagues` | GET | All pre-match meetings |
| `/horse-racing/events?league=X` | GET | Odds for a specific meeting |
| `/horse-racing/events/:fi` | GET | Single event |

> **Note:** Paddy Power's market naming may differ slightly from Bet365 (e.g. "Match Betting" vs "Win"). Parse `mg[].name` dynamically rather than assuming names.

---

## 6. Request Budget Strategy (500/month)

With 500 requests/month across both bookmakers, careful planning is essential.

### Cost Per Workflow

| Action | Requests Used |
|---|---|
| Fetch Bet365 leagues | 1 |
| Fetch Paddy Power leagues | 1 |
| Fetch odds for 1 meeting (Bet365) | 1 |
| Fetch odds for 1 meeting (Paddy Power) | 1 |
| Fetch a single live event check | 1 |
| **Typical full refresh (both books, 5 meetings)** | **~12** |

### Recommended Patterns

**Option A — On-demand pull (lowest usage)**  
Only fetch data when you run the script manually. Good for personal use.
- Estimated usage: 10–20 requests per session
- Monthly capacity: ~25–50 full sessions

**Option B — Morning pre-load (scheduled)**  
Run once per morning to fetch the day's card. Cache results locally.
- Pull both bookmakers' league lists: 2 requests
- Pull odds for each relevant meeting: N requests (1 per meeting per bookmaker)
- For 5 meetings × 2 bookmakers = 12 requests/day
- Monthly cost: ~360 requests (leaves buffer for ad-hoc checks)

**Option C — Live-only during race times (lowest per-race cost)**  
Use `/live-events` once to get all in-running races in a single call.
- 1 request gets ALL live races for that bookmaker
- Poll every few minutes during racing hours only

### Budget Allocation Recommendation (500/month)

```
Pre-match morning card pulls:   ~300 requests  (≈10/day × 30 days)
Live event spot checks:         ~100 requests
Ad-hoc / manual queries:        ~70 requests
Buffer:                         ~30 requests
─────────────────────────────────────────────
Total:                          500 requests
```

### What NOT to Do

- Do **not** poll `/leagues` repeatedly — it changes rarely. Cache for the day.
- Do **not** fetch individual events by `:fi` when `/events?league=X` gives you all of them in one call.
- Do **not** fetch both bookmakers simultaneously at the same second (respect the 1 req/sec rate limit — stagger calls by 1.1s).

---

## 7. Python Implementation

### 7.1 Project Setup

```
horse_racing/
├── .env                  # API key (never commit)
├── requirements.txt
├── client.py             # Base HTTP client with rate limiting
├── fetcher.py            # Horse racing–specific logic
├── budget.py             # Request counter / monthly budget manager
├── cache.py              # Local JSON cache
└── main.py               # CLI entry point
```

**requirements.txt**

```
requests>=2.31.0
python-dotenv>=1.0.0
```

Install:

```bash
pip install -r requirements.txt
```

---

### 7.2 Core Client Module

**`client.py`** — Handles auth, rate limiting, and HTTP errors for both bookmakers.

```python
"""
client.py
Base PulseScore API client.
Handles authentication, rate limiting (1 req/sec), and error handling.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PULSESCORE_API_KEY")
if not API_KEY:
    raise EnvironmentError("PULSESCORE_API_KEY not set. Add it to your .env file.")

BASE_URLS = {
    "bet365":     "https://api.pulsescore.net/api/v2/bet365",
    "paddypower": "https://api.pulsescore.net/api/paddypower",
}

HEADERS = {"X-Secret": API_KEY}
RATE_LIMIT_DELAY = 1.1  # seconds between requests (API limit: 1 req/sec)

_last_request_time: float = 0.0


def _throttle():
    """Enforce minimum gap between API calls."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def get(bookmaker: str, path: str, params: dict = None) -> dict | list:
    """
    Make a GET request to PulseScore.

    Args:
        bookmaker: "bet365" or "paddypower"
        path:      API path, e.g. "/horse-racing/leagues"
        params:    Optional query params dict

    Returns:
        Parsed JSON response (list or dict)

    Raises:
        ValueError:   Unknown bookmaker
        RuntimeError: HTTP error (401, 403, 429, 500, etc.)
    """
    if bookmaker not in BASE_URLS:
        raise ValueError(f"Unknown bookmaker: {bookmaker!r}. Use: {list(BASE_URLS)}")

    url = BASE_URLS[bookmaker] + path
    _throttle()

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e

    if response.status_code == 401:
        raise RuntimeError("401 Unauthorized — check your API key in .env")
    if response.status_code == 403:
        raise RuntimeError("403 Forbidden — this endpoint may require a higher plan")
    if response.status_code == 429:
        raise RuntimeError("429 Rate limit exceeded — too many requests this month or too fast")
    if response.status_code == 404:
        raise RuntimeError(f"404 Not found: {url}")
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    return response.json()
```

---

### 7.3 Horse Racing Fetcher

**`fetcher.py`** — All horse racing–specific API logic. Designed to minimise requests.

```python
"""
fetcher.py
Horse racing odds fetcher for Bet365 and Paddy Power.

Design principles:
- Fetch leagues once, then batch all meetings in as few calls as possible.
- Prefer /events?league=X (returns all races in a meeting) over /events/:fi.
- Cache results so repeated calls within a session don't burn requests.
"""

from client import get
from budget import track_request
from cache import load_cache, save_cache
from datetime import date


BOOKMAKERS = ["bet365", "paddypower"]
SPORT_PATH = "horse-racing"  # PulseScore path segment for horse racing


# ─────────────────────────────────────────────
# 1. Leagues (meetings available today)
# ─────────────────────────────────────────────

def get_leagues(bookmaker: str, use_cache: bool = True) -> list[dict]:
    """
    Fetch all horse racing meetings (leagues) for a bookmaker.
    Costs: 1 request. Cached per day by default.

    Returns list of dicts:
        [{"league": "Cheltenham", "events": [...], "live": 0}, ...]
    """
    cache_key = f"{bookmaker}_leagues_{date.today()}"

    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            return cached

    track_request(bookmaker, "/horse-racing/leagues")
    data = get(bookmaker, f"/{SPORT_PATH}/leagues")

    if use_cache:
        save_cache(cache_key, data)

    return data


def get_all_leagues_both_books(use_cache: bool = True) -> dict[str, list]:
    """
    Fetch leagues from both bookmakers.
    Costs: 2 requests total (or 0 if cached).

    Returns:
        {"bet365": [...], "paddypower": [...]}
    """
    return {bm: get_leagues(bm, use_cache=use_cache) for bm in BOOKMAKERS}


# ─────────────────────────────────────────────
# 2. Odds for a specific meeting
# ─────────────────────────────────────────────

def get_meeting_odds(bookmaker: str, meeting_name: str, use_cache: bool = True) -> list[dict]:
    """
    Fetch all races + odds for a named meeting.
    Costs: 1 request. Cached per day by default.

    Args:
        bookmaker:    "bet365" or "paddypower"
        meeting_name: Exact league/meeting name from get_leagues() (e.g. "Cheltenham")

    Returns list of race event dicts with full market groups (mg).
    """
    cache_key = f"{bookmaker}_meeting_{meeting_name}_{date.today()}"

    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            return cached

    track_request(bookmaker, f"/horse-racing/events?league={meeting_name}")
    data = get(bookmaker, f"/{SPORT_PATH}/events", params={"league": meeting_name})

    if use_cache:
        save_cache(cache_key, data)

    return data


def get_multiple_meetings(bookmaker: str, meeting_names: list[str],
                          use_cache: bool = True) -> dict[str, list]:
    """
    Fetch odds for multiple meetings from one bookmaker.
    Costs: 1 request per meeting.

    Returns:
        {"Cheltenham": [...races], "Ascot": [...races], ...}
    """
    results = {}
    for meeting in meeting_names:
        results[meeting] = get_meeting_odds(bookmaker, meeting, use_cache=use_cache)
    return results


# ─────────────────────────────────────────────
# 3. Live racing (all in-running races, 1 call)
# ─────────────────────────────────────────────

def get_live_races(bookmaker: str) -> list[dict]:
    """
    Fetch all currently live (in-running) horse races.
    Costs: 1 request. Returns empty list if no races are live.

    Note: Live data should NOT be cached.
    """
    track_request(bookmaker, "/live-events?sport=horse-racing")
    return get(bookmaker, "/live-events", params={"sport": "horse-racing"})


def get_live_races_both_books() -> dict[str, list]:
    """
    Fetch live races from both bookmakers.
    Costs: 2 requests.

    Returns:
        {"bet365": [...], "paddypower": [...]}
    """
    return {bm: get_live_races(bm) for bm in BOOKMAKERS}


# ─────────────────────────────────────────────
# 4. Comparison utilities
# ─────────────────────────────────────────────

def compare_odds(meeting_name: str, race_name: str = None) -> dict:
    """
    Fetch Win market odds for a meeting from both bookmakers and compare.
    Costs: 2 requests (or 0 if cached).

    Args:
        meeting_name: Must exist in both bookmakers (names may differ slightly).
        race_name:    Optional race filter (partial match on event 'home' field).

    Returns:
        {
          "meeting": "Cheltenham",
          "races": [
            {
              "race": "14:30 Cheltenham",
              "odds": {
                "Appreciate It": {"bet365": "2.50", "paddypower": "2.60"},
                "Honeysuckle":   {"bet365": "3.00", "paddypower": "2.90"},
              }
            }
          ]
        }
    """
    odds_by_book = {}
    for bm in BOOKMAKERS:
        events = get_meeting_odds(bm, meeting_name)
        odds_by_book[bm] = events

    comparison = {"meeting": meeting_name, "races": []}

    # Use Bet365 as the race-list source; match Paddy Power by race name
    for event in odds_by_book.get("bet365", []):
        race_label = event.get("home", "Unknown Race")
        if race_name and race_name.lower() not in race_label.lower():
            continue

        win_market = _extract_win_market(event)
        if not win_market:
            continue

        b365_odds = {ma["name"]: ma["pa"][0]["decimal"] for ma in win_market["ma"]}

        # Find matching race in Paddy Power
        pp_events = odds_by_book.get("paddypower", [])
        pp_event = _find_matching_race(pp_events, race_label)
        pp_odds = {}
        if pp_event:
            pp_win = _extract_win_market(pp_event)
            if pp_win:
                pp_odds = {ma["name"]: ma["pa"][0]["decimal"] for ma in pp_win["ma"]}

        # Build combined runner odds
        all_runners = set(b365_odds) | set(pp_odds)
        race_odds = {}
        for runner in sorted(all_runners):
            race_odds[runner] = {
                "bet365":     b365_odds.get(runner, "N/A"),
                "paddypower": pp_odds.get(runner, "N/A"),
            }

        comparison["races"].append({"race": race_label, "odds": race_odds})

    return comparison


def _extract_win_market(event: dict) -> dict | None:
    """Return the Win market group from an event, or None."""
    for mg in event.get("mg", []):
        if mg.get("name", "").lower() in ("win", "match betting", "to win"):
            return mg
    # Fallback: return the first market group if no exact match
    mgs = event.get("mg", [])
    return mgs[0] if mgs else None


def _find_matching_race(events: list[dict], race_label: str) -> dict | None:
    """Find a race in a list by partial name match."""
    for event in events:
        if race_label.lower() in event.get("home", "").lower():
            return event
        # Try matching by time component (e.g. "14:30")
        time_part = race_label[:5]
        if time_part and time_part in event.get("home", ""):
            return event
    return None
```

---

### 7.4 Scheduler / Budget Manager

**`budget.py`** — Tracks monthly API usage and warns when approaching the limit.

```python
"""
budget.py
Tracks monthly API request usage.
Persists a count to budget.json so it survives across sessions.
"""

import json
import os
from datetime import date

BUDGET_FILE = "budget.json"
MONTHLY_LIMIT = 500
WARN_AT = 450  # Warn when this many requests have been used


def _load() -> dict:
    if os.path.exists(BUDGET_FILE):
        with open(BUDGET_FILE) as f:
            return json.load(f)
    return {"month": str(date.today())[:7], "count": 0, "log": []}


def _save(data: dict):
    with open(BUDGET_FILE, "w") as f:
        json.dump(data, f, indent=2)


def track_request(bookmaker: str, endpoint: str):
    """
    Record one API request and check against the monthly budget.
    Call this before every API call.

    Raises RuntimeError if the monthly limit has been reached.
    """
    data = _load()
    current_month = str(date.today())[:7]

    # Reset counter at the start of a new month
    if data.get("month") != current_month:
        data = {"month": current_month, "count": 0, "log": []}

    data["count"] += 1
    data["log"].append({
        "date": str(date.today()),
        "bookmaker": bookmaker,
        "endpoint": endpoint,
        "total_so_far": data["count"],
    })

    _save(data)

    if data["count"] > MONTHLY_LIMIT:
        raise RuntimeError(
            f"Monthly API limit of {MONTHLY_LIMIT} requests reached. "
            f"Resets at the start of next month."
        )

    if data["count"] >= WARN_AT:
        remaining = MONTHLY_LIMIT - data["count"]
        print(f"⚠️  Budget warning: {data['count']}/{MONTHLY_LIMIT} requests used "
              f"({remaining} remaining this month)")


def get_status() -> dict:
    """Return current budget status."""
    data = _load()
    current_month = str(date.today())[:7]
    if data.get("month") != current_month:
        return {"month": current_month, "used": 0, "remaining": MONTHLY_LIMIT}
    used = data.get("count", 0)
    return {
        "month": current_month,
        "used": used,
        "remaining": MONTHLY_LIMIT - used,
        "limit": MONTHLY_LIMIT,
    }


def print_status():
    s = get_status()
    print(f"📊 API Budget — {s['month']}: {s['used']}/{s['limit']} used, "
          f"{s['remaining']} remaining")
```

---

**`cache.py`** — Lightweight JSON file cache to avoid re-fetching league/event data.

```python
"""
cache.py
Simple file-based JSON cache. Keys expire daily by default (keyed by date).
"""

import json
import os

CACHE_DIR = ".cache"


def _path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_key = key.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe_key}.json")


def load_cache(key: str):
    """Return cached value or None if not found."""
    p = _path(key)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_cache(key: str, data):
    """Write data to cache."""
    with open(_path(key), "w") as f:
        json.dump(data, f, indent=2)


def clear_cache():
    """Delete all cached files."""
    if os.path.isdir(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, fname))
    print("Cache cleared.")
```

---

### 7.5 CLI Runner

**`main.py`** — Command-line interface to drive all the above modules.

```python
"""
main.py
CLI for fetching horse racing odds from PulseScore.

Usage examples:
    python main.py status                          # Show API budget
    python main.py leagues                         # List today's meetings (both books)
    python main.py live                            # All live races (both books)
    python main.py meeting "Cheltenham"            # Odds for Cheltenham (both books)
    python main.py compare "Cheltenham"            # Side-by-side odds comparison
    python main.py compare "Cheltenham" "14:30"   # Filter to a specific race time
"""

import sys
import json
from budget import print_status, get_status
from fetcher import (
    get_all_leagues_both_books,
    get_live_races_both_books,
    get_meeting_odds,
    compare_odds,
    BOOKMAKERS,
)


def cmd_status():
    print_status()
    s = get_status()
    days_left = 30  # Approximate
    daily_budget = s["remaining"] // max(days_left, 1)
    print(f"💡 Suggested daily spend: ~{daily_budget} requests")


def cmd_leagues():
    print("Fetching today's horse racing meetings...\n")
    all_leagues = get_all_leagues_both_books()
    for bm, leagues in all_leagues.items():
        print(f"── {bm.upper()} ──")
        if not leagues:
            print("  No meetings found.")
        for lg in leagues:
            race_count = len(lg.get("events", []))
            live_flag = " [LIVE]" if lg.get("live") else ""
            print(f"  {lg['league']}{live_flag}  ({race_count} races)")
        print()


def cmd_live():
    print("Fetching live horse races...\n")
    live = get_live_races_both_books()
    for bm, events in live.items():
        print(f"── {bm.upper()} — {len(events)} live races ──")
        for event in events:
            print(f"  {event.get('league', '?')} | {event.get('home', '?')}")
            for mg in event.get("mg", [])[:1]:  # Show first market only
                print(f"    Market: {mg['name']}")
                for ma in mg["ma"][:5]:  # Top 5 runners
                    odds = ma["pa"][0]["decimal"] if ma.get("pa") else "N/A"
                    print(f"      {ma['name']:<30} {odds}")
        print()


def cmd_meeting(meeting_name: str):
    print(f"Fetching odds for '{meeting_name}'...\n")
    for bm in BOOKMAKERS:
        print(f"── {bm.upper()} ──")
        try:
            events = get_meeting_odds(bm, meeting_name)
            if not events:
                print("  No events found for this meeting.")
            for event in events:
                print(f"\n  Race: {event.get('home', '?')}")
                for mg in event.get("mg", []):
                    print(f"    [{mg['name']}]")
                    for ma in mg["ma"]:
                        odds = ma["pa"][0]["decimal"] if ma.get("pa") else "N/A"
                        print(f"      {ma['name']:<30} {odds}")
        except RuntimeError as e:
            print(f"  Error: {e}")
        print()


def cmd_compare(meeting_name: str, race_filter: str = None):
    print(f"Comparing odds for '{meeting_name}'"
          + (f" — race filter: '{race_filter}'" if race_filter else "")
          + "\n")
    result = compare_odds(meeting_name, race_name=race_filter)
    for race_data in result["races"]:
        print(f"  Race: {race_data['race']}")
        print(f"  {'Runner':<30} {'Bet365':>10} {'Paddy Power':>12}")
        print(f"  {'-'*52}")
        for runner, books in race_data["odds"].items():
            b365 = books.get("bet365", "N/A")
            pp   = books.get("paddypower", "N/A")
            # Highlight best price
            best = ""
            try:
                if float(b365) > float(pp):
                    best = " ◄ B365"
                elif float(pp) > float(b365):
                    best = " ◄ PP"
            except (ValueError, TypeError):
                pass
            print(f"  {runner:<30} {b365:>10} {pp:>12}{best}")
        print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args or args[0] == "status":
        cmd_status()
    elif args[0] == "leagues":
        cmd_leagues()
    elif args[0] == "live":
        cmd_live()
    elif args[0] == "meeting" and len(args) >= 2:
        cmd_meeting(args[1])
    elif args[0] == "compare" and len(args) >= 2:
        race_filter = args[2] if len(args) >= 3 else None
        cmd_compare(args[1], race_filter)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
```

---

## 8. Response Structure & Parsing

### Market Group (`mg`) Structure

All events follow the same nested structure:

```
event
└── mg[]              # list of market groups
    ├── name          # "Win", "Each Way", "To Be Placed", etc.
    └── ma[]          # list of market alternatives (runners)
        ├── name      # horse name
        └── pa[]      # list of prices
            └── decimal  # decimal odds string, e.g. "3.50"
```

### Converting Odds Formats

```python
def decimal_to_fractional(decimal: float) -> str:
    """Convert decimal odds to fractional (e.g. 3.5 → '5/2')."""
    from fractions import Fraction
    frac = Fraction(decimal - 1).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"


def decimal_to_american(decimal: float) -> str:
    """Convert decimal odds to American moneyline format."""
    if decimal >= 2.0:
        return f"+{int((decimal - 1) * 100)}"
    else:
        return f"-{int(100 / (decimal - 1))}"


# Example
dec = 3.50
print(decimal_to_fractional(dec))  # "5/2"
print(decimal_to_american(dec))    # "+250"
```

### Common Market Names to Look For

| Market Name | Description |
|---|---|
| `Win` | Outright winner odds |
| `Each Way` | Win + place terms |
| `To Be Placed` | Place-only market |
| `Match Betting` | Paddy Power equivalent of Win |
| `Forecast` | 1-2 finish prediction |
| `Antepost` | Future race (not day-of) |

> **Tip:** Always iterate over `mg` by checking `mg["name"]` rather than assuming a fixed index. Market availability varies by race and bookmaker.

---

## 9. US Markets — Coverage Notes

Both Bet365 and Paddy Power include US horse racing in their coverage. The key meetings to look for in the league list:

### US Tracks Typically Available

| Track | State | Season |
|---|---|---|
| Churchill Downs | Kentucky | Spring/Fall (Kentucky Derby in May) |
| Belmont Park / NYRA | New York | Spring–Fall |
| Santa Anita | California | Year-round |
| Keeneland | Kentucky | April & October |
| Gulfstream Park | Florida | Winter–Spring |
| Saratoga | New York | Summer |
| Del Mar | California | Summer–Fall |
| Pimlico | Maryland | Preakness Stakes in May |

### How to Find US Meetings

```python
from fetcher import get_leagues

leagues = get_leagues("bet365")

us_keywords = [
    "Churchill", "Belmont", "Santa Anita", "Keeneland",
    "Gulfstream", "Saratoga", "Del Mar", "Pimlico",
    "Aqueduct", "Fair Grounds", "Oaklawn", "Tampa Bay Downs"
]

us_meetings = [
    lg for lg in leagues
    if any(kw.lower() in lg["league"].lower() for kw in us_keywords)
]

for m in us_meetings:
    print(m["league"])
```

### Coverage Caveat

US coverage depends on the bookmaker's active markets at the time of your request. Bet365 and Paddy Power are primarily UK/Ireland focused, so:

- **Major US graded stakes races** (Kentucky Derby, Preakness, Belmont Stakes, Breeders' Cup) are reliably available.
- **Day-of-week US card** (everyday racing) may be limited or absent.
- **Antepost US futures** are often available weeks before major events.

If consistent US coverage is a priority, consider supplementing with a US-native bookmaker available on PulseScore (FanDuel, DraftKings) — though those don't carry horse racing in the current valid sports list and would require a separate request budget.

---

## 10. Data Storage & Caching

The built-in cache in `cache.py` stores results as JSON files keyed by `{bookmaker}_{endpoint}_{date}`, so they expire naturally each day.

For longer-term storage, extend with a simple SQLite approach:

```python
"""
store.py — optional SQLite persistence for odds history
"""
import sqlite3
import json
from datetime import datetime

DB = "odds_history.db"


def init_db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at  TEXT,
            bookmaker   TEXT,
            meeting     TEXT,
            race        TEXT,
            runner      TEXT,
            market      TEXT,
            decimal_odds TEXT
        )
    """)
    con.commit()
    con.close()


def save_odds(bookmaker: str, meeting: str, events: list[dict]):
    """Persist parsed odds to SQLite."""
    con = sqlite3.connect(DB)
    now = datetime.utcnow().isoformat()
    rows = []
    for event in events:
        race = event.get("home", "Unknown")
        for mg in event.get("mg", []):
            market = mg.get("name", "")
            for ma in mg.get("ma", []):
                runner = ma.get("name", "")
                odds = ma.get("pa", [{}])[0].get("decimal", "")
                rows.append((now, bookmaker, meeting, race, runner, market, odds))
    con.executemany(
        "INSERT INTO odds (fetched_at, bookmaker, meeting, race, runner, market, decimal_odds) "
        "VALUES (?,?,?,?,?,?,?)",
        rows
    )
    con.commit()
    con.close()
```

---

## 11. Error Handling Reference

### HTTP Errors

| Code | Meaning | Action |
|---|---|---|
| `200` | Success | Parse JSON response |
| `401` | Bad/missing API key | Check `.env` and dashboard |
| `403` | Plan restriction | Endpoint may need PRO plan |
| `404` | Resource not found | Check meeting name spelling |
| `429` | Rate limit hit | Monthly quota exceeded or calling too fast |
| `500` | Server error | Retry after a few seconds |

### Common Issues

**"No events returned for meeting"**  
The meeting name must exactly match the `league` field from `/leagues`. Fetch leagues first and use the exact string.

```python
# Wrong
get_meeting_odds("bet365", "cheltenham")  # lowercase may fail

# Right — use the exact name from get_leagues()
leagues = get_leagues("bet365")
meeting_name = leagues[0]["league"]  # e.g. "Cheltenham"
get_meeting_odds("bet365", meeting_name)
```

**"429 on the first call of the month"**  
You may be calling faster than 1 req/sec. The `_throttle()` function in `client.py` handles this automatically. If calling manually, add `time.sleep(1.1)` between calls.

**"KeyError: 'pa'"**  
Some selections (e.g. non-runners) may have empty or missing `pa` arrays. Always guard:

```python
odds = ma.get("pa", [{}])[0].get("decimal", "N/A")
```

---

## 12. Quick-Reference Cheat Sheet

### Endpoints at a Glance

```
# Bet365
GET https://api.pulsescore.net/api/v2/bet365/horse-racing/leagues
GET https://api.pulsescore.net/api/v2/bet365/horse-racing/events?league=MEETING
GET https://api.pulsescore.net/api/v2/bet365/horse-racing/events/:fi
GET https://api.pulsescore.net/api/v2/bet365/live-events?sport=horse-racing

# Paddy Power
GET https://api.pulsescore.net/api/paddypower/horse-racing/leagues
GET https://api.pulsescore.net/api/paddypower/horse-racing/events?league=MEETING
GET https://api.pulsescore.net/api/paddypower/horse-racing/events/:fi
GET https://api.pulsescore.net/api/paddypower/live-events?sport=horse-racing
```

### One-liner curl tests

```bash
export KEY="your_api_key"

# Today's Bet365 meetings
curl -s -H "X-Secret: $KEY" \
  "https://api.pulsescore.net/api/v2/bet365/horse-racing/leagues" | python3 -m json.tool

# Today's Paddy Power meetings
curl -s -H "X-Secret: $KEY" \
  "https://api.pulsescore.net/api/paddypower/horse-racing/leagues" | python3 -m json.tool

# Odds for a specific Bet365 meeting
curl -s -H "X-Secret: $KEY" \
  "https://api.pulsescore.net/api/v2/bet365/horse-racing/events?league=Cheltenham" \
  | python3 -m json.tool

# All live Paddy Power horse racing
curl -s -H "X-Secret: $KEY" \
  "https://api.pulsescore.net/api/paddypower/live-events?sport=horse-racing" \
  | python3 -m json.tool
```

### Python CLI commands

```bash
python main.py status                       # Budget status
python main.py leagues                      # Today's meetings (both books)
python main.py live                         # Live races now (both books)
python main.py meeting "Cheltenham"         # Full odds for a meeting
python main.py compare "Cheltenham"         # Best price comparison
python main.py compare "Cheltenham" "14:30" # Specific race comparison
```

### Request Cost Summary

| Task | Requests |
|---|---|
| Both bookmakers' league lists | 2 |
| One meeting, one bookmaker | 1 |
| One meeting, both bookmakers | 2 |
| All live races, both bookmakers | 2 |
| Full card (5 meetings, both books) | 12 |
| Full card + live check | 14 |

---

*Document generated from PulseScore API documentation at https://pulsescore.net/docs.*  
*Last reviewed: May 2026. Verify endpoint paths against live docs if behaviour changes.*
