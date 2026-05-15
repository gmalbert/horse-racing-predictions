"""
tvg_client.py — TVG/FanDuel GraphQL API client for US horse racing entries.

Source: https://api.tvg.com/cosmo/v1/graphql (public, no auth required)
Covers: All US tracks available on TVG/FanDuel Racing platform.

The TVG GraphQL API returns real-time race entries with:
  - Horse names, jockeys/drivers, trainers
  - Morning line and live odds
  - Race details: distance, surface, class, purse
  - Horse details: age, sex, sire, dam, owner
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    RaceEntry,
    get_logger,
    write_raw,
    OUTPUT_ROOT,
)

logger = get_logger("tvg_client")

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

GQL_URL = "https://api.tvg.com/cosmo/v1/graphql"
WAGER_PROFILE = "PORT-Generic"

GQL_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.tvg.com",
    "Referer": "https://www.tvg.com/",
    "Accept": "application/json",
}

SOURCE_NAME = "tvg"

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

# Get all US track codes for today
QUERY_US_TRACKS = """
query getFullScheduleTracks($wagerProfile: String, $trackSortBy: TrackListSort,
                             $raceSortBy: RaceListSort, $raceFilterBy: RaceListFilter) {
  tracks(sort: $trackSortBy, profile: $wagerProfile) {
    code
    name
    location { country }
    races(sort: $raceSortBy, filter: $raceFilterBy) {
      number
      mtp
    }
  }
}
"""

# Get full race program (all runners) for a list of track codes
QUERY_TRACK_RUNNERS = """
query getTrackRacesWithRunners($trackCode: [String], $wagerProfile: String) {
  tracks(filter: {code: $trackCode}, profile: $wagerProfile) {
    id
    code
    name
    location { country }
    numberOfRaces
    races {
      id
      number
      postTime
      numRunners
      distance
      surface { name code }
      type { name code }
      raceClass { name id }
      purse
      bettingInterests {
        biNumber
        saddleColor
        numberColor
        currentOdds { numerator denominator }
        morningLineOdds { numerator denominator }
        runners {
          horseName
          runnerId
          scratched
          trainer
          jockey
          weight
          age
          sex
          dam
          sire
          damSire
          ownerName
          timeform { silkUrl silkUrlSvg }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _build_gql_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(GQL_HEADERS)
    return session


# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------

def _post_gql(session: requests.Session, operation: str, variables: dict, query: str) -> Any:
    """POST a GraphQL query and return parsed data dict, or None on error."""
    payload = {
        "operationName": operation,
        "variables": variables,
        "query": query,
    }
    try:
        resp = session.post(GQL_URL, json=payload, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body and not body.get("data"):
            logger.warning("GQL errors for %s: %s", operation, body["errors"][:2])
            return None
        return body.get("data")
    except Exception as exc:
        logger.error("GQL request failed (%s): %s", operation, exc)
        return None


# ---------------------------------------------------------------------------
# Odds formatting helpers
# ---------------------------------------------------------------------------

def _format_odds(odds_obj: dict | None) -> str:
    """Convert {numerator, denominator} to fractional string like '5/2' or '6/1'."""
    if not odds_obj:
        return ""
    num = odds_obj.get("numerator")
    den = odds_obj.get("denominator")
    if num is None:
        return ""
    if den is None or den == 0:
        return f"{num}/1"
    return f"{num}/{den}"


def _odds_to_decimal(odds_obj: dict | None) -> float | None:
    """Convert fractional odds object to decimal (European) odds."""
    if not odds_obj:
        return None
    num = odds_obj.get("numerator")
    den = odds_obj.get("denominator") or 1
    if num is None:
        return None
    return round(1.0 + num / den, 2)


# ---------------------------------------------------------------------------
# Breed / type mapping
# ---------------------------------------------------------------------------

TYPE_CODE_TO_BREED = {
    "T": "Thoroughbred",
    "H": "Harness",
    "Q": "Quarter Horse",
    "G": "Greyhound",
    "P": "Paint",
    "A": "Arabian",
}


def _race_type_to_breed(type_obj: dict | None) -> str:
    if not type_obj:
        return "Thoroughbred"
    code = type_obj.get("code", "T")
    return TYPE_CODE_TO_BREED.get(code, "Thoroughbred")


# ---------------------------------------------------------------------------
# Post time helpers
# ---------------------------------------------------------------------------

def _posttime_to_local_hhmm(post_time_iso: str) -> str:
    """Convert ISO UTC post time to HH:MM local (Eastern Racing time approx)."""
    if not post_time_iso:
        return ""
    try:
        dt = datetime.fromisoformat(post_time_iso.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")  # Return UTC HH:MM - callers can convert if needed
    except Exception:
        return post_time_iso[:16]


def _posttime_to_race_date(post_time_iso: str) -> str:
    """Extract YYYY-MM-DD race date from ISO post time."""
    if not post_time_iso:
        return ""
    try:
        dt = datetime.fromisoformat(post_time_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return post_time_iso[:10]


# ---------------------------------------------------------------------------
# Track fetching
# ---------------------------------------------------------------------------

def get_us_track_codes(session: requests.Session) -> list[tuple[str, str]]:
    """Return list of (code, name) for all US tracks with open races today."""
    data = _post_gql(session, "getFullScheduleTracks", {
        "wagerProfile": WAGER_PROFILE,
        "trackSortBy": {"byName": "ASC"},
        "raceSortBy": {"byMTP": "ASC"},
        "raceFilterBy": {"hasMTP": True, "isOpen": True},
    }, QUERY_US_TRACKS)

    if not data:
        return []

    us_tracks = []
    for track in data.get("tracks", []):
        country = track.get("location", {}).get("country", "")
        if country == "USA":
            # Only include tracks that have races today
            races = track.get("races", [])
            if races:
                us_tracks.append((track["code"], track["name"]))

    logger.info("Found %d US tracks with open races", len(us_tracks))
    return us_tracks


def get_all_us_track_codes(session: requests.Session) -> list[tuple[str, str]]:
    """Return ALL US tracks (even those with no currently open races)."""
    data = _post_gql(session, "getFullScheduleTracks", {
        "wagerProfile": WAGER_PROFILE,
        "trackSortBy": {"byName": "ASC"},
        "raceSortBy": {"byMTP": "ASC"},
        "raceFilterBy": {"hasMTP": True, "isOpen": True},
    }, QUERY_US_TRACKS)

    if not data:
        return []

    us_tracks = []
    for track in data.get("tracks", []):
        country = track.get("location", {}).get("country", "")
        if country == "USA":
            us_tracks.append((track["code"], track["name"]))

    return us_tracks


# ---------------------------------------------------------------------------
# Entry mapping
# ---------------------------------------------------------------------------

def _map_entries(track_data: dict, race_date: str) -> list[RaceEntry]:
    """Convert TVG track GraphQL response to list of RaceEntry objects."""
    entries: list[RaceEntry] = []

    track_code = track_data.get("code", "")
    track_name = track_data.get("name", "")

    for race in track_data.get("races", []):
        post_time = race.get("postTime", "")
        race_number = int(race.get("number") or 0)
        race_time = _posttime_to_local_hhmm(post_time)

        # Surface
        surface_obj = race.get("surface") or {}
        surface_name = surface_obj.get("name", "")
        # Normalize: Dirt, Turf, Synthetic, Harness (same track surface)
        if not surface_name:
            surface_name = "Dirt"

        # Race type / breed
        type_obj = race.get("type") or {}
        breed = _race_type_to_breed(type_obj)

        # Skip greyhound races
        if type_obj.get("code") == "G":
            continue

        # Race class name
        class_obj = race.get("raceClass") or {}
        race_class = class_obj.get("name", "")

        # Distance
        distance = race.get("distance", "") or ""

        # Purse
        purse_raw = race.get("purse") or 0
        purse = f"${int(purse_raw):,}" if purse_raw else ""

        # Race name (TVG doesn't provide one, use class + distance)
        race_name = f"{race_class} {distance}".strip() if race_class else distance

        for bi in race.get("bettingInterests", []):
            bi_number = bi.get("biNumber", 0)
            ml_odds_obj = bi.get("morningLineOdds")
            ml_odds = _format_odds(ml_odds_obj)

            # Decimal odds for raw_extra
            current_odds_obj = bi.get("currentOdds")
            current_decimal = _odds_to_decimal(current_odds_obj)
            ml_decimal = _odds_to_decimal(ml_odds_obj)

            for runner in bi.get("runners", []):
                horse_name = runner.get("horseName", "")
                if not horse_name:
                    continue

                scratched = bool(runner.get("scratched", False))
                jockey = runner.get("jockey", "") or ""
                trainer = runner.get("trainer", "") or ""
                weight = runner.get("weight") or 0
                age = runner.get("age") or ""
                sex = runner.get("sex", "") or ""
                dam = runner.get("dam", "") or ""
                sire = runner.get("sire", "") or ""
                dam_sire = runner.get("damSire", "") or ""
                owner = runner.get("ownerName", "") or ""
                runner_id = runner.get("runnerId", "")

                entry = RaceEntry(
                    track_code=track_code,
                    track_name=track_name,
                    race_date=race_date,
                    race_number=race_number,
                    race_time=race_time,
                    race_name=race_name,
                    race_class=race_class,
                    surface=surface_name,
                    distance=distance,
                    purse=purse,
                    program_number=str(bi_number),
                    runner_name=horse_name,
                    jockey=jockey,
                    trainer=trainer,
                    ml_odds=ml_odds,
                    scratched=scratched,
                    breed=breed,
                    source_name=SOURCE_NAME,
                    source_url=GQL_URL,
                    raw_extra={
                        "runner_id": runner_id,
                        "bi_number": bi_number,
                        "age": age,
                        "sex": sex,
                        "sire": sire,
                        "dam": dam,
                        "dam_sire": dam_sire,
                        "owner": owner,
                        "weight": weight,
                        "current_odds": _format_odds(current_odds_obj),
                        "current_odds_decimal": current_decimal,
                        "ml_odds_decimal": ml_decimal,
                        "race_id": race.get("id", ""),
                        "post_time_utc": post_time,
                        "num_runners": race.get("numRunners", ""),
                        "saddle_color": bi.get("saddleColor", ""),
                        "number_color": bi.get("numberColor", ""),
                    },
                )
                entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_us_tracks(race_date: str, batch_size: int = 10) -> tuple[list[RaceEntry], int, int]:
    """
    Fetch all US race entries for a given date from TVG GraphQL.

    Args:
        race_date: YYYY-MM-DD string for the race date
        batch_size: Number of tracks to query per GraphQL call

    Returns:
        (entries, tracks_attempted, tracks_with_data)
    """
    session = _build_gql_session()
    all_entries: list[RaceEntry] = []

    # Step 1: Get all US track codes
    us_tracks = get_us_track_codes(session)
    if not us_tracks:
        logger.warning("No US tracks found with open races for %s", race_date)
        # Fall back to getting ALL US tracks without the open filter
        us_tracks = get_all_us_track_codes(session)

    if not us_tracks:
        logger.error("No US tracks found at all")
        return [], 0, 0

    tracks_attempted = len(us_tracks)
    tracks_with_data = 0

    # Step 2: Fetch runners in batches
    track_codes = [code for code, _ in us_tracks]

    for i in range(0, len(track_codes), batch_size):
        batch = track_codes[i:i + batch_size]
        logger.info("Fetching runners for tracks: %s", batch)

        data = _post_gql(session, "getTrackRacesWithRunners", {
            "trackCode": batch,
            "wagerProfile": WAGER_PROFILE,
        }, QUERY_TRACK_RUNNERS)

        if not data:
            logger.warning("No data returned for batch: %s", batch)
            continue

        for track_data in data.get("tracks", []):
            code = track_data.get("code", "?")
            entries = _map_entries(track_data, race_date)

            if entries:
                tracks_with_data += 1
                all_entries.extend(entries)
                logger.info("  %s: %d entries across %d races", code, len(entries),
                            len(track_data.get("races", [])))
            else:
                logger.info("  %s: no entries found", code)

            # Save raw data for each track
            try:
                raw_data = track_data
                write_raw(raw_data, SOURCE_NAME, code, race_date)
            except Exception as exc:
                logger.warning("Failed to write raw data for %s: %s", code, exc)

        # Be polite between batches
        if i + batch_size < len(track_codes):
            time.sleep(1.0)

    logger.info(
        "TVG: %d entries from %d/%d tracks",
        len(all_entries), tracks_with_data, tracks_attempted
    )
    return all_entries, tracks_attempted, tracks_with_data


def fetch_track(track_code: str, race_date: str) -> list[RaceEntry]:
    """
    Fetch entries for a single US track. Convenience wrapper for pipeline.

    Args:
        track_code: TVG track code (e.g., 'CD', 'SA', 'LRL')
        race_date: YYYY-MM-DD

    Returns:
        List of RaceEntry objects
    """
    session = _build_gql_session()
    data = _post_gql(session, "getTrackRacesWithRunners", {
        "trackCode": [track_code],
        "wagerProfile": WAGER_PROFILE,
    }, QUERY_TRACK_RUNNERS)

    if not data or not data.get("tracks"):
        logger.warning("No data for track %s", track_code)
        return []

    entries = []
    for track_data in data["tracks"]:
        entries.extend(_map_entries(track_data, race_date))

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch US race entries from TVG GraphQL")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Race date YYYY-MM-DD (default: today)")
    parser.add_argument("--track", help="Specific track code to fetch (optional)")
    parser.add_argument("--list-tracks", action="store_true",
                        help="List available US tracks and exit")
    args = parser.parse_args()

    if args.list_tracks:
        session = _build_gql_session()
        tracks = get_us_track_codes(session)
        print(f"US tracks with open races ({len(tracks)}):")
        for code, name in tracks:
            print(f"  {code}: {name}")
        sys.exit(0)

    if args.track:
        entries = fetch_track(args.track, args.date)
        print(f"\n{args.track} — {len(entries)} entries for {args.date}")
        for e in entries[:5]:
            print(f"  R{e.race_number} #{e.program_number} {e.runner_name} "
                  f"J:{e.jockey} T:{e.trainer} ML:{e.ml_odds}")
    else:
        entries, attempted, with_data = fetch_all_us_tracks(args.date)
        print(f"\nTotal: {len(entries)} entries from {with_data}/{attempted} US tracks")
        # Show sample
        if entries:
            for e in entries[:10]:
                print(f"  {e.track_code} R{e.race_number} #{e.program_number} "
                      f"{e.runner_name} J:{e.jockey} ML:{e.ml_odds}")
