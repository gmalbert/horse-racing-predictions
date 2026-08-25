#!/usr/bin/env python3
"""Collect Racing Post racecards with an ordinary visible browser.

This provider is intended for an authorized, interactively launched self-hosted
runner. It does not alter browser fingerprints, solve challenges, rotate
proxies, or use Playwright stealth plugins. The browser renders the same public
racecard pages a user can open and the script validates the canonical output
before replacing an existing snapshot.

Examples:
    python scripts/fetch_racecards_browser.py --date 2026-08-25
    python scripts/fetch_racecards_browser.py --days 2
    python scripts/fetch_racecards_browser.py --days 2 --profile-dir C:\\rp-profile
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from .fetch_racecards import build_racecard_from_html
except ImportError:
    from fetch_racecards import build_racecard_from_html


BASE_URL = "https://www.racingpost.com"
RACE_PATH_RE = re.compile(
    r"^/racecards/(?P<course_id>\d+)/[^/]+/(?P<date>\d{4}-\d{2}-\d{2})/(?P<race_id>\d+)/?$"
)


def target_dates(date_str: Optional[str], days: Optional[int], timezone: str) -> List[str]:
    """Resolve explicit or rolling dates in the configured business timezone."""
    if date_str:
        datetime.date.fromisoformat(date_str)
        return [date_str]
    if days is None or not 1 <= days <= 7:
        raise ValueError("--days must be between 1 and 7")
    today = datetime.datetime.now(ZoneInfo(timezone)).date()
    return [(today + datetime.timedelta(days=offset)).isoformat() for offset in range(days)]


def _accept_cookies(page) -> None:
    for selector in (
        '#onetrust-accept-btn-handler',
        'button:has-text("Accept All")',
        'button:has-text("Accept")',
        '.truste-button1',
    ):
        try:
            button = page.locator(selector).first
            if button.is_visible(timeout=400):
                button.click()
                page.wait_for_timeout(500)
                return
        except Exception:
            continue


def _race_link(url: str, wanted_date: str) -> Optional[Tuple[str, str]]:
    parsed = urlsplit(urljoin(BASE_URL, url))
    match = RACE_PATH_RE.match(parsed.path)
    if not match or match.group("date") != wanted_date:
        return None
    return match.group("race_id"), parsed.path


def _state_race_links(value: Any, wanted_date: str, depth: int = 0) -> List[Tuple[str, str]]:
    """Find race URLs in Next.js state, including meetings not mounted in the DOM."""
    if depth > 10:
        return []
    races: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        for key in ("raceUrl", "url", "href"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                race = _race_link(candidate, wanted_date)
                if race:
                    races.append(race)
                    break
        for child in value.values():
            races.extend(_state_race_links(child, wanted_date, depth + 1))
    elif isinstance(value, list):
        for child in value:
            races.extend(_state_race_links(child, wanted_date, depth + 1))
    return races


def discover_races(page, date_str: str, timeout_ms: int) -> List[Tuple[str, str]]:
    """Discover individual race URLs from a browser-rendered date listing."""
    listing_url = f"{BASE_URL}/racecards/{date_str}/"
    response = page.goto(listing_url, wait_until="domcontentloaded", timeout=timeout_ms)
    if response is not None and response.status >= 400:
        raise RuntimeError(f"listing page returned HTTP {response.status}: {listing_url}")

    _accept_cookies(page)
    try:
        page.wait_for_selector('a[href*="/racecards/"]', timeout=min(timeout_ms, 15000))
    except Exception:
        pass
    page.wait_for_timeout(750)

    hrefs = page.eval_on_selector_all(
        'a[href*="/racecards/"]',
        "elements => elements.map(element => element.href)",
    )
    races: List[Tuple[str, str]] = []
    try:
        next_data_text = page.locator("#__NEXT_DATA__").text_content(timeout=2000)
        if next_data_text:
            races.extend(_state_race_links(json.loads(next_data_text), date_str))
    except Exception:
        pass

    seen = set()
    for href in hrefs:
        race = _race_link(href, date_str)
        if race:
            races.append(race)

    deduplicated = []
    for race in races:
        if race not in seen:
            seen.add(race)
            deduplicated.append(race)
    return deduplicated


def _nested_value(value: Any, *keys: str) -> Any:
    if not isinstance(value, dict):
        return None
    for key in keys:
        candidate = value.get(key)
        if candidate not in (None, ""):
            return candidate
    return None


def _name(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(_nested_value(value, "name", "styleName", "stylename") or "")


def normalize_runner(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize current and legacy browser JSON runner shapes."""
    horse = raw.get("horse") if isinstance(raw.get("horse"), dict) else {}
    jockey = raw.get("jockey") if isinstance(raw.get("jockey"), dict) else {}
    trainer = raw.get("trainer") if isinstance(raw.get("trainer"), dict) else {}
    figures = raw.get("figuresCalculated") or raw.get("figures") or []
    return {
        "horseAge": _nested_value(raw, "horseAge", "age") or horse.get("age"),
        "horseName": _nested_value(raw, "horseName", "name") or _name(horse),
        "startNumber": _nested_value(raw, "startNumber", "number", "saddleClothNumber"),
        "draw": _nested_value(raw, "draw", "stall"),
        "jockeyName": _nested_value(raw, "jockeyName") or _name(jockey),
        "jockeyUid": _nested_value(raw, "jockeyUid", "jockeyId") or jockey.get("id"),
        "trainerStylename": _nested_value(raw, "trainerStylename", "trainerName") or _name(trainer),
        "trainerId": _nested_value(raw, "trainerId", "trainerUid") or trainer.get("id"),
        "horseUid": _nested_value(raw, "horseUid", "horseId") or horse.get("id"),
        "officialRatingToday": _nested_value(raw, "officialRatingToday", "officialRating", "or"),
        "rpPostmark": _nested_value(raw, "rpPostmark", "rpr"),
        "rpTopspeed": _nested_value(raw, "rpTopspeed", "topspeed", "ts"),
        "weightCarriedLbs": _nested_value(raw, "weightCarriedLbs", "lhWeightCarriedLbs", "weight"),
        "rpHorseHeadGearCode": _nested_value(raw, "rpHorseHeadGearCode", "horseHeadGear", "headgear"),
        "figuresCalculated": figures if isinstance(figures, list) else [],
        "daysSinceLastRun": _nested_value(raw, "daysSinceLastRun", "lastRun"),
        "nonRunner": bool(_nested_value(raw, "nonRunner", "isNonRunner") or False),
        "raceDatetime": _nested_value(raw, "raceDatetime", "raceDateTime", "offTime"),
        "raceTypeCode": _nested_value(raw, "raceTypeCode", "raceType"),
        "courseUid": _nested_value(raw, "courseUid", "courseId"),
        "distanceFurlongRounded": _nested_value(raw, "distanceFurlongRounded", "distanceFurlongs"),
        "distanceYard": _nested_value(raw, "distanceYard", "distanceYards"),
    }


def _runner_collection(value: Any, depth: int = 0) -> List[Dict[str, Any]]:
    """Find a runner collection inside a browser-captured JSON response."""
    if depth > 7:
        return []
    if isinstance(value, dict):
        for key in ("runners", "raceRunners", "cardRunners"):
            collection = value.get(key)
            candidates = list(collection.values()) if isinstance(collection, dict) else collection
            if isinstance(candidates, list):
                normalized = [normalize_runner(item) for item in candidates if isinstance(item, dict)]
                if any(item.get("horseName") for item in normalized):
                    return normalized
        for child in value.values():
            found = _runner_collection(child, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _runner_collection(child, depth + 1)
            if found:
                return found
    return []


def _dom_runners(page) -> List[Dict[str, Any]]:
    """Extract a minimal runner set when rendered JSON is not exposed."""
    raw = page.evaluate(
        r"""
        () => Array.from(document.querySelectorAll(
          '[data-test-selector="RC-cardPage-runnerName"]'
        )).map(nameElement => {
          const row = nameElement.closest(
            '[data-test-selector*="runnerRow"], [class*="runnerRow"], article, li'
          ) || nameElement.parentElement?.parentElement || nameElement.parentElement;
          const text = selector => row?.querySelector(selector)?.textContent?.trim() || '';
          const profileId = kind => {
            const href = row?.querySelector(`a[href*="/profile/${kind}/"]`)?.href || '';
            const match = href.match(/\/(\d+)(?:\/|$)/);
            return match ? Number(match[1]) : null;
          };
          const rowText = row?.textContent || '';
          const numberMatch = rowText.match(/^\s*(\d{1,2})\b/);
          return {
            horseName: nameElement.textContent?.trim() || '',
            horseUid: profileId('horse'),
            jockeyName: text('[data-test-selector*="jockey"], a[href*="/profile/jockey/"]'),
            jockeyUid: profileId('jockey'),
            trainerStylename: text('[data-test-selector*="trainer"], a[href*="/profile/trainer/"]'),
            trainerId: profileId('trainer'),
            startNumber: numberMatch ? Number(numberMatch[1]) : null,
            nonRunner: /non[- ]?runner/i.test(rowText)
          };
        })
        """
    )
    return [normalize_runner(item) for item in raw if item.get("horseName")]


def _first_text(page, selectors: Iterable[str]) -> str:
    for selector in selectors:
        try:
            text = page.locator(selector).first.inner_text(timeout=1000).strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def _page_meta(page, date_str: str, href: str) -> Dict[str, Any]:
    header_text = _first_text(page, ("h1", "header", "body"))
    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", header_text)
    course_match = RACE_PATH_RE.match(urlsplit(href).path)
    race_name = _first_text(page, (
        '[data-test-selector*="raceInstanceTitle"]',
        '[class*="raceInstanceTitle"]',
    ))
    combined = f"{race_name} {header_text}".lower()
    race_type = "C" if "chase" in combined else "H" if "hurdle" in combined else "B" if "bumper" in combined else "F"
    course_name = _first_text(page, (
        '[data-test-selector="RC-courseHeader__name"]',
        '[class*="RC-courseHeader__name"]',
        "h1",
    ))
    return {
        "raceDatetime": f"{date_str}T{time_match.group(1)}:00" if time_match else "",
        "raceTypeCode": race_type,
        "courseUid": int(course_match.group("course_id")) if course_match else None,
        "courseName": course_name,
        "raceName": race_name,
        "going": _first_text(page, ('[data-test-selector*="going"]', '[class*="going"]')),
    }


def collect_race(page, race_id: str, href: str, date_str: str, timeout_ms: int) -> Optional[Dict[str, Any]]:
    """Render one race page and turn captured JSON/DOM data into a racecard."""
    captured_responses = []

    def remember_json(response) -> None:
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            captured_responses.append(response)

    page.on("response", remember_json)
    try:
        url = urljoin(BASE_URL, href)
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if response is not None and response.status >= 400:
            raise RuntimeError(f"race page returned HTTP {response.status}: {url}")
        _accept_cookies(page)
        if page.locator("#__NEXT_DATA__").count() == 0:
            try:
                page.wait_for_selector(
                    '[data-test-selector="RC-cardPage-runnerName"]',
                    timeout=min(timeout_ms, 15000),
                )
            except Exception:
                pass
        page.wait_for_timeout(750)

        runners: List[Dict[str, Any]] = []
        for captured in reversed(captured_responses):
            try:
                runners = _runner_collection(captured.json())
            except Exception:
                continue
            if runners:
                break
        if not runners:
            runners = _dom_runners(page)

        page_html = page.content().encode("utf-8")
        return build_racecard_from_html(
            race_id=race_id,
            href=href,
            date=date_str,
            page_html=page_html,
            runners_list=runners,
            race_meta=_page_meta(page, date_str, href),
        )
    finally:
        page.remove_listener("response", remember_json)


def _write_validated_snapshot(date_str: str, racecards: Dict[str, Any], output_dir: Path) -> Path:
    race_count = sum(
        len(times)
        for courses in racecards.values()
        for times in courses.values()
    )
    runner_count = sum(
        len(race.get("runners") or [])
        for courses in racecards.values()
        for times in courses.values()
        for race in times.values()
    )
    if race_count < 1 or runner_count < 1:
        raise RuntimeError(
            f"refusing to replace snapshot: collected {race_count} races and {runner_count} runners"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"racecards_{date_str}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(racecards, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _launch_context(playwright, headless: bool, profile_dir: Optional[str]):
    launch_options = {"headless": headless, "channel": "chrome"}
    if profile_dir:
        profile_path = Path(profile_dir).expanduser().resolve()
        profile_path.mkdir(parents=True, exist_ok=True)
        try:
            return playwright.chromium.launch_persistent_context(str(profile_path), **launch_options), None
        except Exception as exc:
            raise RuntimeError(f"could not launch persistent system Chrome: {exc}") from exc

    try:
        browser = playwright.chromium.launch(**launch_options)
    except Exception:
        browser = playwright.chromium.launch(headless=headless)
    return browser.new_context(locale="en-GB", timezone_id="Europe/London"), browser


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Racing Post racecards with visible Chrome")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single date (YYYY-MM-DD)")
    group.add_argument("--days", type=int, help="Number of days from today (1-7)")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone used with --days")
    parser.add_argument("--profile-dir", default=os.getenv("RACING_POST_PROFILE_DIR"))
    parser.add_argument("--headless", action="store_true", help="Use only where ordinary headless access is permitted")
    parser.add_argument("--timeout-ms", type=int, default=45000)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--race-attempts", type=int, default=2)
    parser.add_argument("--max-races", type=int, help="Optional diagnostic limit")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--capture-html-dir",
        type=Path,
        help="Optional diagnostic directory for rendered race-page HTML",
    )
    args = parser.parse_args()

    try:
        dates = target_dates(args.date, args.days, args.timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        parser.error(str(exc))

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright is not installed. Run: python -m pip install playwright", file=sys.stderr)
        return 2

    diagnostics_dir = Path(os.getenv("RUNNER_TEMP", "tmp")) / "racing-post-browser"
    failures = []
    with sync_playwright() as playwright:
        context, browser = _launch_context(playwright, args.headless, args.profile_dir)
        page = context.pages[0] if context.pages else context.new_page()
        try:
            for date_str in dates:
                print(f"Discovering Racing Post races for {date_str}...")
                try:
                    discovered = discover_races(page, date_str, args.timeout_ms)
                    if not discovered:
                        raise RuntimeError("no individual race links were rendered")
                    if args.max_races:
                        discovered = discovered[:max(1, args.max_races)]
                    print(f"Found {len(discovered)} races")

                    racecards = defaultdict(lambda: defaultdict(dict))
                    for index, (race_id, href) in enumerate(discovered, start=1):
                        print(f"[{index}/{len(discovered)}] Race {race_id}...", end=" ", flush=True)
                        race = None
                        last_error = "no usable runners"
                        for attempt in range(1, max(1, args.race_attempts) + 1):
                            try:
                                race = collect_race(page, race_id, href, date_str, args.timeout_ms)
                                if race:
                                    break
                                last_error = "no usable runners"
                            except Exception as exc:
                                last_error = str(exc)
                            if attempt < max(1, args.race_attempts):
                                page.wait_for_timeout(1000)
                        if args.capture_html_dir:
                            args.capture_html_dir.mkdir(parents=True, exist_ok=True)
                            (args.capture_html_dir / f"{date_str}_{race_id}.html").write_text(
                                page.content(), encoding="utf-8"
                            )
                        if not race:
                            print(f"FAILED ({last_error})")
                            continue
                        racecards[race["region"]][race["course"]][race["off_time"]] = race
                        print(f"OK ({len(race['runners'])} runners)")
                        if index < len(discovered):
                            time.sleep(max(0.0, args.delay_seconds))

                    collected_count = sum(
                        len(times)
                        for courses in racecards.values()
                        for times in courses.values()
                    )
                    if collected_count != len(discovered):
                        raise RuntimeError(
                            f"refusing partial snapshot: collected {collected_count} of "
                            f"{len(discovered)} discovered races"
                        )
                    output_path = _write_validated_snapshot(date_str, racecards, args.output_dir)
                    print(f"Saved validated browser snapshot to {output_path}")
                except Exception as exc:
                    failures.append(f"{date_str}: {exc}")
                    diagnostics_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        page.screenshot(path=str(diagnostics_dir / f"{date_str}.png"), full_page=True)
                        (diagnostics_dir / f"{date_str}.html").write_text(page.content(), encoding="utf-8")
                    except Exception:
                        pass
        finally:
            context.close()
            if browser is not None:
                browser.close()

    if failures:
        print("Browser collection failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
