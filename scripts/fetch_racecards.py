#!/usr/bin/env python
"""
Standalone racecards scraper for Racing Post.

Outputs JSON in same format as external scraper, compatible with predict_todays_races.py.

Usage:
  python scripts/fetch_racecards.py --date 2025-12-22
  python scripts/fetch_racecards.py --days 2  # Next 2 days
  python scripts/fetch_racecards.py --date 2025-12-22 --country uk
  python scripts/fetch_racecards.py --date 2025-12-22 --country us
"""

import argparse
import datetime
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from lxml import html


SESSION = requests.Session()
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
]


# ============================================================================
# Utility Functions (inlined)
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize horse/trainer/jockey name."""
    if not name:
        return ""
    # Replace common variations
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)  # Collapse whitespace
    return name


def get_request(url: str, headers: dict = None) -> Tuple[int, requests.Response]:
    """Make HTTP GET request with retry logic."""
    base_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-GB,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    }

    max_attempts = 4
    retry_statuses = {403, 406, 429, 503}

    for attempt in range(max_attempts):
        try:
            request_headers = dict(base_headers)
            request_headers['User-Agent'] = USER_AGENTS[attempt % len(USER_AGENTS)]
            if headers:
                request_headers.update(headers)

            resp = SESSION.get(url, headers=request_headers, timeout=(10, 30))
            if resp.status_code in retry_statuses and attempt < (max_attempts - 1):
                time.sleep((2 ** attempt) + 0.5)
                continue
            return resp.status_code, resp
        except requests.RequestException as e:
            if attempt == (max_attempts - 1):
                print(f"Failed to fetch {url}: {e}")
                return 0, None
            time.sleep((2 ** attempt) + 0.5)  # Exponential backoff with jitter
    return 0, None


def find(doc, tag: str, class_name: str) -> str:
    """Find first element by tag and class, return text content."""
    try:
        elements = doc.xpath(f".//{tag}[contains(@class, '{class_name}')]")
        if elements:
            return elements[0].text_content().strip()
    except Exception:
        pass
    return ""


def get_region(course_id: str) -> str:
    """Map course_id to region (GB/IRE/FR/etc)."""
    # Simplified mapping - extend as needed
    gb_courses = {16, 513, 206}  # Musselburgh, Wolverhampton, Deauville (example)
    if int(course_id) in gb_courses:
        return "GB"
    return "GB"  # Default to GB for UK racing


def get_surface(going: str) -> str:
    """Determine surface from going description."""
    going_lower = going.lower()
    if any(x in going_lower for x in ['standard', 'tapeta', 'polytrack', 'fibresand']):
        return "AW"
    return "Turf"


def valid_meeting(course_name: str) -> bool:
    """Check if meeting is valid (filter out non-GB/IRE)."""
    # Allow all UK/IRE meetings for now
    return True


# ============================================================================
# Race Type Mapping
# ============================================================================

RACE_TYPE = {
    'F': 'Flat',
    'X': 'Flat',
    'C': 'Chase',
    'H': 'Hurdle',
    'B': 'NH Flat',
    'W': 'NH Flat',
}


# ============================================================================
# Scraping Logic
# ============================================================================

def get_race_urls(dates: List[str], country: str = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get race URLs for given dates.
    
    Args:
        dates: List of dates in YYYY-MM-DD format
        country: Country filter (uk, ire, us, fr, uae, etc.) or None for all
    """
    race_urls = defaultdict(list)

    race_href_re = re.compile(
        r'^/racecards/(?P<course_id>\d+)/[^/]+/(?P<race_date>\d{4}-\d{2}-\d{2})/(?P<race_id>\d+)/?$'
    )

    def _extract_links_from_doc(doc, wanted_date: str) -> List[Tuple[str, str]]:
        extracted: List[Tuple[str, str]] = []

        # Legacy structure (kept for backwards compatibility)
        for meeting in doc.xpath('//section[@data-accordion-row]'):
            course = meeting.xpath(".//span[contains(@class, 'RC-accordion__courseName')]")
            if course and valid_meeting(course[0].text_content().strip().lower()):
                for race in meeting.xpath(".//a[contains(@class, 'RC-meetingItem__link')]"):
                    race_id = race.attrib.get('data-race-id')
                    href = race.attrib.get('href')
                    if race_id and href:
                        extracted.append((race_id, href))

        # Newer structure: discover race links from href pattern directly.
        for href in doc.xpath('//a/@href'):
            if not isinstance(href, str):
                continue
            match = race_href_re.match(href)
            if not match:
                continue
            if match.group('race_date') != wanted_date:
                continue
            extracted.append((match.group('race_id'), href))

        # Deduplicate while preserving order
        seen = set()
        deduped: List[Tuple[str, str]] = []
        for race_id, href in extracted:
            key = (race_id, href)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((race_id, href))
        return deduped
    
    for date in dates:
        candidate_urls = []
        if country:
            candidate_urls.append(f'https://www.racingpost.com/racecards/{date}/{country}')
            candidate_urls.append(f'https://www.racingpost.com/racecards/{date}/{country}/')
        candidate_urls.append(f'https://www.racingpost.com/racecards/{date}')
        candidate_urls.append(f'https://www.racingpost.com/racecards/{date}/')
        candidate_urls.append('https://www.racingpost.com/racecards/')
        candidate_urls.append('https://www.racingpost.com/racecards/tomorrow/')

        try:
            next_day = (
                datetime.datetime.strptime(date, '%Y-%m-%d').date()
                + datetime.timedelta(days=1)
            ).isoformat()
            candidate_urls.append(f'https://www.racingpost.com/racecards/{next_day}/')
        except Exception:
            pass

        merged_links: List[Tuple[str, str]] = []
        for url in candidate_urls:
            status, response = get_request(url)
            if not response:
                continue
            if status not in (200, 406):
                continue

            try:
                doc = html.fromstring(response.content)
            except Exception:
                continue

            links = _extract_links_from_doc(doc, date)
            if links:
                merged_links.extend(links)

        # Final dedupe for the date
        seen = set()
        for race_id, href in merged_links:
            key = (race_id, href)
            if key in seen:
                continue
            seen.add(key)
            race_urls[date].append((race_id, href))

        if not race_urls[date]:
            print(f"Failed to discover races for {date} from listing pages")

        print(f"Found {len(race_urls[date])} races for {date}")
    
    return dict(race_urls)


def get_pattern(race_name: str) -> str:
    """Extract pattern from race name (Group/Grade/Listed)."""
    regex_group = r'(\(|\s)((G|g)rade|(G|g)roup) (\d|[A-Ca-c]|I*)(\)|\s)'
    match = re.search(regex_group, race_name)
    
    if match:
        pattern = f'{match.groups()[1]} {match.groups()[4]}'.title()
        return pattern
    
    if any(x in race_name.lower() for x in {'listed race', '(listed'}):
        return 'Listed'
    
    return ''


def parse_age_and_rating(doc) -> Tuple[str, str]:
    """Parse age band and rating band from header."""
    raw = find(doc, 'span', 'RC-header__rpAges')
    parts = raw.strip('()').split()
    age = parts[0] if len(parts) > 0 else None
    rating = parts[1] if len(parts) > 1 else None
    return age, rating


def parse_field_size(doc) -> int:
    """Parse field size from header."""
    raw = find(doc, 'div', 'RC-headerBox__runners').lower()
    if 'runners:' in raw:
        segment = raw.split('runners:', 1)[1]
        num_str = segment.split('(')[0].strip()
        try:
            return int(num_str)
        except ValueError:
            pass
    return None


def parse_going(doc) -> str:
    """Parse going description."""
    raw = find(doc, 'div', 'RC-headerBox__going').lower()
    going = raw.split('going:', 1)[1].strip().title() if 'going:' in raw else ''
    return going


def parse_prize(doc) -> str:
    """Parse prize money."""
    raw = find(doc, 'div', 'RC-headerBox__winner').lower()
    if 'winner:' in raw:
        return raw.split('winner:', 1)[1].strip()
    return None


def extract_runners_from_next_data(page_html: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract runner payload from Next.js JSON embedded in racecard HTML.

    Returns:
        (runners, race_meta)
        runners: list of runner dicts normalized to legacy cardrunners-style keys
        race_meta: race-level fields used by downstream mapping
    """
    try:
        text = page_html.decode('utf-8', errors='ignore')
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            text,
            re.S,
        )
        if not match:
            return [], {}

        next_data = json.loads(match.group(1))
        data = (
            next_data.get('props', {})
            .get('pageProps', {})
            .get('initialState', {})
            .get('racePage', {})
            .get('data', {})
        )
        race = data.get('race', {})
        runners_raw = data.get('runners', []) or []
        if not runners_raw:
            return [], {}

        race_meta = {
            'raceDatetime': race.get('localMeetingRaceDateTime') or race.get('raceTime'),
            'raceTypeCode': race.get('raceType'),
            'courseUid': race.get('courseId'),
            'courseName': race.get('meetingName') or race.get('courseName'),
            'raceName': race.get('raceTitle'),
            'distanceFurlongRounded': race.get('distanceFurlongs'),
            'distanceYard': race.get('distanceYards'),
            'distance': race.get('displayDistance'),
            'going': race.get('going'),
            'region': race.get('countryCode'),
            'raceClass': race.get('raceClass'),
            'ageBand': race.get('agesAllowed'),
            'ratingBand': race.get('officialRatingBandDesc'),
            'prize': race.get('formattedTotalPrizeMoney'),
            'fieldSize': race.get('numberOfRunners'),
            'surfaceType': race.get('surfaceType') or race.get('awSurfaceType'),
        }

        normalized = []
        for runner in runners_raw:
            normalized.append({
                'horseAge': runner.get('age'),
                'horseName': runner.get('horseName'),
                'startNumber': runner.get('startNumber'),
                'draw': runner.get('draw'),
                'jockeyName': runner.get('jockeyName'),
                'jockeyUid': runner.get('jockeyId'),
                'trainerStylename': runner.get('trainerName'),
                'trainerId': runner.get('trainerId'),
                'horseUid': runner.get('horseId'),
                'officialRatingToday': runner.get('officialRatingToday'),
                'rpPostmark': runner.get('rpPostmark'),
                'rpTopspeed': runner.get('rpTopspeed'),
                'weightCarriedLbs': runner.get('lhWeightCarriedLbs') or runner.get('weightCarried'),
                'rpHorseHeadGearCode': runner.get('horseHeadGear') or runner.get('headGearCode'),
                'figuresCalculated': runner.get('figuresCalculated') or runner.get('formFiguresData') or [],
                'daysSinceLastRun': runner.get('daysSinceLastRun'),
                'nonRunner': runner.get('nonRunner', False),
                # Keep legacy field name consumed by off-time parsing.
                'raceDatetime': race_meta['raceDatetime'],
                'raceTypeCode': race_meta['raceTypeCode'],
                'courseUid': race_meta['courseUid'],
                'distanceFurlongRounded': race_meta['distanceFurlongRounded'],
                'distanceYard': race_meta['distanceYard'],
            })

        return normalized, race_meta
    except Exception:
        return [], {}


def build_racecard_from_html(
    race_id: str,
    href: str,
    date: str,
    page_html: bytes,
    runners_list: List[Dict[str, Any]] = None,
    race_meta: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build the canonical racecard object from rendered Racing Post HTML.

    ``runners_list`` and ``race_meta`` may be supplied by a browser response or
    DOM extraction. Embedded Next.js data remains the fallback for the requests
    scraper. Returning ``None`` means that the page did not contain usable
    runner data and must not be persisted as a successful fetch.
    """
    try:
        doc = html.fromstring(page_html)
    except Exception:
        return None

    runners_list = list(runners_list or [])
    race_meta = dict(race_meta or {})

    embedded_runners, embedded_meta = extract_runners_from_next_data(page_html)
    if embedded_runners and len(embedded_runners) >= len(runners_list):
        runners_list = embedded_runners
    for key, value in embedded_meta.items():
        if value not in (None, ""):
            race_meta[key] = value

    if not runners_list:
        return None

    runner = runners_list[0]
    url_racecard = href if href.startswith('http') else f'https://www.racingpost.com{href}'
    race = {}

    race['href'] = url_racecard
    race['race_id'] = int(race_id)
    race['date'] = date

    date_str = runner.get('raceDatetime', '') or race_meta.get('raceDatetime', '')
    if date_str:
        try:
            dt = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            race['off_time'] = dt.strftime('%H:%M')
        except Exception:
            time_match = re.search(r'\b(\d{1,2}:\d{2})\b', str(date_str))
            race['off_time'] = time_match.group(1) if time_match else ""
    else:
        race['off_time'] = ""

    race['course_id'] = runner.get('courseUid') or race_meta.get('courseUid')
    if not race['course_id']:
        course_match = re.search(r'/racecards/(\d+)/', url_racecard)
        race['course_id'] = int(course_match.group(1)) if course_match else None
    race['course'] = find(doc, 'h1', 'RC-courseHeader__name') or race_meta.get('courseName', '')
    if not race['course']:
        course_match = re.search(r'/racecards/\d+/([^/]+)/', url_racecard)
        if course_match:
            race['course'] = course_match.group(1).replace('-', ' ').title()
    race['course_detail'] = find(doc, 'span', 'RC-header__straightRoundJubilee').strip('()')

    if race['course'] == 'Belmont At The Big A':
        race['course_id'] = 255
        race['course'] = 'Aqueduct'

    race['region'] = race_meta.get('region')
    if not race['region']:
        try:
            race['region'] = get_region(str(race['course_id']))
        except (TypeError, ValueError):
            race['region'] = 'GB'

    race['race_name'] = (
        find(doc, 'span', 'RC-header__raceInstanceTitle')
        or race_meta.get('raceName', '')
    )
    race['race_type'] = RACE_TYPE.get(
        runner.get('raceTypeCode') or race_meta.get('raceTypeCode') or 'F',
        'Flat'
    )

    race['distance_f'] = runner.get('distanceFurlongRounded') or race_meta.get('distanceFurlongRounded')
    race['distance_y'] = runner.get('distanceYard') or race_meta.get('distanceYard')
    race['distance_round'] = find(doc, 'strong', 'RC-header__raceDistanceRound')
    race['distance'] = find(doc, 'span', 'RC-header__raceDistance').strip('()')
    race['distance'] = race['distance'] or race['distance_round'] or race_meta.get('distance', '')

    race['pattern'] = get_pattern(race['race_name'].lower())
    race_class_str = find(doc, 'span', 'RC-header__raceClass')
    race_class_str = race_class_str.replace('Class', '').strip('()').strip()
    race['race_class'] = int(race_class_str) if race_class_str.isdigit() else None
    if race['race_class'] is None:
        try:
            race['race_class'] = int(race_meta.get('raceClass'))
        except (TypeError, ValueError):
            pass
    race['race_class'] = 1 if not race['race_class'] and race['pattern'] else race['race_class']

    race['age_band'], race['rating_band'] = parse_age_and_rating(doc)
    race['age_band'] = race['age_band'] or race_meta.get('ageBand')
    race['rating_band'] = race['rating_band'] or race_meta.get('ratingBand')
    race['prize'] = parse_prize(doc) or race_meta.get('prize')
    race['field_size'] = parse_field_size(doc) or race_meta.get('fieldSize') or len(runners_list)
    race['handicap'] = race['rating_band'] is not None or 'handicap' in race['race_name'].lower()
    race['going'] = parse_going(doc) or race_meta.get('going', '')
    race['surface'] = race_meta.get('surfaceType') or get_surface(race['going'])

    race['runners'] = []
    for runner_json in runners_list:
        race['runners'].append({
            'age': runner_json.get('horseAge'),
            'name': normalize_name(runner_json.get('horseName', '')),
            'number': runner_json.get('startNumber'),
            'draw': runner_json.get('draw'),
            'jockey': normalize_name(runner_json.get('jockeyName', '')),
            'jockey_id': runner_json.get('jockeyUid'),
            'trainer': normalize_name(runner_json.get('trainerStylename', '')),
            'trainer_id': runner_json.get('trainerId'),
            'horse_id': runner_json.get('horseUid'),
            'ofr': runner_json.get('officialRatingToday'),
            'rpr': runner_json.get('rpPostmark'),
            'ts': runner_json.get('rpTopspeed'),
            'lbs': runner_json.get('weightCarriedLbs'),
            'headgear': runner_json.get('rpHorseHeadGearCode'),
            'form': ''.join(
                f.get('formFigure', f.get('figure', ''))
                for f in runner_json.get('figuresCalculated', [])
            )[::-1] if runner_json.get('figuresCalculated') else '',
            'last_run': runner_json.get('daysSinceLastRun'),
            'non_runner': runner_json.get('nonRunner', False),
        })

    if not any(runner['name'] for runner in race['runners']):
        return None
    return race


def scrape_racecards(race_urls: Dict[str, List[Tuple[str, str]]], date: str) -> Dict:
    """Scrape racecards for a given date."""
    races = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for race_id, href in race_urls[date]:
        print(f"Scraping race {race_id}...", end=' ')

        url_base = 'https://www.racingpost.com'
        url_racecard = f'{url_base}{href}'
        url_runners = f'{url_base}/profile/horse/data/cardrunners/{race_id}.json'

        status_racecard, resp_racecard = get_request(url_racecard)
        status_runners, resp_runners = get_request(url_runners)

        if status_racecard != 200 or not resp_racecard:
            print(f"FAILED (status: {status_racecard}, {status_runners})")
            continue

        runners_list: List[Dict[str, Any]] = []
        if status_runners == 200 and resp_runners is not None:
            try:
                runners_json = resp_runners.json()['runners']
                runners_list = list(runners_json.values())
            except Exception:
                runners_list = []

        race = build_racecard_from_html(
            race_id=race_id,
            href=href,
            date=date,
            page_html=resp_racecard.content,
            runners_list=runners_list,
        )
        if not race:
            print(f"FAILED (status: {status_racecard}, {status_runners})")
            continue

        races[race['region']][race['course']][race['off_time']] = race
        print("OK")
        time.sleep(0.5)

    return races


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scrape racecards from Racing Post.'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--date', help='Single date (YYYY-MM-DD)')
    group.add_argument('--days', type=int, help='Number of days from today (1-7)')
    
    parser.add_argument(
        '--country',
        help='Country filter (uk, ire, us, fr, uae, etc.) - omit for all countries',
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine dates
    if args.date:
        dates = [args.date]
    else:
        if not (1 <= args.days <= 7):
            print("Error: --days must be between 1 and 7")
            sys.exit(1)
        dates = [
            (datetime.date.today() + datetime.timedelta(days=i)).isoformat()
            for i in range(args.days)
        ]
    
    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get race URLs
    country_msg = f" ({args.country.upper()})" if args.country else " (all countries)"
    print(f"\nFetching race URLs for {len(dates)} date(s){country_msg}...")
    race_urls = get_race_urls(dates, args.country)
    
    # Scrape racecards
    any_scraped = False
    any_discovered = False
    for date in dates:
        if date not in race_urls or not race_urls[date]:
            print(f"\nNo races found for {date}")
            # Persist an empty file so the UI can surface the date state explicitly.
            output_path = output_dir / f'racecards_{date}.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            print(f"Saved empty racecards to {output_path}")
            continue

        any_discovered = True
        
        print(f"\nScraping {len(race_urls[date])} races for {date}...")
        racecards = scrape_racecards(race_urls, date)
        
        # Save to JSON
        output_path = output_dir / f'racecards_{date}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(racecards, f, indent=2, ensure_ascii=False)

        scraped_count = sum(
            len(times)
            for region in racecards.values()
            for course in region.values()
            for times in [course]
        )
        if scraped_count > 0:
            any_scraped = True
        
        print(f"\nSaved racecards to {output_path}")

    if any_discovered and not any_scraped:
        print("\nError: races were discovered but none could be scraped (likely Racing Post anti-bot blocking).")
        sys.exit(1)


if __name__ == '__main__':
    main()
