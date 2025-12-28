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
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-GB,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            return resp.status_code, resp
        except requests.RequestException as e:
            if attempt == 2:
                print(f"Failed to fetch {url}: {e}")
                return 0, None
            time.sleep(2 ** attempt)  # Exponential backoff
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
    
    for date in dates:
        # Racing Post supports country-specific URLs
        if country:
            url = f'https://www.racingpost.com/racecards/{date}/{country}'
        else:
            url = f'https://www.racingpost.com/racecards/{date}'
        
        status, response = get_request(url)
        
        if status != 200 or not response:
            print(f"Failed to fetch racecards for {date}")
            continue
        
        doc = html.fromstring(response.content)
        
        for meeting in doc.xpath('//section[@data-accordion-row]'):
            course = meeting.xpath(".//span[contains(@class, 'RC-accordion__courseName')]")
            if course and valid_meeting(course[0].text_content().strip().lower()):
                for race in meeting.xpath(".//a[@class='RC-meetingItem__link js-navigate-url']"):
                    race_id = race.attrib.get('data-race-id')
                    href = race.attrib.get('href')
                    if race_id and href:
                        race_urls[date].append((race_id, href))
        
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
        
        if status_racecard != 200 or status_runners != 200:
            print(f"FAILED (status: {status_racecard}, {status_runners})")
            continue
        
        try:
            doc = html.fromstring(resp_racecard.content)
        except Exception as e:
            print(f"FAILED (parse error: {e})")
            continue
        
        try:
            runners_json = resp_runners.json()['runners']
            runners_list = list(runners_json.values())
            if not runners_list:
                print("SKIP (no runners)")
                continue
            runner = runners_list[0]
        except Exception as e:
            print(f"FAILED (JSON error: {e})")
            continue
        
        race = {}
        
        race['href'] = url_racecard
        race['race_id'] = int(race_id)
        race['date'] = date
        
        # Race time
        date_str = runner.get('raceDatetime', '')
        if date_str:
            try:
                dt = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                race['off_time'] = dt.strftime('%H:%M')
            except Exception:
                race['off_time'] = ""
        else:
            race['off_time'] = ""
        
        # Course
        race['course_id'] = runner.get('courseUid')
        race['course'] = find(doc, 'h1', 'RC-courseHeader__name')
        race['course_detail'] = find(doc, 'span', 'RC-header__straightRoundJubilee').strip('()')
        
        # Handle special course names
        if race['course'] == 'Belmont At The Big A':
            race['course_id'] = 255
            race['course'] = 'Aqueduct'
        
        race['region'] = get_region(str(race['course_id']))
        
        # Race details
        race['race_name'] = find(doc, 'span', 'RC-header__raceInstanceTitle')
        race['race_type'] = RACE_TYPE.get(runner.get('raceTypeCode', 'F'), 'Flat')
        
        # Distance
        race['distance_f'] = runner.get('distanceFurlongRounded')
        race['distance_y'] = runner.get('distanceYard')
        race['distance_round'] = find(doc, 'strong', 'RC-header__raceDistanceRound')
        race['distance'] = find(doc, 'span', 'RC-header__raceDistance').strip('()')
        race['distance'] = race['distance'] or race['distance_round']
        
        # Class and pattern
        race['pattern'] = get_pattern(race['race_name'].lower())
        race_class_str = find(doc, 'span', 'RC-header__raceClass')
        race_class_str = race_class_str.replace('Class', '').strip('()').strip()
        race['race_class'] = int(race_class_str) if race_class_str.isdigit() else None
        race['race_class'] = 1 if not race['race_class'] and race['pattern'] else race['race_class']
        
        race['age_band'], race['rating_band'] = parse_age_and_rating(doc)
        race['prize'] = parse_prize(doc)
        race['field_size'] = parse_field_size(doc)
        
        race['handicap'] = race['rating_band'] is not None or 'handicap' in race['race_name'].lower()
        
        race['going'] = parse_going(doc)
        race['surface'] = get_surface(race['going'])
        
        # Runners (simplified - just basic info)
        race['runners'] = []
        for runner_json in runners_list:
            r = {
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
                'form': ''.join(f.get('formFigure', '') for f in runner_json.get('figuresCalculated', []))[::-1] if runner_json.get('figuresCalculated') else '',
                'last_run': runner_json.get('daysSinceLastRun'),
                'non_runner': runner_json.get('nonRunner', False),
            }
            race['runners'].append(r)
        
        # Store in nested structure
        races[race['region']][race['course']][race['off_time']] = race
        print("OK")
        
        time.sleep(0.5)  # Be polite to the server
    
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
    for date in dates:
        if date not in race_urls or not race_urls[date]:
            print(f"\nNo races found for {date}")
            continue
        
        print(f"\nScraping {len(race_urls[date])} races for {date}...")
        racecards = scrape_racecards(race_urls, date)
        
        # Save to JSON
        output_path = output_dir / f'racecards_{date}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(racecards, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved racecards to {output_path}")


if __name__ == '__main__':
    main()
