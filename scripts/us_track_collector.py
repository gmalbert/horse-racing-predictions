"""Bounded, isolated collection of dated US entry cards.

Never infer a card date from the requested URL or a calendar widget. The
provider's card heading must confirm both the track and the requested date.
HTTP failures, unavailable cards and parser failures have distinct statuses.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
BLOCK_MARKERS = ("pardon our interruption", "access denied", "just a moment", "verify you are human", "captcha")


def clean(value):
    return " ".join(value.replace("\xa0", " ").split())


def soup_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "select"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def contains_date(text, date_str):
    """Match actual full dates, never a bare day/month from a calendar."""
    day = datetime.strptime(date_str, "%Y-%m-%d")
    text = clean(text).lower()
    forms = [date_str, f"{day.month}/{day.day}/{day.year}", day.strftime("%m/%d/%Y"),
             day.strftime("%m/%d/%y"), f"{day:%B} {day.day}, {day.year}",
             f"{day:%b} {day.day}, {day.year}"]
    return any(re.search(r"(?<![\w/])" + re.escape(v.lower()) + r"(?![\w/])", text) for v in forms)


def link_rank(url, label, date_str):
    text = (url + " " + label).lower()
    day = datetime.strptime(date_str, "%Y-%m-%d")
    score = 0
    if any(v in text for v in (date_str, day.strftime("%m%d%y"), day.strftime("%m-%d-%Y"))) or contains_date(label, date_str):
        score += 100
    if any(v in text for v in ("entries", "/entry/", "racecard")):
        score += 20
    if any(v in text for v in ("results", "chart", "replay", "news", "stakes", "calendar", "conditions")):
        score -= 30
    if any(v in text for v in ("login", "purchase", "premium", "signup")):
        score -= 200
    return score


def discover_links(html, base_url, date_str):
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).hostname or ""
    candidates = {}
    for tag in soup.select("a[href], iframe[src]"):
        url = urljoin(base_url, tag.get("href") or tag.get("src"))
        parsed = urlparse(url)
        host = parsed.hostname or ""
        allowed = host == base_host or any(host == domain or host.endswith('.' + domain)
                  for domain in ("equibase.com", "ustrotting.com", "drf.com"))
        if parsed.scheme not in {"http", "https"} or not allowed:
            continue
        label = tag.get_text(" ", strip=True)
        rank = link_rank(url, label, date_str)
        if rank > 0 and not parsed.fragment:
            candidates[url] = max(rank, candidates.get(url, -999))
    return sorted(candidates, key=lambda u: (-candidates[u], u))


def error_kind(message, status=None, text=""):
    lower = message.lower()
    if status in (401, 403) or any(m in text.lower() for m in BLOCK_MARKERS):
        return "blocked"
    if status == 404:
        return "not_found"
    for token, kind in [("name_not_resolved", "dns"), ("nameresolution", "dns"),
                        ("certificate", "tls"), ("ssl", "tls"), ("cert_", "tls"),
                        ("http2", "http2"), ("timeout", "timeout"), ("timed out", "timeout"),
                        ("interrupted by another", "navigation")]:
        if token in lower:
            return kind
    return "http" if status and status >= 400 else "transport"


def _http_get(url):
    response = requests.get(url, timeout=(8, 15), headers={"User-Agent": USER_AGENT})
    response.encoding = response.apparent_encoding or response.encoding
    return response.status_code, response.url, response.text


def _parse_drf_harness(payload, track_id, code, cfg, date_str, url):
    """Validate DRF's public entry-page data against track and actual post date."""
    stamp = datetime.strptime(date_str, '%Y-%m-%d').strftime('%m-%d-%Y')
    if payload.get('weather', {}).get('trackId') != track_id or not payload.get('isData'):
        return []
    races = []
    for race in payload.get('races', []):
        key = race.get('raceKey') or {}
        if race.get('isCancelled') or race.get('hasResults') or race.get('hasFinished'):
            continue
        # selectedDate alone can echo a request. The actual post timestamp must
        # also agree; DRF's legacy raceKey.raceDate serialization is unreliable.
        post_ms = race.get('postTimeLong')
        if (key.get('trackId') != track_id or key.get('country') != 'USA'
                or race.get('selectedDate') != stamp or not isinstance(post_ms, (int, float))):
            return []
        actual_day = datetime.fromtimestamp(post_ms / 1000, timezone.utc).astimezone(ZoneInfo('America/New_York')).date()
        if actual_day.isoformat() != date_str:
            return []
        runners = []
        for runner in race.get('runners') or []:
            if runner.get('scratchIndicator'):
                continue
            horse = clean(runner.get('horseName') or '')
            number = str(runner.get('programNumber') or '').strip()
            if not horse or not re.fullmatch(r'\d+[A-Z]?', number, re.I):
                continue
            def person(field):
                name = runner.get(field) or {}
                return clean(' '.join(str(name.get(part) or '') for part in ('firstName', 'middleName', 'lastName')))
            driver = person('jockey')
            runners.append({'horse': horse, 'number': number, 'draw': str(runner.get('postPos') or ''),
                            'jockey': driver, 'driver': driver, 'trainer': person('trainer'),
                            'ml_odds': str(runner.get('morningLineOdds') or ''), 'weight': ''})
        if not runners:
            return []
        description = race.get('raceTypeDescription') or ''
        purse = re.search(r'Purse:\s*\$([\d,]+)', description, re.I)
        distance = clean(description.split('|', 1)[0])
        if distance.upper() == 'ONE MILE':
            distance = '1 mile'
        races.append({'race': f"Race {key['raceNumber']}", 'race_number': key['raceNumber'],
                      'race_name': race.get('raceName') or f"Race {key['raceNumber']}",
                      'race_time': race.get('postTime') or '', 'date': date_str, 'track': code,
                      'course': cfg['name'], 'type': 'Harness', 'distance': distance,
                      'surface': race.get('surfaceDescription') or '', 'race_class': race.get('raceClass') or '',
                      'purse': purse.group(1) if purse else '', 'source': 'DRF-Harness-entries',
                      'source_url': url, 'runners': runners})
    return races


def _drf_harness(cfg, code, date_str):
    """Independent provider: the public JSON used by DRF's harness entry page."""
    stamp = datetime.strptime(date_str, '%Y-%m-%d').strftime('%m-%d-%Y')
    url = f'https://harness.drf.com/entries/tracks/date/{stamp}'
    attempts = []
    try:
        response = requests.get(url, timeout=(5, 12), headers={'User-Agent': USER_AGENT})
        attempts.append({'url': url, 'transport': 'drf-harness', 'status_code': response.status_code})
        response.raise_for_status()
        listing = response.json()
        ignored = {'park', 'raceway', 'racecourse', 'racing', 'harrahs', 'harrah', 's'}
        words = set(re.sub(r'[^a-z0-9]+', ' ', cfg['name'].lower()).split()) - ignored
        tracks = [t for t in listing.get('trackList', []) if t.get('country') == 'USA'
                  and words <= set(re.sub(r'[^a-z0-9]+', ' ', t.get('trackName', '').lower()).split())]
        if len(tracks) != 1:
            if not tracks and stamp in listing.get('dates', []):
                return {'status': 'not_listed', 'races': [], 'url': url, 'attempts': attempts}
            raise ValueError('DRF harness listing has no unique matching track')
        track_id = tracks[0]['trackId']
        url = f'https://harness.drf.com/entries/entryDetails/id/{track_id}/country/USA/date/{stamp}'
        response = requests.get(url, timeout=(5, 12), headers={'User-Agent': USER_AGENT})
        attempts.append({'url': url, 'transport': 'drf-harness', 'status_code': response.status_code})
        response.raise_for_status()
        races = _parse_drf_harness(response.json(), track_id, code, cfg, date_str, url)
        if not races:
            raise ValueError('DRF card failed dated runner validation')
        return {'status': 'ok', 'races': races, 'url': url, 'attempts': attempts}
    except Exception as exc:
        attempts.append({'url': url, 'transport': 'drf-harness', 'error_kind': error_kind(str(exc)), 'error': str(exc)})
        return {'status': 'collection_error', 'races': [], 'url': url, 'attempts': attempts}


def _usta_postback(session, response, link, values=None):
    """Submit the public ASP.NET form using its current server-issued state."""
    soup = BeautifulSoup(response.text, 'html.parser')
    target = re.search(r"__doPostBack\('([^']+)'", link.get('href', ''))
    if target is None:
        raise ValueError('USTA desktop navigation control is missing its form target')
    data = {i['name']: i.get('value', '') for i in soup.select('input[type="hidden"][name]')}
    data.update(values or {})
    data.update(__EVENTTARGET=target.group(1), __EVENTARGUMENT='')
    return session.post(response.url, data=data, timeout=(5, 10))


def _usta_desktop(cfg, date_str):
    """Independent desktop host fallback when the mobile host is unreachable."""
    attempts = []
    with requests.Session() as session:
        session.headers['User-Agent'] = USER_AGENT
        response = None

        def inspect(phase):
            response.encoding = response.apparent_encoding or response.encoding
            kind = error_kind('', response.status_code, soup_text(response.text)) if response.status_code >= 400 else None
            attempts.append({'url': response.url, 'final_url': response.url, 'transport': 'usta-desktop',
                             'phase': phase, 'status_code': response.status_code, 'error_kind': kind,
                             'error': None})
            response.raise_for_status()
            if any(marker in soup_text(response.text).lower() for marker in BLOCK_MARKERS):
                raise ValueError('USTA desktop returned an access-denied page')
            return BeautifulSoup(response.text, 'html.parser')

        try:
            response = session.get('https://racing.ustrotting.com/default.aspx', timeout=(5, 10))
            soup = inspect('track search')
            select = soup.select_one('select[id$="ddlsearchtrack_pm"]')
            link = soup.select_one('a[id$="lnkbtnsearchtrack_pm"]')
            if select is None or link is None:
                raise ValueError('USTA desktop track search markup is unrecognized')
            if not any(o.get('value') == cfg['usta'] for o in select.select('option')):
                # An absent track is not proof that the requested date has no card.
                raise ValueError('Track absent from USTA desktop search')
            response = _usta_postback(session, response, link, {select['name']: cfg['usta']})
            soup = inspect('dated card selection')
            candidates = [a for a in soup.select('a[href]') if 'lnkbtnrace_card' in a.get('href', '')
                          and contains_date(a.get_text(' ', strip=True), date_str)
                          and '(Qua)' not in a.find_parent('tr').get_text(' ', strip=True)]
            if not candidates:
                if soup.select('a[id*="lnkbtnrace_card"]'):
                    return {'html': '', 'url': response.url, 'status': 'not_listed', 'attempts': attempts}
                raise ValueError('USTA desktop card list markup is unrecognized')
            response = _usta_postback(session, response, candidates[0])
            soup = inspect('card summary')
            expected_races = []
            for row in soup.select('tr'):
                link = row.select_one('a[id*="lnkbtnrace_chart"]')
                if link is not None and link.find_parent('tr') is row and re.search(r'\bENTRIES\b', row.get_text(' ', strip=True)):
                    match = re.search(r'Race\s+(\d+)', link.get_text(' ', strip=True), re.I)
                    if match:
                        expected_races.append(int(match.group(1)))
            link = soup.select_one('a[id$="lnkbtnViewCard"]')
            if link is None:
                raise ValueError('USTA desktop full-card link is missing')
            response = _usta_postback(session, response, link)
            inspect('full card')
            return {'html': response.text, 'url': response.url, 'status': 'ok', 'attempts': attempts,
                    'expected_races': expected_races}
        except Exception as exc:
            final_url = response.url if response is not None else 'https://racing.ustrotting.com/default.aspx'
            attempts.append({'url': final_url, 'transport': 'usta-desktop', 'status_code': None,
                             'error_kind': error_kind(str(exc)), 'error': str(exc)})
            return {'html': '', 'url': final_url, 'status': 'collection_error', 'attempts': attempts}


def _parse_usta_desktop(html, url, code, cfg, date_str):
    """Read desktop entries by named columns; completed charts lack ML headers."""
    soup = BeautifulSoup(html, 'html.parser')
    races, current, columns = [], None, {}
    for tag in soup.find_all(['b', 'tr']):
        if tag.name == 'b':
            heading = clean(tag.get_text(' ', strip=True))
            match = re.match(r'RACE\s+(\d+)\s+-\s+', heading, re.I)
            if match:
                current, columns = None, {}
                if not contains_date(heading, date_str):
                    continue
                words = re.sub(r'[^a-z0-9]+', ' ', cfg['name'].lower()).split()
                if not all(w in heading.lower() for w in words if w not in {'raceway', 'racing'}):
                    continue
                info = tag.find_parent(class_='chartheaderinfo')
                metadata = clean(info.get_text(' ', strip=True)) if info else ''
                post = re.search(r'Post\s*(?:&nbsp)?Time:\s*(?:&nbsp)?(\d{1,2}:\d{2}\s*[AP]M)', metadata, re.I)
                current = {'race': f'Race {match.group(1)}', 'race_number': int(match.group(1)),
                           'race_name': f'Race {match.group(1)}', 'race_time': post.group(1) if post else '',
                           'distance': '1 mile', 'surface': '', 'type': 'Harness', 'date': date_str,
                           'track': code, 'course': cfg['name'], 'source': 'USTA-entries',
                           'source_url': url, 'runners': []}
                races.append(current)
            continue
        if current is None:
            continue
        cells = tag.find_all(['td', 'th'], recursive=False)
        labels = [clean(c.get_text(' ', strip=True)).lower() for c in cells]
        if {'pp', 'horse', 'driver', 'trainer', 'ml'}.issubset(labels):
            columns, index = {}, 0
            for cell, label in zip(cells, labels):
                columns[label] = index
                index += int(cell.get('colspan', 1))
            continue
        if not {'datagriditem', 'datagridaltitem'}.intersection(tag.get('class', [])) or not columns:
            continue
        def value(name):
            index = columns[name]
            if index >= len(cells):
                return ''
            # Alternate drivers appear below the named primary driver.
            link = cells[index].find('a')
            return clean((link or cells[index]).get_text(' ', strip=True))
        number, horse = value('pp'), value('horse')
        if not re.fullmatch(r'\d+[A-Z]?', number, re.I) or not horse:
            continue
        if re.search(r'\b(?:scratched|scr)\b', tag.get_text(' ', strip=True), re.I):
            continue
        current['runners'].append({'number': number, 'horse': horse, 'draw': number,
                                   'jockey': value('driver'), 'driver': value('driver'),
                                   'trainer': value('trainer'), 'ml_odds': value('ml'), 'weight': ''})
    return [r for r in races if r['runners']]


async def fetch_page(context, url):
    """HTTP/1.1 first; fresh browser tab per retry, always closed after failure."""
    attempts = []
    for transport in ("http", "browser"):
        page = None
        try:
            if transport == "http":
                status, final, html = await asyncio.to_thread(_http_get, url)
            else:
                page = await context.new_page()
                response = await page.goto(url, wait_until="domcontentloaded", timeout=18000)
                await page.wait_for_timeout(800)
                status, final, html = response.status if response else None, page.url, await page.content()
            text = soup_text(html)
            blocked = status in (401, 403) or any(m in text.lower() for m in BLOCK_MARKERS)
            kind = error_kind("", status, text) if blocked or (status and status >= 400) else None
            attempts.append({"url": url, "final_url": final, "transport": transport,
                             "status_code": status, "error_kind": kind, "error": None,
                             "response_excerpt": text[:500] if kind else ''})
            # Do not retry access denials/404s as if they were transport failures.
            # Try a separate, public entry provider instead.
            if kind or len(text) > 200 or transport == "browser":
                return {"url": final, "status_code": status, "html": html, "text": text,
                        "error_kind": kind, "attempts": attempts}
        except Exception as exc:
            attempts.append({"url": url, "transport": transport, "status_code": None,
                             "error_kind": error_kind(str(exc)), "error": str(exc)})
        finally:
            if page is not None:
                await page.close()
    return {"url": url, "status_code": None, "html": "", "text": "",
            "error_kind": attempts[-1]["error_kind"], "attempts": attempts}


def parse_entries(html, url, code, cfg, date_str):
    """Parse provider tables by column names, with track/date provenance."""
    soup = BeautifulSoup(html, "html.parser")
    host = urlparse(url).hostname or ''
    is_usta = host == 'ustrotting.com' or host.endswith('.ustrotting.com')
    is_equibase = host == 'equibase.com' or host.endswith('.equibase.com')
    if not is_usta and not is_equibase:
        return [], 'unsupported_source'
    headings = soup.select('h1, h2.track-info, h4:not(.card-header)')
    heading_text = " ".join(h.get_text(" ", strip=True) for h in headings)
    # The USTA card heading / Equibase track-info heading is separate from date selectors.
    if not contains_date(heading_text, date_str):
        return [], "date_unverified"
    track_words = re.sub(r"[^a-z0-9]+", " ", cfg["name"].lower()).split()
    if not all(word in heading_text.lower() for word in track_words if word not in {"racing", "raceway", "race", "course"}):
        return [], "track_mismatch"
    blocks = soup.select(".card" if is_usta else ".entryRace")
    if not is_usta and not blocks and not any(
            clean(th.get_text(' ', strip=True)).lower() == 'horse' for th in soup.select('table th')):
        # Equibase returns a correctly headed empty card scaffold for dates
        # without published entries. This is different from changed table markup.
        return [], 'not_listed'
    races = []
    for block in blocks:
        header = block.select_one(".card-header" if is_usta else "h4")
        if header is None:
            continue
        header_text = clean(header.get_text(" ", strip=True))
        race_match = re.search(r"\brace\s+(\d+)\b", header_text, re.I)
        if race_match is None:
            continue
        if is_usta and not header.select_one('.race-status.entries'):
            # Completed/cancelled results must never become future entries.
            continue
        runners = []
        for table in block.find_all("table"):
            headers = [clean(h.get_text(" ", strip=True)).lower() for h in table.select("thead th")]
            if "horse" not in headers:
                continue
            columns = {name: i for i, name in enumerate(headers)}
            for row in table.select("tbody tr"):
                cells = row.find_all("td", recursive=False)
                def value(*keys):
                    for key in keys:
                        idx = columns.get(key)
                        if idx is not None and idx < len(cells):
                            return clean(cells[idx].get_text(" ", strip=True))
                    return ""
                horse = re.sub(r"\s+\([A-Z]{2,3}\)$", "", value("horse"))
                number = value("p#", "pp")
                if not horse or not re.fullmatch(r"(?:\d+[A-Z]?|AE\d+)", number, re.I):
                    continue
                if re.search(r"\b(scratched|scr)\b", row.get_text(" ", strip=True), re.I) or "scratch" in ' '.join(row.get('class', [])).lower():
                    continue
                runners.append({"horse": horse, "number": number, "draw": value("pp"),
                                "jockey": value("jockey", "driver"), "driver": value("driver"),
                                "trainer": value("trainer"), "ml_odds": value("m/l", "odds"),
                                "weight": value("wgt")})
            break  # Ignore duplicate mobile/desktop presentation tables.
        if runners:
            race_number = int(race_match.group(1))
            post = re.search(r"\b\d{1,2}:\d{2}\s*[AP]M(?:\s+[ECMP][SD]?T)?", header_text, re.I)
            metadata = block.select_one('.race-info.marginTop20')
            details = metadata.select_one('h5') if metadata else None
            fields = [clean(t) for t in details.stripped_strings] if details else []
            race_class = metadata.select_one('h3') if metadata else None
            races.append({"race": f"Race {race_number}", "race_number": race_number,
                          "race_name": f"Race {race_number}", "race_time": post.group() if post else "",
                          "race_class": race_class.get_text(' ', strip=True) if race_class else '',
                          "purse": fields[0] if fields else '',
                          "distance": fields[1] if len(fields) > 1 else ('1 mile' if is_usta else ''),
                          "surface": fields[2] if len(fields) > 2 else '',
                          "type": 'Harness' if is_usta else ('Quarter Horse' if len(fields) > 1 and
                                  re.search(r'\b(?:Y|Yards)\b', fields[1], re.I) else 'Thoroughbred'),
                          "date": date_str, "track": code, "course": cfg["name"],
                          "source": "USTA-entries" if is_usta else "Equibase-entries",
                          "source_url": url, "runners": runners})
    return races, "ok" if races else "no_runner_tables"


async def collect_track(context, code, cfg, date_str, *, attempts=None):
    datetime.strptime(date_str, "%Y-%m-%d")
    attempts = [] if attempts is None else attempts
    races, documents = [], []
    provider_not_listed = False
    queue = []
    if cfg.get("usta"):
        queue.append(f"https://m.ustrotting.com/entriesandresults/tracklist.cfm?track_code={cfg['usta']}")
    if cfg.get("equibase"):
        stamp = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m%d%y")
        queue.extend(f"https://{host}/static/entry/{cfg['equibase']}{stamp}USA-EQB.html"
                     for host in ("tvg.equibase.com", "www.equibase.com"))
    queue.append(cfg["url"])
    visited = set()
    while queue and len(visited) < 7:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        page = await fetch_page(context, url)
        attempts.extend(page["attempts"])
        documents.append(page)
        if page["error_kind"]:
            if cfg.get('usta') and 'm.ustrotting.com/entriesandresults/tracklist.cfm' in url:
                independent = await asyncio.to_thread(_drf_harness, cfg, code, date_str)
                attempts.extend(independent['attempts'])
                if independent['races']:
                    races = independent['races']
                    documents.append({'url': independent['url'], 'html': '',
                                      'text': json.dumps(races), 'status_code': 200, 'error_kind': None})
                    break
                if independent['status'] == 'not_listed':
                    provider_not_listed = True
                desktop = await asyncio.to_thread(_usta_desktop, cfg, date_str)
                attempts.extend(desktop['attempts'])
                parsed = _parse_usta_desktop(desktop['html'], desktop['url'], code, cfg, date_str)
                missing = set(desktop.get('expected_races', [])) - {r['race_number'] for r in parsed}
                if missing:
                    attempts.append({'url': desktop['url'], 'transport': 'usta-desktop',
                                     'error_kind': 'incomplete_card', 'error': f'Missing published races: {sorted(missing)}'})
                    parsed = []
                if parsed:
                    races = parsed
                    documents.append({'url': desktop['url'], 'html': desktop['html'],
                                      'text': soup_text(desktop['html']), 'status_code': 200, 'error_kind': None})
                    break
                if desktop['status'] == 'not_listed':
                    provider_not_listed = True
            if page["error_kind"] == "not_found" and "/static/entry/" in url:
                provider_not_listed = True
            continue
        soup = BeautifulSoup(page["html"], "html.parser")
        if "tracklist.cfm" in url:
            track_heading = ' '.join(h.get_text(' ', strip=True) for h in soup.select('h3, h4'))
            expected = [w for w in cfg['name'].lower().split() if w not in {'raceway', 'racing'}]
            if not all(w in track_heading.lower() for w in expected):
                dates = soup.select('select#slctRaceDate option')
                tracks = soup.select('select#slctTrackCode option')
                requested_day_listed = any(o.get('value') == date_str for o in dates)
                track_in_listing = any(o.get('value') == cfg['usta'] for o in tracks)
                if requested_day_listed and tracks and not track_in_listing:
                    provider_not_listed = True
                    attempts[-1]['card_status'] = 'track_not_in_current_listing'
                else:
                    attempts[-1]['card_status'] = 'track_mismatch'
                continue
            links = [urljoin(page["url"], a['href']) for a in soup.select('a[href*="racecardbytrack.cfm"]')
                     if contains_date(a.get_text(" ", strip=True), date_str) and '(Qua)' not in a.get_text()]
            if links:
                queue[0:0] = links
            elif soup.select('a[href*="racecardbytrack.cfm"]'):
                provider_not_listed = True
            attempts[-1]["card_status"] = "date_link_found" if links else "date_not_listed"
            continue
        parsed, reason = parse_entries(page['html'], page['url'], code, cfg, date_str)
        attempts[-1]["card_status"] = reason
        if reason == 'not_listed':
            provider_not_listed = True
        if parsed:
            races = parsed
            break
        queue.extend(link for link in discover_links(page['html'], page['url'], date_str)
                     if link not in visited and not re.search(r"\.(pdf|csv|xml)(?:\?|$)", link, re.I))
    status = "ok" if races else ("not_listed" if provider_not_listed else "collection_error")
    selected = next((d for d in documents if races and d['url'] == races[0]['source_url']), None)
    if selected is None and documents:
        selected = next((d for d in reversed(documents) if not d['error_kind']), documents[-1])
    selected = selected or {"url": cfg['url'], "html": "", "text": "", "status_code": None, "error_kind": "transport"}
    return {"date": date_str, "track_code": code, "track_name": cfg['name'], "status": status,
            "attempts": attempts, "racecards": races, "html": selected['html'],
            "text_sample": selected['text'], "selected": {"final_url": selected['url'],
            "status_code": selected['status_code'], "blocked": selected['error_kind'] == 'blocked',
            "score": len(races) * 10 if races else -1},
            "best_page": {"url": selected['url'], "race_sections": [], "summary_fields": {}},
            "best_quality_score": len(races) * 10 if races else -1,
            "pages_tried": attempts, "race_sections": [], "race_section_count": len(races),
            "summary": {}, "downloaded_artifacts": [], "discovered_urls": []}


async def pull_tracks(tracks, date_str):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        # HTTP/1.1 transport is used first. This also avoids known broken HTTP/2
        # server implementations when browser rendering is necessary.
        browser = await p.chromium.launch(headless=True, args=["--disable-http2"])
        try:
            semaphore = asyncio.Semaphore(3)
            async def one(code, cfg):
                async with semaphore:
                    context = None
                    attempts = []
                    try:
                        try:
                            context = await browser.new_context(user_agent=USER_AGENT)
                            result = await asyncio.wait_for(collect_track(context, code, cfg, date_str,
                                                                          attempts=attempts), timeout=70)
                        except asyncio.TimeoutError:
                            result = {"date": date_str, "track_code": code, "track_name": cfg['name'],
                                      "status": "collection_error", "racecards": [], "html": "",
                                      "text_sample": "", "attempts": attempts + [{"url": cfg['url'],
                                      "error_kind": "timeout", "error": "70 second track budget exceeded"}]}
                        except Exception as exc:
                            # A parser/browser failure at one track must not cancel
                            # the other in-flight tracks in asyncio.gather.
                            result = {"date": date_str, "track_code": code, "track_name": cfg['name'],
                                      "status": "collection_error", "racecards": [], "html": "",
                                      "text_sample": "", "attempts": attempts + [{"url": cfg['url'],
                                      "error_kind": 'collector', "error": str(exc)}]}
                        print(f"{code}: {result['status']}, {len(result['racecards'])} races", flush=True)
                        return result
                    finally:
                        if context is not None:
                            await context.close()
            return await asyncio.gather(*(one(code, cfg) for code, cfg in tracks.items()))
        finally:
            await browser.close()


def save_group(group, date_str, records, root=ROOT):
    out = root / 'data' / 'raw' / f'us_t1_{group}'
    tracks_dir = out / 'tracksite'
    (tracks_dir / 'html').mkdir(parents=True, exist_ok=True)
    summaries = []
    for record in records:
        record = dict(record)
        code = record['track_code']
        (tracks_dir / 'html' / f'{code}_{date_str}.html').write_text(record.pop('html'), encoding='utf-8')
        (tracks_dir / f'{code}_{date_str}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        # Keep full diagnostics in the report uploaded by Actions.
        summaries.append({k: v for k, v in record.items() if k != 'text_sample'})
    report = {'date': date_str, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'tracks': summaries, 'summary': {'total_tracks': len(records),
              'successful_pulls': sum(r['status'] == 'ok' for r in records),
              'collection_errors': sum(r['status'] == 'collection_error' for r in records)}}
    path = out / f't1_{group}_tracksite_pull_report_{date_str}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return path


def run_cli(group, tracks):
    parser = argparse.ArgumentParser(description='Collect dated US entries with source diagnostics')
    parser.add_argument('--date', required=True)
    parser.add_argument('--tracks', nargs='+', choices=list(tracks), help='Limit collection to named tracks')
    args = parser.parse_args()
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        parser.error('--date must be YYYY-MM-DD')
    selected = {code: tracks[code] for code in args.tracks} if args.tracks else tracks
    records = asyncio.run(pull_tracks(selected, args.date))
    print(f'Saved source report: {save_group(group, args.date, records)}')
    if records and all(r['status'] == 'collection_error' for r in records):
        raise SystemExit(1)


def canonical_entry_rows(record, base_row):
    rows = []
    for race in record.get('racecards', []):
        if race.get('date') != record.get('date') or record.get('status') != 'ok':
            continue
        for runner in race.get('runners', []):
            row = base_row(record['date'], record['track_code'], record['track_name'], race['source_url'])
            row.update(event_type='entries', source=race['source'], race_number=race['race_number'],
                       race_name=race['race_name'], race_time=race.get('race_time', ''),
                       distance=race.get('distance', ''), surface=race.get('surface', ''),
                       purse=race.get('purse', ''), race_class=race.get('race_class', ''),
                       runner_number=runner['number'], runner_name=runner['horse'],
                       jockey=runner.get('jockey', ''), driver=runner.get('driver', ''),
                       trainer=runner.get('trainer', ''), ml_odds=runner.get('ml_odds', ''),
                       weight=runner.get('weight', ''), draw=runner.get('draw', ''))
            rows.append(row)
    return rows


def append_canonical(path, frame, columns):
    import pandas as pd
    try:
        previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    except pd.errors.EmptyDataError:
        previous = pd.DataFrame(columns=columns)
    merged = pd.concat([previous, frame], ignore_index=True)
    merged = merged.drop_duplicates(subset=['date', 'track_code', 'event_type', 'race_number',
                                             'runner_number', 'runner_name'], keep='last')
    merged.to_csv(path, index=False)
