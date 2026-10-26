"""Verify desktop USTA coverage independently of a working mobile feed in CI."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from us_track_collector import _usta_desktop, _parse_usta_desktop
from us_track_sources import ALL_TRACKS


def check_track(item, date_str):
    code, expected = item
    cfg = ALL_TRACKS[code]
    response = _usta_desktop(cfg, date_str)
    races = _parse_usta_desktop(response['html'], response['url'], code, cfg, date_str)
    actual = {r['race_number'] for r in races}
    required = expected | set(response.get('expected_races', []))
    return {'track': code, 'status': 'PASS' if response['status'] == 'ok' and required <= actual else 'FAIL',
            'expected_races': sorted(required), 'collected_races': sorted(actual),
            'runners': sum(len(r['runners']) for r in races), 'attempts': response['attempts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    datetime.strptime(args.date, '%Y-%m-%d')
    raw = Path('data/raw')
    cards = json.loads((raw / f'us_racecards_{args.date}.json').read_text(encoding='utf-8'))
    if cards.get('date') != args.date or cards.get('fallback_used'):
        raise SystemExit('Desktop verification requires fresh cards for the requested date')
    expected = {}
    for race in cards.get('racecards', []):
        if race.get('type') == 'Harness' and race.get('runners'):
            expected.setdefault(race['track'], set()).add(race['race_number'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        tracks = list(pool.map(lambda item: check_track(item, args.date), expected.items()))
    report = {'date': args.date, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PASS' if all(t['status'] == 'PASS' for t in tracks) else 'FAIL', 'tracks': tracks}
    path = raw / f'us_harness_fallback_check_{args.date}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for track in tracks:
        print(f"{track['track']}: desktop {track['status']}, {len(track['collected_races'])} races, {track['runners']} runners")
    print(f"Desktop fallback verification: {report['status']} ({len(tracks)} tracks); {path}")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
