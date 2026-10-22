"""Publish only collected cards from an isolated checkout of an explicit branch."""
import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo


def git(*args, cwd):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def publish(source, branch, dates):
    source = Path(source).resolve()
    cards = []
    for day in dates:
        path = source / 'data' / 'raw' / f'racecards_{day}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        races = [race for courses in payload.values() for meeting in courses.values()
                 for race in meeting.values()]
        if not races or not all(r.get('date') == day for r in races) or not any(r.get('runners') for r in races):
            raise ValueError(f'Invalid or empty racecards: {path}')
        cards.append(path)
    remote = git('remote', 'get-url', 'origin', cwd=source)
    with tempfile.TemporaryDirectory(prefix='hrp-publish-') as folder:
        checkout = Path(folder) / 'repo'
        git('clone', '--single-branch', '--branch', branch, remote, str(checkout), cwd=source)
        for key in ['user.name', 'user.email']:
            git('config', key, git('config', key, cwd=source), cwd=checkout)
        for path in cards:
            dest = checkout / 'data' / 'raw' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            git('add', '--', str(dest.relative_to(checkout)), cwd=checkout)
        if git('diff', '--cached', '--name-only', cwd=checkout):
            git('commit', '-m', 'chore: refresh Racing Post browser racecards [skip ci]',
                '-m', '## Data refresh\n- Publish validated browser cards for ' + ', '.join(dates)
                + '.\n- Publish only collected snapshots from an isolated checkout; preserve development work.',
                cwd=checkout)
        # Explicit target works even when the source development branch has no upstream.
        git('push', 'origin', f'HEAD:refs/heads/{branch}', cwd=checkout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--branch', default='main')
    parser.add_argument('--date')
    parser.add_argument('--days', type=int, default=2)
    parser.add_argument('--timezone', default='America/New_York')
    args = parser.parse_args()
    start = date.fromisoformat(args.date) if args.date else datetime.now(ZoneInfo(args.timezone)).date()
    dates = [(start + timedelta(days=i)).isoformat() for i in range(1 if args.date else args.days)]
    publish(args.source, args.branch, dates)


if __name__ == '__main__':
    main()
