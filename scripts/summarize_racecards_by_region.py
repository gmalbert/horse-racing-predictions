#!/usr/bin/env python3
"""Summarize races and participants by region from a racecards JSON file.

Usage:
  python scripts/summarize_racecards_by_region.py data/raw/2026-01-03.json

The script is deliberately tolerant of a few common JSON shapes.
"""
import json
import argparse
from pathlib import Path
import sys


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_cards(obj):
    if isinstance(obj, dict):
        if "racecards" in obj and isinstance(obj["racecards"], list):
            return obj["racecards"]
        # try to find the first list-of-dicts value
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return []
    if isinstance(obj, list):
        return obj
    return []


def get_runners(rc):
    for key in ("runners", "horses", "entries", "participants", "fields"):
        val = rc.get(key)
        if isinstance(val, list):
            return val
    # fall back to count fields if present
    for key in ("runners_count", "num_runners", "field_size"):
        if key in rc and isinstance(rc[key], int):
            return list(range(rc[key]))
    return []


def summarize(path: Path):
    data = load_json(path)
    cards = iter_cards(data)
    counts = {}
    total_races = 0
    total_participants = 0
    for rc in cards:
        region = rc.get("region") or rc.get("course_region") or rc.get("country") or rc.get("course") or "unknown"
        runners = get_runners(rc)
        n = len(runners)
        counts.setdefault(region, {"races": 0, "participants": 0})
        counts[region]["races"] += 1
        counts[region]["participants"] += n
        total_races += 1
        total_participants += n

    for region in sorted(counts):
        v = counts[region]
        print(f"{region}: {v['races']} races, {v['participants']} participants")
    print(f"TOTAL: {total_races} races, {total_participants} participants")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", default="data/raw/2026-01-03.json")
    args = p.parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        summarize(path)
    except Exception as e:
        print("Error processing file:", e, file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
