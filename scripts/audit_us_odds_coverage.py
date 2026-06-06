#!/usr/bin/env python3
"""Audit US odds source coverage and recommend fallback hierarchy.

Creates a daily audit report that summarizes odds availability across:
- US racecards (morning line / ml_odds)
- NYRA entries
- OddsPortal graded-stakes feed

Outputs:
- data/processed/us_odds_coverage_audit_<date>.json
- data/processed/us_odds_coverage_audit_<date>.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

FALLBACK_HIERARCHY = [
    "1) us_racecards_<date>.json (TVG/graphql ml_odds)",
    "2) nyra_entries_<date>.json (NYRA ml_odds where available)",
    "3) oddsportal_us_<date>.json (major stakes only)",
    "4) last-known odds snapshot from previous day (stale flag required)",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_racecard_ml_coverage(payload: dict) -> dict:
    races = payload.get("racecards") or []
    total_runners = 0
    runners_with_ml = 0
    for race in races:
        for runner in race.get("runners") or []:
            total_runners += 1
            ml = (runner.get("ml_odds") or runner.get("odds") or "").strip()
            if ml and ml not in {"-", "N/A"}:
                runners_with_ml += 1
    coverage = (runners_with_ml / total_runners) if total_runners else 0.0
    return {
        "races": len(races),
        "total_runners": total_runners,
        "runners_with_odds": runners_with_ml,
        "coverage": coverage,
    }


def _count_nyra_ml_coverage(payload: dict) -> dict:
    tracks = payload.get("tracks") or {}
    races = 0
    total_runners = 0
    runners_with_ml = 0

    if isinstance(tracks, dict):
        for race_list in tracks.values():
            for race in race_list or []:
                races += 1
                for runner in race.get("runners") or []:
                    total_runners += 1
                    ml = (runner.get("ml_odds") or runner.get("odds") or "").strip()
                    if ml and ml not in {"-", "N/A"}:
                        runners_with_ml += 1

    coverage = (runners_with_ml / total_runners) if total_runners else 0.0
    return {
        "races": races,
        "total_runners": total_runners,
        "runners_with_odds": runners_with_ml,
        "coverage": coverage,
    }


def _count_oddsportal(payload: dict) -> dict:
    races = payload.get("races") or []
    total_runners = 0
    runners_with_odds = 0
    bookmakers = set()

    for race in races:
        for bm in race.get("bookmakers") or []:
            if bm:
                bookmakers.add(str(bm))
        for runner in race.get("runners") or []:
            total_runners += 1
            odds = runner.get("odds") or {}
            has_any = any(v is not None for v in odds.values()) if isinstance(odds, dict) else False
            if has_any:
                runners_with_odds += 1

    coverage = (runners_with_odds / total_runners) if total_runners else 0.0
    return {
        "races": len(races),
        "total_runners": total_runners,
        "runners_with_odds": runners_with_odds,
        "coverage": coverage,
        "bookmakers": sorted(bookmakers),
    }


def _to_md(date_str: str, summary: dict) -> str:
    lines = [
        "# US Odds Coverage Audit",
        "",
        f"- Date: {date_str}",
        f"- Generated At (UTC): {summary['generated_at_utc']}",
        "",
        "## Source Coverage",
        "",
    ]

    for src_name in ["us_racecards", "nyra_entries", "oddsportal"]:
        src = summary["sources"][src_name]
        lines.append(f"- {src_name}: available={src['available']}, races={src['races']}, coverage={src['coverage']:.1%}")

    lines.extend(["", "## Fallback Hierarchy", ""])
    lines.extend([f"- {item}" for item in FALLBACK_HIERARCHY])

    if summary.get("alerts"):
        lines.extend(["", "## Alerts", ""])
        for alert in summary["alerts"]:
            lines.append(f"- {alert}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit US odds coverage")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--min-primary-coverage", type=float, default=0.50)
    args = parser.parse_args()

    date_str = args.date
    sources = {
        "us_racecards": {
            "path": RAW_DIR / f"us_racecards_{date_str}.json",
            "available": False,
            "races": 0,
            "coverage": 0.0,
        },
        "nyra_entries": {
            "path": RAW_DIR / f"nyra_entries_{date_str}.json",
            "available": False,
            "races": 0,
            "coverage": 0.0,
        },
        "oddsportal": {
            "path": RAW_DIR / f"oddsportal_us_{date_str}.json",
            "available": False,
            "races": 0,
            "coverage": 0.0,
        },
    }

    alerts = []

    rc_path = sources["us_racecards"]["path"]
    if rc_path.exists():
        rc_data = _read_json(rc_path)
        rc_cov = _count_racecard_ml_coverage(rc_data)
        sources["us_racecards"].update({
            "available": True,
            "races": rc_cov["races"],
            "coverage": rc_cov["coverage"],
            "total_runners": rc_cov["total_runners"],
            "runners_with_odds": rc_cov["runners_with_odds"],
        })

    nyra_path = sources["nyra_entries"]["path"]
    if nyra_path.exists():
        nyra_data = _read_json(nyra_path)
        nyra_cov = _count_nyra_ml_coverage(nyra_data)
        sources["nyra_entries"].update({
            "available": True,
            "races": nyra_cov["races"],
            "coverage": nyra_cov["coverage"],
            "total_runners": nyra_cov["total_runners"],
            "runners_with_odds": nyra_cov["runners_with_odds"],
        })

    op_path = sources["oddsportal"]["path"]
    if op_path.exists():
        op_data = _read_json(op_path)
        op_cov = _count_oddsportal(op_data)
        sources["oddsportal"].update({
            "available": True,
            "races": op_cov["races"],
            "coverage": op_cov["coverage"],
            "total_runners": op_cov["total_runners"],
            "runners_with_odds": op_cov["runners_with_odds"],
            "bookmakers": op_cov["bookmakers"],
        })

    primary = sources["us_racecards"]
    if not primary["available"]:
        alerts.append("Primary source missing: us_racecards file not found.")
    elif primary["coverage"] < args.min_primary_coverage:
        alerts.append(
            f"Primary source coverage low: {primary['coverage']:.1%} < {args.min_primary_coverage:.1%}."
        )

    if not sources["nyra_entries"]["available"] and not sources["oddsportal"]["available"]:
        alerts.append("No fallback odds sources available (NYRA + OddsPortal both missing).")

    summary = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: {
                **payload,
                "path": str(payload["path"]),
            }
            for name, payload in sources.items()
        },
        "fallback_hierarchy": FALLBACK_HIERARCHY,
        "alerts": alerts,
        "status": "PASS" if not alerts else "WARN",
    }

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_json = PROC_DIR / f"us_odds_coverage_audit_{date_str}.json"
    out_md = PROC_DIR / f"us_odds_coverage_audit_{date_str}.md"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_to_md(date_str, summary), encoding="utf-8")

    print(f"Saved JSON audit: {out_json}")
    print(f"Saved Markdown audit: {out_md}")
    print(f"Status: {summary['status']}")

    # Warning only; does not fail pipeline by default.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
