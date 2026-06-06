#!/usr/bin/env python3
"""Validate daily US ingestion health and detect coverage drift.

This script is designed for CI/workflow guardrails. It checks that:
- core US racecard/source-report artifacts exist
- race/runner volumes are above minimum thresholds
- ML odds coverage is above minimum
- today's race volume is not drifting too far below recent history

Outputs:
- data/processed/us_ingestion_validation_<date>.json
- data/processed/us_ingestion_validation_<date>.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _summarize_racecards(payload: dict) -> dict:
    racecards = payload.get("racecards") or []
    total_races = len(racecards)
    total_runners = 0
    runners_with_ml = 0
    tracks = set()

    for race in racecards:
        track = (race.get("track") or race.get("course") or "").strip()
        if track:
            tracks.add(track)
        runners = race.get("runners") or []
        total_runners += len(runners)
        for r in runners:
            ml = (r.get("ml_odds") or r.get("odds") or "").strip()
            if ml and ml not in {"-", "N/A"}:
                runners_with_ml += 1

    ml_coverage = (runners_with_ml / total_runners) if total_runners else 0.0
    return {
        "total_races": total_races,
        "total_runners": total_runners,
        "unique_tracks": len(tracks),
        "runners_with_ml_odds": runners_with_ml,
        "ml_odds_coverage": ml_coverage,
    }


def _collect_recent_totals(current_date: str, lookback_days: int) -> list[int]:
    totals = []
    for path in sorted(RAW_DIR.glob("us_source_report_*.json")):
        if current_date in path.name:
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        source_counts = payload.get("source_counts") or {}
        total_races = _safe_int(source_counts.get("total_races"), 0)
        if total_races > 0:
            totals.append(total_races)
    return totals[-lookback_days:]


def _to_markdown(result: dict) -> str:
    checks = result["checks"]
    lines = [
        "# US Ingestion Validation Report",
        "",
        f"- Date: {result['date']}",
        f"- Status: **{result['status']}**",
        f"- Generated At (UTC): {result['generated_at_utc']}",
        "",
        "## Metrics",
        "",
        f"- Total races: {result['metrics']['total_races']}",
        f"- Total runners: {result['metrics']['total_runners']}",
        f"- Unique tracks: {result['metrics']['unique_tracks']}",
        f"- ML odds coverage: {result['metrics']['ml_odds_coverage']:.1%}",
        "",
        "## Checks",
        "",
    ]

    for chk in checks:
        icon = "PASS" if chk["ok"] else "FAIL"
        lines.append(f"- [{icon}] {chk['name']}: {chk['message']}")

    if result.get("alerts"):
        lines.extend(["", "## Alerts", ""])
        for alert in result["alerts"]:
            lines.append(f"- {alert}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate US ingestion health")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--min-total-races", type=int, default=8)
    parser.add_argument("--min-total-runners", type=int, default=50)
    parser.add_argument("--min-ml-odds-coverage", type=float, default=0.50)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--min-drift-ratio", type=float, default=0.50)
    args = parser.parse_args()

    date_str = args.date
    racecards_path = RAW_DIR / f"us_racecards_{date_str}.json"
    source_report_path = RAW_DIR / f"us_source_report_{date_str}.json"

    checks = []
    alerts = []

    racecards_exists = racecards_path.exists()
    source_report_exists = source_report_path.exists()

    checks.append(
        {
            "name": "racecards_file_exists",
            "ok": racecards_exists,
            "message": str(racecards_path),
        }
    )
    checks.append(
        {
            "name": "source_report_exists",
            "ok": source_report_exists,
            "message": str(source_report_path),
        }
    )

    metrics = {
        "total_races": 0,
        "total_runners": 0,
        "unique_tracks": 0,
        "runners_with_ml_odds": 0,
        "ml_odds_coverage": 0.0,
        "historical_median_total_races": 0.0,
    }

    if racecards_exists:
        payload = _read_json(racecards_path)
        metrics.update(_summarize_racecards(payload))

    checks.append(
        {
            "name": "min_total_races",
            "ok": metrics["total_races"] >= args.min_total_races,
            "message": f"{metrics['total_races']} >= {args.min_total_races}",
        }
    )
    checks.append(
        {
            "name": "min_total_runners",
            "ok": metrics["total_runners"] >= args.min_total_runners,
            "message": f"{metrics['total_runners']} >= {args.min_total_runners}",
        }
    )
    checks.append(
        {
            "name": "min_ml_odds_coverage",
            "ok": metrics["ml_odds_coverage"] >= args.min_ml_odds_coverage,
            "message": (
                f"{metrics['ml_odds_coverage']:.1%} >= {args.min_ml_odds_coverage:.1%}"
            ),
        }
    )

    historical_totals = _collect_recent_totals(date_str, args.lookback_days)
    if historical_totals:
        median_total = float(statistics.median(historical_totals))
        metrics["historical_median_total_races"] = median_total
        min_allowed = median_total * args.min_drift_ratio
        drift_ok = metrics["total_races"] >= min_allowed
        checks.append(
            {
                "name": "coverage_drift_guard",
                "ok": drift_ok,
                "message": (
                    f"today={metrics['total_races']} vs median={median_total:.1f}, "
                    f"min_allowed={min_allowed:.1f}"
                ),
            }
        )
        if not drift_ok:
            alerts.append("Coverage drift detected: race volume significantly below recent baseline.")
    else:
        checks.append(
            {
                "name": "coverage_drift_guard",
                "ok": True,
                "message": "Insufficient history for drift baseline; check skipped",
            }
        )

    # Cross-check source report totals if available.
    if source_report_exists:
        report = _read_json(source_report_path)
        src_counts = report.get("source_counts") or {}
        report_total = _safe_int(src_counts.get("total_races"), 0)
        checks.append(
            {
                "name": "source_report_total_matches_racecards",
                "ok": report_total == metrics["total_races"],
                "message": f"report_total={report_total}, racecards_total={metrics['total_races']}",
            }
        )

    status = "PASS" if all(c["ok"] for c in checks) else "FAIL"

    result = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "metrics": metrics,
        "alerts": alerts,
    }

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_json = PROC_DIR / f"us_ingestion_validation_{date_str}.json"
    out_md = PROC_DIR / f"us_ingestion_validation_{date_str}.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(result), encoding="utf-8")

    print(f"Validation status: {status}")
    print(f"Saved JSON report: {out_json}")
    print(f"Saved Markdown report: {out_md}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
