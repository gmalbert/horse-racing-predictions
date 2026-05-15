"""
Run the master Tier 1 daily pipeline for all three breeds.

Orchestrates:
1) Thoroughbred: pull_t1_tb_tracks.py → extract_t1_tb_pdf_artifacts.py → parse_t1_tb_tracksite.py
2) Quarter Horse: pull_t1_qh_tracks.py → extract_t1_qh_pdf_artifacts.py → parse_t1_qh_tracksite.py
3) Harness: pull_t1_h_tracks.py → extract_t1_h_pdf_artifacts.py → parse_t1_h_tracksite.py

Outputs:
- data/raw/us_t1_tb/t1_tb_pipeline_report_<date>.json
- data/raw/us_t1_qh/t1_qh_pipeline_report_<date>.json
- data/raw/us_t1_h/t1_h_pipeline_report_<date>.json
- data/processed/ canonical CSVs and JSONs for all three breeds
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-40:]),
    }


def _dates_from_args(date_arg: str | None, days: int) -> list[str]:
    if date_arg:
        start = datetime.strptime(date_arg, "%Y-%m-%d")
    else:
        start = datetime.now()
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def run_for_date(date_str: str) -> dict:
    py = sys.executable

    # Thoroughbred
    tb_step1 = _run([py, "scripts/pull_t1_tb_tracks.py", "--date", date_str])
    tb_step2 = _run([py, "scripts/extract_t1_tb_pdf_artifacts.py", "--date", date_str])
    tb_step3 = _run([py, "scripts/parse_t1_tb_tracksite.py", "--date", date_str, "--append"])
    tb_ok = all(step["returncode"] == 0 for step in [tb_step1, tb_step2, tb_step3])

    # Quarter Horse
    qh_step1 = _run([py, "scripts/pull_t1_qh_tracks.py", "--date", date_str])
    qh_step2 = _run([py, "scripts/extract_t1_qh_pdf_artifacts.py", "--date", date_str])
    qh_step3 = _run([py, "scripts/parse_t1_qh_tracksite.py", "--date", date_str, "--append"])
    qh_ok = all(step["returncode"] == 0 for step in [qh_step1, qh_step2, qh_step3])

    # Harness
    h_step1 = _run([py, "scripts/pull_t1_h_tracks.py", "--date", date_str])
    h_step2 = _run([py, "scripts/extract_t1_h_pdf_artifacts.py", "--date", date_str])
    h_step3 = _run([py, "scripts/parse_t1_h_tracksite.py", "--date", date_str, "--append"])
    h_ok = all(step["returncode"] == 0 for step in [h_step1, h_step2, h_step3])

    return {
        "date": date_str,
        "thoroughbred": {
            "ok": tb_ok,
            "steps": {
                "pull_track_sites": tb_step1,
                "extract_pdf_artifacts": tb_step2,
                "parse_to_canonical": tb_step3,
            },
        },
        "quarter_horse": {
            "ok": qh_ok,
            "steps": {
                "pull_track_sites": qh_step1,
                "extract_pdf_artifacts": qh_step2,
                "parse_to_canonical": qh_step3,
            },
        },
        "harness": {
            "ok": h_ok,
            "steps": {
                "pull_track_sites": h_step1,
                "extract_pdf_artifacts": h_step2,
                "parse_to_canonical": h_step3,
            },
        },
        "overall_ok": tb_ok and qh_ok and h_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run master Tier 1 daily pipeline (all breeds)")
    parser.add_argument("--date", type=str, help="Start date YYYY-MM-DD (default=today)")
    parser.add_argument("--days", type=int, default=1, help="Number of sequential days to run")
    args = parser.parse_args()

    dates = _dates_from_args(args.date, args.days)
    runs = [run_for_date(d) for d in dates]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs,
    }

    out_dir = PROJECT_ROOT / "data" / "raw" / "us_t1_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"t1_all_pipeline_report_{dates[0]}"
    if len(dates) > 1:
        out_name += f"_to_{dates[-1]}"
    out_path = out_dir / f"{out_name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved pipeline report: {out_path}")
    for r in runs:
        breeds_status = (
            f"TB={r['thoroughbred']['ok']} "
            f"QH={r['quarter_horse']['ok']} "
            f"H={r['harness']['ok']}"
        )
        overall = "OK" if r["overall_ok"] else "FAILED"
        print(f"{r['date']}: {overall} ({breeds_status})")


if __name__ == "__main__":
    main()
