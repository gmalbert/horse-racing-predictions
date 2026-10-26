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
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONUTF8': '1'},
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

    def breed_steps(group):
        pull = _run([py, f"scripts/pull_t1_{group}_tracks.py", "--date", date_str])
        extract = _run([py, f"scripts/extract_t1_{group}_pdf_artifacts.py", "--date", date_str])
        parse = _run([py, f"scripts/parse_t1_{group}_tracksite.py", "--date", date_str, "--append"])
        return pull, extract, parse

    # Breeds write distinct paths. Run them concurrently so one site's timeout
    # does not exhaust the all-sources parent's 420 second budget.
    with ThreadPoolExecutor(max_workers=3) as pool:
        tb_future = pool.submit(breed_steps, "tb")
        qh_future = pool.submit(breed_steps, "qh")
        h_future = pool.submit(breed_steps, "h")
        tb_step1, tb_step2, tb_step3 = tb_future.result()
        qh_step1, qh_step2, qh_step3 = qh_future.result()
        h_step1, h_step2, h_step3 = h_future.result()
    tb_ok = all(step["returncode"] == 0 for step in [tb_step1, tb_step2, tb_step3])
    qh_ok = all(step["returncode"] == 0 for step in [qh_step1, qh_step2, qh_step3])
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
    if not all(r['overall_ok'] for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
