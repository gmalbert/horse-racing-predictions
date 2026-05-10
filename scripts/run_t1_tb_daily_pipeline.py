"""
Run the T1-TB direct-source daily pipeline.

Step 3 of the T1-TB direct-source pipeline.
Order executed:
1) pull_t1_tb_tracks.py
2) extract_t1_tb_pdf_artifacts.py
3) parse_t1_tb_tracksite.py --append

Outputs:
- data/raw/us_t1_tb/t1_tb_pipeline_report_<date>.json
- data/processed/us_t1_tb_canonical_<date>.csv
- data/processed/us_t1_tb_canonical_all.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_tb"


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

    step1 = _run([py, "scripts/pull_t1_tb_tracks.py", "--date", date_str])
    step2 = _run([py, "scripts/extract_t1_tb_pdf_artifacts.py", "--date", date_str])
    step3 = _run([py, "scripts/parse_t1_tb_tracksite.py", "--date", date_str, "--append"])

    ok = all(step["returncode"] == 0 for step in [step1, step2, step3])
    return {
        "date": date_str,
        "ok": ok,
        "steps": {
            "pull_track_sites": step1,
            "extract_pdf_artifacts": step2,
            "parse_to_canonical": step3,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1-TB direct-source daily pipeline")
    parser.add_argument("--date", type=str, help="Start date YYYY-MM-DD (default=today)")
    parser.add_argument("--days", type=int, default=1, help="Number of sequential days to run")
    args = parser.parse_args()

    dates = _dates_from_args(args.date, args.days)
    runs = [run_for_date(d) for d in dates]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs,
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"t1_tb_pipeline_report_{dates[0]}"
    if len(dates) > 1:
        out_name += f"_to_{dates[-1]}"
    out_path = RAW_DIR / f"{out_name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved pipeline report: {out_path}")
    for r in runs:
        print(f"{r['date']}: {'OK' if r['ok'] else 'FAILED'}")


if __name__ == "__main__":
    main()
