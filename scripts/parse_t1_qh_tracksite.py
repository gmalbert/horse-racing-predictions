"""
Parse direct track-site pull outputs into canonical race/runner rows for Quarter Horse.

Step 3 of the T1-QH direct-source pipeline.

Inputs:
- data/raw/us_t1_qh/tracksite/<TRACK>_<date>.json
- optional: data/raw/us_t1_qh/tracksite/pdf_extracted_<date>.json

Outputs:
- data/processed/us_t1_qh_canonical_<date>.csv
- data/processed/us_t1_qh_canonical_<date>.json
- optional append: data/processed/us_t1_qh_canonical_all.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKSITE_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_qh" / "tracksite"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_TRACKS = ["LA", "RUD", "DLD", "EVD", "ZIA", "SUN"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_row(date_str: str, code: str, name: str, selected_url: str) -> dict:
    return {
        "date": date_str,
        "track_code": code,
        "track_name": name,
        "breed": "Quarter Horse",
        "source": "track_sites_playwright",
        "source_url": selected_url,
        "event_type": None,
        "race_number": None,
        "race_name": None,
        "race_time": None,
        "distance": None,
        "surface": None,
        "track_condition": None,
        "purse": None,
        "runner_number": None,
        "runner_name": None,
        "jockey": None,
        "win_payoff": None,
        "place_payoff": None,
        "show_payoff": None,
        "notes": None,
    }


def _parse_summary_event(track_json: dict, pdf_info: dict | None) -> list[dict]:
    """Parse entries/summary page into a single summary row with metadata."""
    date_str = track_json.get("date")
    code = track_json.get("track_code")
    name = track_json.get("track_name")
    best_page = track_json.get("best_page", {})
    src = best_page.get("url")
    summary = best_page.get("summary_fields", {})

    row = _base_row(date_str, code, name, src)
    row["event_type"] = "entries_summary"

    first_race_time = summary.get("first_race_post")
    if first_race_time:
        row["race_time"] = first_race_time

    race_count = summary.get("race_count_hint")
    next_day = summary.get("next_race_day")

    notes = []
    if next_day:
        notes.append(f"next_race_day={next_day}")
    if race_count is not None:
        notes.append(f"race_count_hint={race_count}")
    if summary.get("has_entries_section"):
        notes.append("has_entries_section=true")
    if summary.get("has_results_section"):
        notes.append("has_results_section=true")

    if pdf_info:
        post_times = pdf_info.get("post_times") or []
        if post_times and not row["race_time"]:
            row["race_time"] = post_times[0]
        if post_times:
            notes.append("pdf_post_times=" + ",".join(post_times[:4]))
        title = pdf_info.get("title")
        if title:
            notes.append(f"pdf_title={title}")
        if pdf_info.get("needs_ocr"):
            notes.append("pdf_needs_ocr=true")

    row["notes"] = "; ".join(notes) if notes else None
    return [row]


def parse_date(date_str: str, append: bool = False) -> tuple[Path, Path, int]:
    pdf_path = TRACKSITE_DIR / f"pdf_extracted_{date_str}.json"
    pdf_data = _load_json(pdf_path) if pdf_path.exists() else {"pdfs": []}

    pdf_by_track = {}
    for item in pdf_data.get("pdfs", []):
        # Try to infer track code from filename
        filename = item.get("filename", "").lower()
        track_code = None
        for tc in TARGET_TRACKS:
            if tc.lower() in filename:
                track_code = tc
                break

        if not track_code:
            continue

        existing = pdf_by_track.get(track_code)
        if existing is None:
            pdf_by_track[track_code] = item
            continue

        def _score(pdf_item: dict) -> int:
            if not pdf_item.get("extracted_successfully"):
                return -1
            return int(pdf_item.get("extracted_text_chars") or 0)

        if _score(item) > _score(existing):
            pdf_by_track[track_code] = item

    all_rows: list[dict] = []

    for code in TARGET_TRACKS:
        file_path = TRACKSITE_DIR / f"{code}_{date_str}.json"
        if not file_path.exists():
            continue

        track_json = _load_json(file_path)
        all_rows.extend(_parse_summary_event(track_json, pdf_by_track.get(code)))

    df = pd.DataFrame(all_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_out = PROCESSED_DIR / f"us_t1_qh_canonical_{date_str}.csv"
    json_out = PROCESSED_DIR / f"us_t1_qh_canonical_{date_str}.json"

    df.to_csv(csv_out, index=False)
    json_out.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    if append:
        all_csv = PROCESSED_DIR / "us_t1_qh_canonical_all.csv"
        if all_csv.exists():
            prev = pd.read_csv(all_csv)
            merged = pd.concat([prev, df], ignore_index=True)
            merged = merged.drop_duplicates(
                subset=["date", "track_code", "event_type", "race_number"],
                keep="last",
            )
        else:
            merged = df
        merged.to_csv(all_csv, index=False)

    return csv_out, json_out, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse T1-QH track-site pulls into canonical rows")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--append", action="store_true", help="Append into data/processed/us_t1_qh_canonical_all.csv")
    args = parser.parse_args()

    csv_out, json_out, row_count = parse_date(args.date, append=args.append)
    print(f"Saved CSV: {csv_out}")
    print(f"Saved JSON: {json_out}")
    print(f"Rows: {row_count}")


if __name__ == "__main__":
    main()
