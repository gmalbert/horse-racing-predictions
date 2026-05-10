"""
Extract text and schedule hints from downloaded T1-TB PDF artifacts.

Step 2 of the T1-TB direct-source pipeline.

Input:
- data/raw/us_t1_tb/tracksite/artifacts/*_<date>_*.pdf

Output:
- data/raw/us_t1_tb/tracksite/pdf_extracted_<date>.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_tb" / "tracksite" / "artifacts"
TRACKSITE_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_tb" / "tracksite"


def _parse_pdf_text(text: str) -> dict:
    cleaned = text or ""
    lower = cleaned.lower()

    post_times = sorted(
        set(
            m.group(1).upper().replace(" ", "")
            for m in re.finditer(r"post\s*time[:\s]+([0-9]{1,2}:[0-9]{2}\s*[AP]M)", cleaned, flags=re.IGNORECASE)
        )
    )

    explicit_dates = sorted(
        set(
            m.group(0)
            for m in re.finditer(
                r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?\b",
                cleaned,
                flags=re.IGNORECASE,
            )
        )
    )

    title = None
    m_title = re.search(r"(\d{4}\s+LIVE\s+RACING\s+SCHEDULE)", cleaned, flags=re.IGNORECASE)
    if m_title:
        title = m_title.group(1).upper()

    needs_ocr = len(cleaned.strip()) < 30

    return {
        "title": title,
        "post_times": post_times,
        "explicit_dates": explicit_dates[:40],
        "mentions_entries": "entries" in lower,
        "mentions_results": "results" in lower,
        "needs_ocr": needs_ocr,
    }


def extract_for_date(date_str: str) -> Path:
    files = sorted(ARTIFACT_DIR.glob(f"*_{date_str}_*.pdf"))
    out_items = []

    for file_path in files:
        track_code = file_path.name.split("_")[0]
        text_parts = []
        page_count = 0

        try:
            with pdfplumber.open(str(file_path)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages[:10]:
                    text_parts.append(page.extract_text() or "")
            text = "\n".join(text_parts)
            parsed = _parse_pdf_text(text)

            out_items.append(
                {
                    "track_code": track_code,
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "bytes": file_path.stat().st_size,
                    "page_count": page_count,
                    "extracted_text_chars": len(text),
                    "text_sample": text[:4000],
                    **parsed,
                }
            )
        except Exception as exc:  # noqa: BLE001
            out_items.append(
                {
                    "track_code": track_code,
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "error": str(exc),
                }
            )

    payload = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": out_items,
    }

    out_path = TRACKSITE_DIR / f"pdf_extracted_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text/metadata from T1-TB PDF artifacts")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    args = parser.parse_args()

    out_path = extract_for_date(args.date)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
