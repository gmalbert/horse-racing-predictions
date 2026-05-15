"""
Extract text and metadata from Quarter Horse PDF artifacts.

Usage:
    python scripts/extract_t1_qh_pdf_artifacts.py --date 2026-05-09
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_qh"
TRACKSITE_DIR = OUT_DIR / "tracksite"
ARTIFACT_DIR = TRACKSITE_DIR / "artifacts"


def extract_pdf_metadata(pdf_path: Path) -> dict:
    """Extract text, metadata, and structured fields from a PDF."""
    result = {
        "filename": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "extracted_successfully": False,
        "page_count": 0,
        "extracted_text_chars": 0,
        "needs_ocr": False,
        "text_sample": "",
        "post_times": [],
        "dates": [],
        "title": "",
    }

    if not pdfplumber:
        result["needs_ocr"] = True
        return result

    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["page_count"] = len(pdf.pages)

            full_text = ""
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
                except Exception:
                    pass

            result["extracted_text_chars"] = len(full_text)
            if result["extracted_text_chars"] < 100:
                result["needs_ocr"] = True

            result["text_sample"] = full_text[:500]

            # Extract post times (heuristic: HH:MM AM/PM)
            post_times = re.findall(r"\d{1,2}:\d{2}\s*[ap]m", full_text, flags=re.IGNORECASE)
            result["post_times"] = list(set(post_times))[:5]

            # Extract dates (heuristic: Month DD or similar)
            dates = re.findall(
                r"([A-Z][a-z]+,?\s+[A-Z][a-z]+\s+\d{1,2})",
                full_text,
            )
            result["dates"] = list(set(dates))[:3]

            # Try to extract title from metadata
            if pdf.metadata and pdf.metadata.get("Title"):
                result["title"] = pdf.metadata["Title"][:100]

            result["extracted_successfully"] = True

    except Exception as e:
        result["error"] = str(e)[:100]
        result["needs_ocr"] = True

    return result


def main(date_str: str):
    if not ARTIFACT_DIR.exists():
        print(f"No artifact directory: {ARTIFACT_DIR}")
        return

    pdf_files = list(ARTIFACT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {ARTIFACT_DIR}")
        return

    extracted = []
    for pdf_path in pdf_files:
        print(f"Extracting {pdf_path.name}...")
        metadata = extract_pdf_metadata(pdf_path)
        extracted.append(metadata)

    # Write extraction report
    report = {
        "breed": "Quarter Horse",
        "date": date_str,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "total_pdfs": len(extracted),
        "pdfs": extracted,
    }

    report_path = OUT_DIR / f"pdf_extracted_{date_str}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Extraction report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Date as YYYY-MM-DD")
    args = parser.parse_args()

    date_arg = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    main(date_arg)
