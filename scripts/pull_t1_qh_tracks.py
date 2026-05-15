"""
Pull Tier 1 Quarter Horse track data directly from track websites.

No Equibase dependency. The script visits official Quarter Horse track sites with Playwright,
tries common entries/results URLs, captures raw HTML/text snapshots, and
extracts race-like sections from page text.

Outputs:
- data/raw/us_t1_qh/tracksite/<track_code>_<date>.json
- data/raw/us_t1_qh/tracksite/html/<track_code>_<date>.html
- data/raw/us_t1_qh/t1_qh_tracksite_pull_report_<date>.json

Usage:
    python scripts/pull_t1_qh_tracks.py --date 2026-05-09
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from playwright.async_api import async_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_qh"
TRACKSITE_DIR = OUT_DIR / "tracksite"
HTML_DIR = TRACKSITE_DIR / "html"
ARTIFACT_DIR = TRACKSITE_DIR / "artifacts"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TRACKS = {
    "LA": {
        "name": "Los Alamitos",
        "base": "https://www.losalamitosonline.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "RUD": {
        "name": "Ruidoso Downs",
        "base": "https://www.ruidosodowns.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "DLD": {
        "name": "Delta Downs",
        "base": "https://www.deltadowns.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "EVD": {
        "name": "Evangeline Downs",
        "base": "https://www.evangelinedowns.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "ZIA": {
        "name": "Zia Park",
        "base": "https://www.ziaparkracing.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "SUN": {
        "name": "Sunland Park",
        "base": "https://www.sunland.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
}


def _extract_race_sections(page_text: str, max_sections: int = 12) -> list[dict]:
    normalized = re.sub(r"\r", "", page_text)
    chunks = re.split(r"(?=\bRace\s+\d{1,2}\b)", normalized, flags=re.IGNORECASE)
    out: list[dict] = []

    for chunk in chunks:
        if not re.search(r"\bRace\s+\d{1,2}\b", chunk, flags=re.IGNORECASE):
            continue
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue

        header = lines[0][:200]
        snippet = "\n".join(lines[:10])[:2000]
        out.append({"header": header, "snippet": snippet})
        if len(out) >= max_sections:
            break

    return out


def _quality_score(title: str, text: str) -> int:
    score = 0
    lower = f"{title}\n{text[:20000]}".lower()
    if "entry" in lower or "entries" in lower:
        score += 2
    if "race 1" in lower:
        score += 4
    if "post time" in lower:
        score += 1
    if "results" in lower:
        score += 1
    if re.search(r"\brace\s+\d+\b", lower):
        score += 2
    if "cloudflare" in lower or "access denied" in lower or "forbidden" in lower:
        score -= 3
    return score


def _extract_summary_fields(text: str) -> dict:
    lower = text.lower()

    first_race = None
    m_first = re.search(r"first race\s*[:\-]?\s*([0-9]{1,2}:[0-9]{2}\s*[ap]m?)", lower, flags=re.IGNORECASE)
    if m_first:
        first_race = m_first.group(1).replace(" ", "")

    race_day = None
    m_day = re.search(
        r"next race day\s*[:\-]?\s*([a-z]+,\s+[a-z]+\s+\d{1,2})",
        text,
        flags=re.IGNORECASE,
    )
    if m_day:
        race_day = m_day.group(1).strip()

    race_count = None
    m_count = re.search(r"\b(\d{1,2})\s+races\b", lower)
    if m_count:
        race_count = int(m_count.group(1))

    has_entries = "entries" in lower
    has_results = "results" in lower

    return {
        "first_race_post": first_race,
        "next_race_day": race_day,
        "race_count_hint": race_count,
        "has_entries_section": has_entries,
        "has_results_section": has_results,
    }


def _download_artifacts(page_text: str, base_url: str, track_code: str, date_str: str) -> list[dict]:
    """Download PDF/CSV/XML artifacts linked from the page."""
    artifacts = []
    link_pattern = r'href=["\']((?:https?://|/)[^"\']*\.(?:pdf|csv|xml|ics))'
    matches = re.findall(link_pattern, page_text, flags=re.IGNORECASE)

    for link in matches:
        full_url = urljoin(base_url, link) if link.startswith("/") else link
        try:
            resp = requests.get(full_url, timeout=10, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()

            parsed = urlparse(full_url)
            filename = Path(parsed.path).name
            if not filename:
                filename = f"{track_code}_{date_str}_{len(artifacts)}.bin"

            artifact_path = ARTIFACT_DIR / filename
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(resp.content)

            artifacts.append({
                "url": full_url,
                "filename": filename,
                "size_bytes": len(resp.content),
                "saved_path": str(artifact_path.relative_to(PROJECT_ROOT)),
            })
        except Exception as e:
            print(f"  Failed to download artifact {full_url}: {e}")

    return artifacts


async def _pull_track(page, track_code: str, track_info: dict, date_str: str) -> dict:
    """Pull one track."""
    base = track_info["base"]
    paths = track_info["paths"]
    results = {
        "track_code": track_code,
        "track_name": track_info["name"],
        "date": date_str,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "pages_tried": [],
        "best_page": None,
        "best_quality_score": -999,
    }

    for path in paths:
        url = urljoin(base, path)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(2000)
            html_text = await page.content()
            title = await page.title()
        except Exception as e:
            print(f"  {track_code} @ {path}: {type(e).__name__}")
            results["pages_tried"].append({"path": path, "url": url, "error": str(e)[:100]})
            continue

        quality = _quality_score(title, html_text)
        results["pages_tried"].append({
            "path": path,
            "url": url,
            "title": title[:100],
            "quality_score": quality,
            "char_count": len(html_text),
        })

        if quality > results["best_quality_score"]:
            results["best_quality_score"] = quality
            results["best_page"] = {
                "url": url,
                "title": title[:100],
                "html_length": len(html_text),
                "quality_score": quality,
            }

            # Discover internal links matching entry/result/race patterns
            discovered = set()
            link_pattern = r'href=["\']((?:https?://)?[^"\']*(?:entry|result|race)[^"\']*)'
            for m in re.finditer(link_pattern, html_text, flags=re.IGNORECASE):
                candidate = m.group(1)
                if candidate.startswith("http"):
                    discovered.add(candidate)
                elif candidate.startswith("/"):
                    discovered.add(urljoin(base, candidate))

            # Probe top discovered link
            if discovered:
                probe_url = sorted(discovered, key=lambda x: _quality_score("", ""), reverse=True)[0]
                try:
                    await page.goto(probe_url, wait_until="domcontentloaded", timeout=15000)
                    await page.wait_for_timeout(2000)
                    probe_html = await page.content()
                    probe_title = await page.title()
                    probe_quality = _quality_score(probe_title, probe_html)
                    if probe_quality > results["best_quality_score"]:
                        results["best_quality_score"] = probe_quality
                        results["best_page"] = {
                            "url": probe_url,
                            "title": probe_title[:100],
                            "html_length": len(probe_html),
                            "quality_score": probe_quality,
                        }
                        html_text = probe_html
                except Exception:
                    pass

            # Extract race sections + summary
            race_sections = _extract_race_sections(html_text)
            summary = _extract_summary_fields(html_text)

            # Download artifacts
            artifacts = _download_artifacts(html_text, base, track_code, date_str)

            results["best_page"]["race_sections"] = race_sections[:6]
            results["best_page"]["summary_fields"] = summary
            results["best_page"]["artifacts"] = artifacts

            # Save HTML snapshot
            html_path = HTML_DIR / f"{track_code}_{date_str}.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html_text, encoding="utf-8", errors="replace")

    return results


async def main(date_str: str):
    TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(user_agent=USER_AGENT)

        try:
            pulls = []
            for track_code, track_info in TRACKS.items():
                print(f"Pulling {track_code}...")
                result = await _pull_track(page, track_code, track_info, date_str)
                pulls.append(result)
                print(f"  Best score: {result['best_quality_score']}")

            # Write pull report
            report = {
                "breed": "Quarter Horse",
                "date": date_str,
                "pulled_at": datetime.now(timezone.utc).isoformat(),
                "tracks": pulls,
                "summary": {
                    "total_tracks": len(pulls),
                    "successful_pulls": sum(1 for p in pulls if p["best_quality_score"] >= 0),
                },
            }

            report_path = OUT_DIR / f"t1_qh_tracksite_pull_report_{date_str}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nPull report: {report_path}")

        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Date as YYYY-MM-DD")
    args = parser.parse_args()

    date_arg = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    asyncio.run(main(date_arg))
