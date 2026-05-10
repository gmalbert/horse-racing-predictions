"""
Pull Tier 1 Thoroughbred track data directly from track websites (NYRA-style).

No Equibase dependency. The script visits official track sites with Playwright,
tries common entries/results URLs, captures raw HTML/text snapshots, and
extracts race-like sections from page text.

Outputs:
- data/raw/us_t1_tb/tracksite/<track_code>_<date>.json
- data/raw/us_t1_tb/tracksite/html/<track_code>_<date>.html
- data/raw/us_t1_tb/t1_tb_tracksite_pull_report_<date>.json

Usage:
    python scripts/pull_t1_tb_tracks.py --date 2026-05-09
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
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "us_t1_tb"
TRACKSITE_DIR = OUT_DIR / "tracksite"
HTML_DIR = TRACKSITE_DIR / "html"
ARTIFACT_DIR = TRACKSITE_DIR / "artifacts"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TRACKS = {
    "TAM": {
        "name": "Tampa Bay Downs",
        "base": "https://www.tampabaydowns.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "CT": {
        "name": "Charles Town",
        "base": "https://www.ctownraces.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "MNR": {
        "name": "Mountaineer",
        "base": "https://www.cnty.com/mountaineer",
        "paths": [
            "/racing/entries-results",
            "/racing/entries",
            "/racing",
        ],
    },
    "CBY": {
        "name": "Canterbury Park",
        "base": "https://www.canterburypark.com",
        "paths": [
            "/racing/entries/",
            "/racing/entries-results/",
            "/racing/",
        ],
    },
    "PID": {
        "name": "Presque Isle Downs",
        "base": "https://www.presqueisledowns.com",
        "paths": [
            "/racing/entries-results",
            "/racing/entries",
            "/",
        ],
    },
    "PRX": {
        "name": "Parx",
        "base": "https://www.parxracing.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "PEN": {
        "name": "Penn National",
        "base": "https://www.hollywoodpnrc.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "ELP": {
        "name": "Ellis Park",
        "base": "https://www.ellisparkracing.com",
        "paths": [
            "/racing/entries",
            "/racing/entries-results",
            "/racing",
        ],
    },
    "LS": {
        "name": "Lone Star Park",
        "base": "https://www.lonestarpark.com",
        "paths": [
            "/racing/entries/",
            "/racing/entries-results/",
            "/racing/",
        ],
    },
    "HOU": {
        "name": "Sam Houston",
        "base": "https://www.shrp.com",
        "paths": [
            "/racing/entries/",
            "/racing/entries-results/",
            "/racing/",
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
        "next_race_day": race_day,
        "first_race_time": first_race,
        "race_count_hint": race_count,
        "mentions_entries": has_entries,
        "mentions_results": has_results,
    }


def _candidate_artifact_urls(discovered_urls: list[str], selected_url: str) -> list[str]:
    urls = list(discovered_urls) + [selected_url]
    out = []
    for u in urls:
        lu = u.lower()
        if any(token in lu for token in [".pdf", ".csv", ".xml", ".ics", "/pdf/"]):
            out.append(u)

    dedup = []
    seen = set()
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)
    return dedup


def _download_artifacts(track_code: str, date_str: str, urls: list[str]) -> list[dict]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    headers = {"User-Agent": USER_AGENT}

    for idx, url in enumerate(urls, start=1):
        try:
            resp = requests.get(url, timeout=45, headers=headers, allow_redirects=True)
            ctype = (resp.headers.get("content-type") or "").lower()
            body = resp.content

            ext = ".bin"
            if ".pdf" in url.lower() or "application/pdf" in ctype:
                ext = ".pdf"
            elif url.lower().endswith(".csv") or "text/csv" in ctype:
                ext = ".csv"
            elif url.lower().endswith(".xml") or "xml" in ctype:
                ext = ".xml"
            elif url.lower().endswith(".ics") or "text/calendar" in ctype:
                ext = ".ics"
            elif "text/html" in ctype:
                ext = ".html"

            file_path = ARTIFACT_DIR / f"{track_code}_{date_str}_{idx:02d}{ext}"
            file_path.write_bytes(body)

            result.append(
                {
                    "url": url,
                    "status_code": resp.status_code,
                    "content_type": ctype,
                    "bytes": len(body),
                    "saved_file": str(file_path.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            result.append({"url": url, "error": str(exc)})

    return result


async def _fetch_candidate(page, url: str) -> dict:
    try:
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
        except Exception:
            # Fallback: try a less strict load event in case the page never settles.
            resp = await page.goto(url, wait_until="load", timeout=20_000)
        await page.wait_for_timeout(1200)
        title = await page.title()
        html = await page.content()
        text = await page.inner_text("body")
        return {
            "url": page.url,
            "requested_url": url,
            "status_code": resp.status if resp else None,
            "title": title,
            "html": html,
            "text": text,
            "score": _quality_score(title, text),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "url": url,
            "requested_url": url,
            "status_code": None,
            "title": "",
            "html": "",
            "text": "",
            "score": -10,
            "error": str(exc),
        }


def _discover_links(html: str, base_url: str) -> list[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    wanted = []
    keywords = [
        "entry",
        "entries",
        "result",
        "results",
        "entries-results",
        "race-information",
        "racebook",
        "calendar",
    ]

    base_host = urlparse(base_url).netloc.lower()
    for href in hrefs:
        href = (href or "").strip()
        if not href or href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in {"http", "https"}:
            continue
        if base_host not in parsed.netloc.lower():
            continue

        lower = full.lower()
        if any(k in lower for k in keywords):
            wanted.append(full)

    # Keep order, remove dupes.
    deduped = []
    seen = set()
    for url in wanted:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


async def _pull_track(context, code: str, cfg: dict) -> dict:
    page = await context.new_page()
    attempts = []
    attempted_urls = set()
    try:
        seed_urls = [cfg["base"].rstrip("/") + path for path in cfg["paths"]]
        discovered_urls = []

        for candidate in seed_urls:
            if candidate in attempted_urls:
                continue
            attempted_urls.add(candidate)
            result = await _fetch_candidate(page, candidate)
            attempts.append(
                {
                    "requested_url": result["requested_url"],
                    "final_url": result["url"],
                    "status_code": result["status_code"],
                    "score": result["score"],
                    "title": result["title"],
                    "error": result["error"],
                }
            )

            if result["html"]:
                discovered_urls.extend(_discover_links(result["html"], result["url"]))

        # Follow discovered links likely to contain entries/results cards.
        for candidate in discovered_urls[:12]:
            if candidate in attempted_urls:
                continue
            attempted_urls.add(candidate)
            result = await _fetch_candidate(page, candidate)
            attempts.append(
                {
                    "requested_url": result["requested_url"],
                    "final_url": result["url"],
                    "status_code": result["status_code"],
                    "score": result["score"],
                    "title": result["title"],
                    "error": result["error"],
                }
            )

        best = max(attempts, key=lambda x: x["score"])

        # Refetch best so we can keep full html/text for that page only.
        best_full = await _fetch_candidate(page, best["requested_url"])
        race_sections = _extract_race_sections(best_full["text"])

        blocked_markers = ["cloudflare", "access denied", "pardon our interruption", "captcha"]
        lower_sample = f"{best_full['title']}\n{best_full['text'][:8000]}".lower()
        blocked = any(marker in lower_sample for marker in blocked_markers)

        return {
            "track_code": code,
            "track_name": cfg["name"],
            "base": cfg["base"],
            "attempts": attempts,
            "selected": {
                "requested_url": best_full["requested_url"],
                "final_url": best_full["url"],
                "status_code": best_full["status_code"],
                "title": best_full["title"],
                "score": best_full["score"],
                "blocked": blocked,
            },
            "race_sections": race_sections,
            "race_section_count": len(race_sections),
            "html": best_full["html"],
            "text_sample": best_full["text"][:6000],
            "discovered_urls": discovered_urls,
        }
    finally:
        await page.close()


async def pull_all_tracks() -> list[dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        try:
            results = []
            for code, cfg in TRACKS.items():
                print(f"  Pulling {code} {cfg['name']}")
                res = await _pull_track(context, code, cfg)
                results.append(res)
            return results
        finally:
            await context.close()
            await browser.close()


def save_outputs(date_str: str, pulled: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for item in pulled:
        code = item["track_code"]
        html_path = HTML_DIR / f"{code}_{date_str}.html"
        html_path.write_text(item["html"] or "", encoding="utf-8")

        summary = _extract_summary_fields(item.get("text_sample", ""))
        artifact_candidates = _candidate_artifact_urls(
            item.get("discovered_urls", []),
            item["selected"]["final_url"],
        )
        artifacts = _download_artifacts(code, date_str, artifact_candidates[:8])

        record = {
            "date": date_str,
            "track_code": code,
            "track_name": item["track_name"],
            "base": item["base"],
            "attempts": item["attempts"],
            "selected": item["selected"],
            "race_section_count": item["race_section_count"],
            "race_sections": item["race_sections"],
            "text_sample": item["text_sample"],
            "summary": summary,
            "discovered_urls": item.get("discovered_urls", []),
            "artifact_candidates": artifact_candidates,
            "downloaded_artifacts": artifacts,
            "html_file": str(html_path.relative_to(PROJECT_ROOT)),
        }

        out_path = TRACKSITE_DIR / f"{code}_{date_str}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        counts[code] = {
            "track": item["track_name"],
            "race_section_count": item["race_section_count"],
            "selected_url": item["selected"]["final_url"],
            "status_code": item["selected"]["status_code"],
            "blocked": item["selected"]["blocked"],
            "score": item["selected"]["score"],
            "summary": summary,
            "downloaded_artifact_count": len([a for a in artifacts if a.get("saved_file")]),
        }

    report = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "track_sites_playwright",
        "track_count": len(pulled),
        "tracks": counts,
    }

    report_path = OUT_DIR / f"t1_tb_tracksite_pull_report_{date_str}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Tier 1 Thoroughbred track-site data")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    args = parser.parse_args()

    datetime.strptime(args.date, "%Y-%m-%d")

    print(f"Pulling T1 Thoroughbred track-site data for {args.date}")
    pulled = asyncio.run(pull_all_tracks())
    report_path = save_outputs(args.date, pulled)

    print("Done")
    print(f"  Report: {report_path}")
    for item in pulled:
        s = item["selected"]
        print(
            f"  {item['track_code']:>3} {item['track_name']:<22} "
            f"sections={item['race_section_count']:<2} status={s['status_code']} blocked={s['blocked']}"
        )


if __name__ == "__main__":
    main()
