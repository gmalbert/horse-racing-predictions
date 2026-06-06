"""
Fetch US racecards from all implemented roadmap sources and produce one merged racecards file.

DESIGN GOALS (v2 - cache-first, parallel, bounded):
  - Cache-first: skip T1 pipeline / NYRA if output files already exist for the date.
  - Parallel: T2/T3 Playwright probes run concurrently (one page per track).
  - Bounded: 8s per page.goto(), max 2 URL paths per track, 90s global budget.
  - JSON discovery: parallel HTTP via thread pool (opt-in via --discover).

Outputs:
- data/raw/us_racecards_YYYY-MM-DD.json
- data/raw/us_source_report_YYYY-MM-DD.json
- data/raw/us_json_discovery_YYYY-MM-DD.json  (only with --discover)

Usage:
    python scripts/fetch_us_all_sources.py --date 2026-05-10
    python scripts/fetch_us_all_sources.py --date 2026-05-10 --discover
    python scripts/fetch_us_all_sources.py --date 2026-05-10 --force
    python scripts/fetch_us_all_sources.py --date 2026-05-10 --no-playwright
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

T2_TB_TRACKS = {
    "T2-TB-01": {"code": "GP",  "name": "Gulfstream Park",  "base": "https://www.gulfstreampark.com"},
    "T2-TB-02": {"code": "SA",  "name": "Santa Anita",      "base": "https://www.santaanita.com"},
    "T2-TB-03": {"code": "DMR", "name": "Del Mar",           "base": "https://www.dmtc.com"},
    "T2-TB-04": {"code": "OP",  "name": "Oaklawn",           "base": "https://www.oaklawn.com"},
    "T2-TB-05": {"code": "FG",  "name": "Fair Grounds",      "base": "https://www.fairgroundsracecourse.com"},
    "T2-TB-06": {"code": "MTH", "name": "Monmouth Park",     "base": "https://www.monmouthpark.com"},
}

T2_H_TRACKS = {
    "T2-H-01": {"code": "MDL", "name": "Meadowlands",  "base": "https://playmeadowlands.com"},
    "T2-H-02": {"code": "YON", "name": "Yonkers",       "base": "https://www.empirecitycasino.com"},
    "T2-H-03": {"code": "HOO", "name": "Hoosier Park",  "base": "https://www.caesars.com/harrahs-hoosier-park"},
}

T3_TRACKS = {
    "T3-02": {"code": "CD",  "name": "Churchill Downs", "base": "https://www.churchilldowns.com"},
    "T3-03": {"code": "KEE", "name": "Keeneland",        "base": "https://www.keeneland.com"},
}

JSON_DISCOVERY_TARGETS = {
    "J-01": {"name": "Meadowlands",     "base": "https://playmeadowlands.com"},
    "J-02": {"name": "Yonkers",          "base": "https://www.empirecitycasino.com"},
    "J-03": {"name": "Hoosier Park",    "base": "https://www.caesars.com/harrahs-hoosier-park"},
    "J-04": {"name": "Scioto Downs",    "base": "https://www.sciotodowns.com"},
    "J-05": {"name": "Northfield Park", "base": "https://www.northfieldpark.com"},
    "J-06": {"name": "Pocono Downs",    "base": "https://www.caesars.com/mohegan-pennsylvania"},
    "J-07": {"name": "Harrah Philly",   "base": "https://www.caesars.com/harrahs-philadelphia"},
    "J-08": {"name": "Rosecroft",       "base": "https://www.rosecroft.com"},
    "J-09": {"name": "Running Aces",    "base": "https://www.runningaces.com"},
    "J-10": {"name": "Tioga Downs",     "base": "https://www.tiogadowns.com"},
    "J-11": {"name": "Vernon Downs",    "base": "https://www.vernondowns.com"},
    "J-12": {"name": "Plainridge Park", "base": "https://www.plainridgeparkcasino.com"},
    "J-13": {"name": "NYRA",            "base": "https://www.nyra.com"},
    "J-14": {"name": "CDI family",      "base": "https://www.churchilldowns.com"},
    "J-15": {"name": "TVG/TwinSpires",  "base": "https://www.tvg.com"},
}

PROBE_PATHS = ["/racing/entries", "/"]

COMMON_JSON_PATHS = [
    "/api/races", "/api/racecards", "/api/entries",
    "/api/v1/races", "/api/v1/entries", "/wp-json/", "/feed.json",
]


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run(cmd: list, timeout_seconds: int = 300) -> dict:
    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=timeout_seconds)
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-30:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(cmd),
            "returncode": 124,
            "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-20:]) if exc.stdout else "",
            "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-20:]) if exc.stderr else "",
            "timed_out": True,
        }


def _quality_score(text: str) -> int:
    lower = text[:20000].lower()
    score = 0
    if "entries" in lower or "entry" in lower:
        score += 2
    if "results" in lower:
        score += 1
    if re.search(r"\brace\s+\d+\b", lower):
        score += 3
    if "post time" in lower:
        score += 1
    if any(kw in lower for kw in ("access denied", "forbidden", "cloudflare", "just a moment")):
        score -= 4
    return score


def _extract_race_sections(text: str, max_sections: int = 12) -> list:
    chunks = re.split(r"(?=\bRace\s+\d{1,2}\b)", text, flags=re.IGNORECASE)
    out = []
    for chunk in chunks:
        if not re.search(r"\bRace\s+\d{1,2}\b", chunk, flags=re.IGNORECASE):
            continue
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        out.append({"header": lines[0][:160], "snippet": "\n".join(lines[:12])[:2400]})
        if len(out) >= max_sections:
            break
    return out


def _extract_runners_from_snippet(snippet: str) -> list:
    runners = []
    for line in snippet.splitlines():
        m = re.match(r"^\s*(\d{1,2})[\.)\-\s]+([A-Za-z0-9''.,\-\s]{2,})$", line.strip())
        if m:
            runners.append({"number": m.group(1), "horse": m.group(2).strip()})
        if len(runners) >= 20:
            break
    return runners


def _summary_to_races(track_code, track_name, date_str, source_url, sections, source_tag) -> list:
    if sections:
        return [
            {
                "race": sec.get("header") or f"Race {i}",
                "race_name": sec.get("header") or f"Race {i}",
                "race_time": "",
                "course": track_name,
                "track": track_code,
                "surface": "",
                "distance": "",
                "source": source_tag,
                "source_url": source_url,
                "date": date_str,
                "runners": _extract_runners_from_snippet(sec.get("snippet", "")),
            }
            for i, sec in enumerate(sections, start=1)
        ]
    return [{
        "race": "Race 1",
        "race_name": "Entries Summary",
        "race_time": "",
        "course": track_name,
        "track": track_code,
        "surface": "",
        "distance": "",
        "source": source_tag,
        "source_url": source_url,
        "date": date_str,
        "runners": [],
    }]


def _run_t1_pipeline(date_str: str, force: bool) -> dict:
    report_path = RAW_DIR / "us_t1_all" / f"t1_all_pipeline_report_{date_str}.json"
    if not force and report_path.exists():
        print(f"[SKIP] T1 pipeline - cached report found: {report_path.name}")
        return {"skipped": True, "reason": "cached"}
    print(f"[RUN]  T1 all pipeline for {date_str} ...")
    return _run([sys.executable, "scripts/run_t1_all_daily_pipeline.py", "--date", date_str, "--days", "1"], timeout_seconds=420)


def _run_nyra(date_str: str, force: bool) -> dict:
    nyra_path = RAW_DIR / f"nyra_entries_{date_str}.json"
    if not force and nyra_path.exists():
        print(f"[SKIP] NYRA fetch - cached file found: {nyra_path.name}")
        return {"skipped": True, "reason": "cached"}
    print(f"[RUN]  NYRA entries fetch for {date_str} ...")
    return _run([sys.executable, "scripts/fetch_nyra_entries.py", "--date", date_str, "--tracks", "BEL", "AQU", "SAR"], timeout_seconds=180)


async def _probe_one_track(page, task_id: str, item: dict, group: str, date_str: str, deadline: float):
    code = item["code"]
    name = item["name"]
    base = item["base"].rstrip("/")
    best = {"score": -999, "url": None, "sections": []}
    tries = []

    for path in PROBE_PATHS:
        if time.monotonic() > deadline:
            tries.append({"url": base + path, "error": "global_deadline_exceeded"})
            break
        url = base + path
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=8000)
            await page.wait_for_timeout(400)
            text = await page.inner_text("body")
            score = _quality_score(text)
            tries.append({"url": url, "score": score, "char_count": len(text)})
            if score > best["score"]:
                best = {"score": score, "url": url, "sections": _extract_race_sections(text)}
        except Exception as exc:
            tries.append({"url": url, "error": str(exc)[:100]})

    report_row = {
        "task_id": task_id,
        "group": group,
        "track_code": code,
        "track_name": name,
        "best_score": best["score"],
        "selected_url": best["url"],
        "race_section_count": len(best["sections"]),
        "pages_tried": tries,
    }
    races = _summary_to_races(code, name, date_str, best["url"] or base, best["sections"], f"{group.lower()}_tracksite")
    return report_row, races


async def _pull_track_group_parallel(group_name: str, tracks: dict, date_str: str, deadline: float):
    from playwright.async_api import async_playwright

    report_rows = []
    merged_races = []

    if time.monotonic() > deadline:
        print(f"[SKIP] {group_name} - global deadline exceeded")
        return report_rows, merged_races

    print(f"[PROBE] {group_name} - {len(tracks)} tracks in parallel ...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            pages = []
            for _ in tracks:
                ctx = await browser.new_context(user_agent=USER_AGENT, ignore_https_errors=True)
                page = await ctx.new_page()
                pages.append(page)

            tasks = [
                _probe_one_track(page, task_id, item, group_name, date_str, deadline)
                for page, (task_id, item) in zip(pages, tracks.items())
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                row, races = result
                report_rows.append(row)
                merged_races.extend(races)
        finally:
            await browser.close()

    return report_rows, merged_races


def _collect_existing_t1(date_str: str):
    source_rows = []
    races = []

    t1_reports = [
        (RAW_DIR / "us_t1_tb" / f"t1_tb_tracksite_pull_report_{date_str}.json", "T1-TB"),
        (RAW_DIR / "us_t1_qh" / f"t1_qh_tracksite_pull_report_{date_str}.json", "T1-QH"),
        (RAW_DIR / "us_t1_h"  / f"t1_h_tracksite_pull_report_{date_str}.json",  "T1-H"),
    ]

    for path, group in t1_reports:
        data = _load_json(path)
        if not data:
            continue

        tracks_obj = data.get("tracks", {})
        if isinstance(tracks_obj, dict):
            items = [
                (code, {
                    "track_name": td.get("track", code),
                    "score": td.get("score", -999),
                    "url": td.get("selected_url"),
                    "sections": [],
                })
                for code, td in tracks_obj.items()
            ]
        else:
            items = []
            for row in (tracks_obj or []):
                code = row.get("track_code") or "UNK"
                bp = row.get("best_page") or {}
                items.append((code, {
                    "track_name": row.get("track_name", code),
                    "score": row.get("best_quality_score", -999),
                    "url": bp.get("url"),
                    "sections": bp.get("race_sections", []),
                }))

        for code, td in items:
            source_rows.append({
                "task_id": group,
                "group": group,
                "track_code": code,
                "track_name": td.get("track_name"),
                "best_score": td.get("score", -999),
                "selected_url": td.get("url"),
                "race_section_count": len(td.get("sections") or []),
                "pages_tried": [],
            })
            races.extend(
                _summary_to_races(
                    code, td.get("track_name", code), date_str,
                    td.get("url") or "", td.get("sections") or [],
                    f"{group.lower()}_existing",
                )
            )

    return source_rows, races


def _collect_t1_canonical_runner_races(date_str: str) -> list:
    """Load runner-level races from canonical T1 JSON outputs when available."""
    out = []
    files = [
        (PROC_DIR / f"us_t1_tb_canonical_{date_str}.json", "T1-TB-canonical"),
        (PROC_DIR / f"us_t1_qh_canonical_{date_str}.json", "T1-QH-canonical"),
        (PROC_DIR / f"us_t1_h_canonical_{date_str}.json", "T1-H-canonical"),
    ]

    for path, source_tag in files:
        rows = _load_json(path)
        if not isinstance(rows, list) or not rows:
            continue

        grouped = {}
        for row in rows:
            runner_name = (row.get("runner_name") or "").strip()
            if not runner_name:
                continue

            track_code = (row.get("track_code") or "UNK").strip()
            track_name = (row.get("track_name") or track_code).strip()
            race_number = row.get("race_number")
            race_name = (row.get("race_name") or "").strip()
            race_time = (row.get("race_time") or "").strip()
            distance = (row.get("distance") or "").strip()
            surface = (row.get("surface") or "").strip()
            source_url = (row.get("source_url") or "").strip()

            race_key = (
                track_code,
                track_name,
                str(race_number) if race_number is not None else "",
                race_name,
                race_time,
                distance,
                surface,
                source_url,
            )
            if race_key not in grouped:
                grouped[race_key] = {
                    "race": race_name or (f"Race {race_number}" if race_number is not None else "Race"),
                    "race_name": race_name or (f"Race {race_number}" if race_number is not None else "Race"),
                    "race_time": race_time,
                    "course": track_name,
                    "track": track_code,
                    "surface": surface,
                    "distance": distance,
                    "source": source_tag,
                    "source_url": source_url,
                    "date": date_str,
                    "runners": [],
                }

            grouped[race_key]["runners"].append({
                "number": str(row.get("runner_number") or "").strip(),
                "horse": runner_name,
                "jockey": (row.get("jockey") or "").strip(),
                "trainer": (row.get("trainer") or "").strip(),
                "weight": (row.get("weight") or row.get("weight_lbs") or ""),
            })

        out.extend(grouped.values())

    return out


def _probe_json_target(task_id: str, item: dict) -> dict:
    base = item["base"].rstrip("/")
    name = item["name"]
    hits = []

    for path in COMMON_JSON_PATHS:
        url = base + path
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": USER_AGENT})
            ct = (resp.headers.get("content-type") or "").lower()
            body = resp.text.strip()
            if resp.status_code < 400 and ("json" in ct or body.startswith("{") or body.startswith("[")):
                hits.append({"url": url, "status": resp.status_code, "content_type": ct})
        except Exception:
            pass

    try:
        resp = requests.get(base + "/", timeout=6, headers={"User-Agent": USER_AGENT})
        for m in re.findall(r"https?://[^\"'\s>]+(?:\.json|/api/[^\"'\s>]+)", resp.text, re.IGNORECASE):
            hits.append({"url": m, "status": None, "content_type": "html_scan"})
    except Exception:
        pass

    seen = set()
    unique = []
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"])
            unique.append(h)

    return {
        "task_id": task_id,
        "source_track": name,
        "status": "In Progress" if unique else "Backlog",
        "endpoint_discovery": bool(unique),
        "parser_complete": False,
        "hits": unique[:20],
    }


def _discover_json_endpoints(date_str: str) -> dict:
    print(f"[DISCOVER] JSON endpoints - {len(JSON_DISCOVERY_TARGETS)} targets ...")
    discovered = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_probe_json_target, tid, item): tid for tid, item in JSON_DISCOVERY_TARGETS.items()}
        for future in futures:
            try:
                discovered.append(future.result(timeout=30))
            except Exception:
                discovered.append({"task_id": futures[future], "error": "failed"})

    payload = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": discovered,
        "summary": {
            "total": len(discovered),
            "with_discovery": sum(1 for d in discovered if d.get("endpoint_discovery")),
        },
    }
    out = RAW_DIR / f"us_json_discovery_{date_str}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] JSON discovery -> {out.name}")
    return payload


def _ingest_nyra(date_str: str) -> list:
    nyra_path = RAW_DIR / f"nyra_entries_{date_str}.json"
    nyra_raw = _load_json(nyra_path) or {}
    nyra_tracks = nyra_raw.get("tracks", {}) if isinstance(nyra_raw, dict) else {}
    races = []
    if isinstance(nyra_tracks, dict):
        for track_code, track_races in nyra_tracks.items():
            for race in track_races or []:
                race_copy = dict(race)
                race_copy.setdefault("track", track_code)
                race_copy.setdefault("course", race_copy.get("course") or track_code)
                race_copy.setdefault("source", "T3-NYRA")
                race_copy.setdefault("date", date_str)
                races.append(race_copy)
    return races


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch all US sources -> merged racecards JSON")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Re-pull even if cached")
    parser.add_argument("--discover", action="store_true", help="Run JSON endpoint discovery (adds ~60s)")
    parser.add_argument("--no-playwright", action="store_true", help="Skip T2/T3 Playwright probes")
    args = parser.parse_args()

    date_str = args.date
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_racecards = RAW_DIR / f"us_racecards_{date_str}.json"
    if not args.force and out_racecards.exists():
        print(f"[SKIP] us_racecards_{date_str}.json already exists. Use --force to re-build.")
        return

    t_start = time.monotonic()
    playwright_deadline = t_start + 90

    print(f"\n{'='*60}")
    print(f"  US ALL-SOURCES FETCH  -  {date_str}")
    print(f"{'='*60}\n")

    t1_pipeline_result = _run_t1_pipeline(date_str, args.force)
    nyra_result = _run_nyra(date_str, args.force)
    print(f"  T1+NYRA elapsed: {time.monotonic()-t_start:.0f}s")

    t2_tb_report, t2_tb_races = [], []
    t2_h_report, t2_h_races = [], []
    t3_report, t3_races = [], []

    if not args.no_playwright:
        try:
            t2_tb_report, t2_tb_races = asyncio.run(
                _pull_track_group_parallel("T2-TB", T2_TB_TRACKS, date_str, playwright_deadline)
            )
            print(f"  T2-TB: {len(t2_tb_races)} races  ({time.monotonic()-t_start:.0f}s)")

            t2_h_report, t2_h_races = asyncio.run(
                _pull_track_group_parallel("T2-H", T2_H_TRACKS, date_str, playwright_deadline)
            )
            print(f"  T2-H:  {len(t2_h_races)} races  ({time.monotonic()-t_start:.0f}s)")

            t3_report, t3_races = asyncio.run(
                _pull_track_group_parallel("T3", T3_TRACKS, date_str, playwright_deadline)
            )
            print(f"  T3:    {len(t3_races)} races  ({time.monotonic()-t_start:.0f}s)")
        except Exception as exc:
            print(f"[WARN] Playwright probes failed: {exc}")
    else:
        print("[SKIP] Playwright probes (--no-playwright)")

    t1_rows, t1_races = _collect_existing_t1(date_str)
    t1_runner_races = _collect_t1_canonical_runner_races(date_str)
    if t1_runner_races:
        runner_tracks = {(r.get("track") or "").strip() for r in t1_runner_races}
        # Replace placeholder entries for tracks where we have canonical runner-level races.
        t1_races = [
            r for r in t1_races
            if ((r.get("track") or "").strip() not in runner_tracks)
        ]
        t1_races.extend(t1_runner_races)
    print(f"  T1 collected: {len(t1_races)} races")

    nyra_races = _ingest_nyra(date_str)
    print(f"  NYRA races: {len(nyra_races)}")

    json_discovery = None
    if args.discover:
        json_discovery = _discover_json_endpoints(date_str)

    merged_races = []
    merged_races.extend(nyra_races)
    merged_races.extend(t1_races)
    merged_races.extend(t2_tb_races)
    merged_races.extend(t2_h_races)
    merged_races.extend(t3_races)
    for race in merged_races:
        race["date"] = date_str

    source_counts = {
        "nyra_races": len(nyra_races),
        "t1_races": len(t1_races),
        "t2_tb_races": len(t2_tb_races),
        "t2_h_races": len(t2_h_races),
        "t3_races": len(t3_races),
        "total_races": len(merged_races),
    }

    racecards_payload = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "us_all_sources_v2",
        "racecards": merged_races,
        "source_counts": source_counts,
    }

    fallback_used = False
    attempted_source_counts = dict(source_counts)
    if source_counts["total_races"] == 0 and out_racecards.exists():
        previous = _load_json(out_racecards) or {}
        previous_races = previous.get("racecards") or []
        if previous_races:
            print("[WARN] Fresh fetch returned zero races; reusing existing non-empty racecards snapshot")
            racecards_payload = previous
            racecards_payload["fallback_used"] = True
            racecards_payload["fallback_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
            source_counts = previous.get("source_counts") or {
                "nyra_races": 0,
                "t1_races": 0,
                "t2_tb_races": 0,
                "t2_h_races": 0,
                "t3_races": 0,
                "total_races": len(previous_races),
            }
            fallback_used = True

    out_racecards.write_text(json.dumps(racecards_payload, indent=2), encoding="utf-8")

    source_report = {
        "date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
        "pipeline_runs": {"t1_all": t1_pipeline_result, "nyra": nyra_result},
        "groups": {"T1": t1_rows, "T2-TB": t2_tb_report, "T2-H": t2_h_report, "T3": t3_report},
        "source_counts": source_counts,
        "attempted_source_counts": attempted_source_counts,
        "fallback_used": fallback_used,
        "json_discovery_summary": json_discovery.get("summary") if json_discovery else None,
    }
    out_report = RAW_DIR / f"us_source_report_{date_str}.json"
    out_report.write_text(json.dumps(source_report, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE  ({elapsed:.0f}s)")
    print(f"  Merged races : {len(merged_races)}")
    for k, v in source_counts.items():
        if k != "total_races":
            print(f"    {k:<18}: {v}")
    print(f"  -> {out_racecards.name}")
    print(f"  -> {out_report.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()