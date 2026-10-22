"""Exercise the built React app against FastAPI with isolated betting state.

Run from the repository root after `npm run build --prefix web/frontend`:
    python web/tests/smoke_web.py
Use --url http://localhost:8080 for read-only checks of an existing Compose stack.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "web" / "test-results"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_url(url, process):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"Server never became ready: {url}")


def settle_page(page):
    expect(page.locator("main h1")).to_be_visible()
    expect(page.locator(".loading")).to_have_count(0, timeout=120_000)
    expect(page.locator(".error-box")).to_have_count(0)


def exercise(url, writable):
    errors, failures, timings = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("requestfailed", lambda request: failures.append(request.url))
        page.on("response", lambda response: failures.append(f"{response.status} {response.url}") if response.status >= 400 else None)
        for route, tabs in [("/", 5), ("/us-racing", 5), ("/data-explorer", 4),
                            ("/trainer-stats", 4), ("/jockey-stats", 4),
                            ("/betting-manager", 4), ("/backtest", 4)]:
            start = time.perf_counter()
            page.goto(url + route)
            settle_page(page)
            timings.append({"route": route, "ready_ms": round((time.perf_counter() - start) * 1000)})
            buttons = page.locator(".tabs").first.locator("button")
            expect(buttons).to_have_count(tabs)
            for i in range(tabs):
                buttons.nth(i).click()
                settle_page(page)
            # Recharts animates bars on mount; wait for the visible bar geometry.
            bars = page.locator(".recharts-bar-rectangle path")
            if bars.count():
                expect(bars.first).to_have_attribute("d", re.compile(r".*[1-9].*"))
            text = page.locator("main").inner_text()
            assert "\u00f0\u0178" not in text and "\ufffd" not in text, f"Broken encoding at {route}"
            assert "Historical data: latest" not in text
            expect(page.locator(".sidebar > .brand")).to_have_count(0)
            if route == "/betting-manager":
                assert "£" not in text and "$" in text, "Betting Manager must display USD"
            expect(page.get_by_role("link", name="Betting Oracle")).to_be_visible()
            page.screenshot(path=str(OUTPUT / f"{route.strip('/') or 'international'}.png"), full_page=True)
            print(f"PASS {route}: {tabs} tabs", flush=True)

        page.goto(url + "/data-explorer")
        settle_page(page)
        page.get_by_role("button", name="Course", exact=True).click()
        page.get_by_role("textbox", name="Search Course", exact=True).fill("Ascot")
        page.get_by_role("checkbox", name="Ascot", exact=True).check()
        page.get_by_role("button", name="Done", exact=True).click()
        expect(page.get_by_role("button", name="Course", exact=True)).to_contain_text("Ascot")
        with page.expect_response(lambda response: "/api/explorer/query" in response.url and response.request.method == "POST") as filtered:
            page.get_by_role("button", name="Apply Filters", exact=True).click()
        assert filtered.value.request.post_data_json["courses"] == ["Ascot"]
        assert filtered.value.json()["total_rows"] > 0
        page.get_by_role("button", name="Reset", exact=True).click()
        expect(page.get_by_role("button", name="Course", exact=True)).to_contain_text("All course")
        page.get_by_label("Horse name", exact=True).fill("[")
        page.get_by_role("button", name="Apply Filters", exact=True).click()
        expect(page.get_by_role("button", name="Apply Filters", exact=True)).to_be_enabled(timeout=120_000)
        settle_page(page)
        page.get_by_role("button", name="Reset", exact=True).click()
        settle_page(page)
        page.locator(".tabs button").filter(has_text="Raw Data").click()
        expect(page.get_by_role("button", name="Next", exact=False)).to_be_enabled(timeout=120_000)
        page.get_by_role("button", name="Next", exact=False).click()
        expect(page.get_by_text("Page 2 of", exact=False)).to_be_visible()

        page.goto(url)
        settle_page(page)
        course = page.get_by_role("combobox", name="Course", exact=True)
        if course.count() and course.locator("option").count() > 2:
            course.select_option(index=1)
            expect(page.get_by_role("combobox", name="Select race", exact=True).locator("option")).not_to_have_count(0)
            page.get_by_role("button", name="Reset race filters").click()

        for route, entity in [("/trainer-stats", "trainer"), ("/jockey-stats", "jockey")]:
            page.goto(url + route)
            settle_page(page)
            page.get_by_role("searchbox", name=f"Search {entity}", exact=True).fill("no-such-name-12345")
            expect(page.locator("table").first).to_contain_text("No results match your selection.")
            page.get_by_role("button", name="Reset leaderboard", exact=True).click()
            expect(page.locator("table").first).not_to_contain_text("No results match your selection.")

        if writable:
            page.goto(url + "/betting-manager")
            settle_page(page)
            amount = page.get_by_role("spinbutton", name="Set bankroll (USD)", exact=True)
            amount.fill("1200")
            amount.press("Enter")
            expect(page.locator(".save-confirmation")).to_have_text("Bankroll updated.")
            assert context.request.get(url + "/api/bankroll").json()["current_bankroll"] == 1200
            amount.fill("1000")
            amount.press("Enter")
            expect(page.locator(".save-confirmation")).to_have_text("Bankroll updated.")
            page.locator(".tabs button").filter(has_text="Log Bet").click()
            page.get_by_label("Horse name", exact=True).fill("Migration Smoke Horse")
            page.get_by_label("Market decimal odds").fill("3")
            page.get_by_label("Stake $").fill("10")
            page.get_by_role("button", name="Log Bet", exact=True).click()
            expect(page.get_by_text("Logged bet on Migration Smoke Horse.")).to_be_visible()
            page.locator(".tabs button").filter(has_text="P&L Tracker").click()
            page.get_by_role("button", name="Lost", exact=True).click()
            expect(page.get_by_role("button", name="Lost", exact=True)).to_have_count(0)
            page.reload()
            settle_page(page)
            page.locator(".tabs button").filter(has_text="P&L Tracker").click()
            expect(page.get_by_text("Migration Smoke Horse", exact=True)).to_be_visible()
            state = context.request.get(url + "/api/bankroll").json()
            assert state["current_bankroll"] == 990 and state["total_pnl"] == -10, state
            print("PASS browser bet logging, settlement and reload persistence", flush=True)

        page.set_viewport_size({"width": 390, "height": 844})
        for route in ["/", "/us-racing", "/data-explorer", "/trainer-stats", "/jockey-stats", "/betting-manager", "/backtest"]:
            page.goto(url + route)
            settle_page(page)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), f"Mobile overflow: {route}"
            assert page.locator(".tabs").first.bounding_box()["height"] <= 60, f"Mobile tabs wrap: {route}"
            page.screenshot(path=str(OUTPUT / f"{route.strip('/') or 'international'}-mobile.png"))
        page.goto(url)
        settle_page(page)
        page.get_by_role("button", name="Menu", exact=True).click()
        page.get_by_role("link", name="Jockey Stats").click()
        settle_page(page)
        expect(page.get_by_role("button", name="Menu", exact=True)).to_have_attribute("aria-expanded", "false")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile page overflows"
        page.screenshot(path=str(OUTPUT / "mobile.png"), full_page=True)

        # Expected outage responses are checked in a separate context.
        assert not errors, errors
        assert not failures, failures
        fallback = browser.new_page()
        fallback.route("**/api/**", lambda route: route.fulfill(status=503, json={"detail": "test outage"}))
        fallback.goto(url + "/data-explorer")
        expect(fallback.get_by_text("showing the precomputed overview", exact=False)).to_be_visible()
        expect(fallback.locator("table tbody tr").first).to_be_visible()
        fallback.goto(url + "/betting-manager")
        expect(fallback.get_by_role("button", name="Retry connection")).to_be_visible()
        fallback.close()

        # A failed static fetch must recover through the user-visible Retry action.
        retry = browser.new_page()
        calls = [0]
        def flaky_export(route):
            calls[0] += 1
            if calls[0] == 1:
                route.fulfill(status=503, json={"detail": "temporary export failure"})
            else:
                route.continue_()
        retry.route("**/data/backtest.json", flaky_export)
        retry.goto(url + "/backtest")
        expect(retry.locator(".error-box")).to_be_visible()
        retry.get_by_role("button", name="Retry", exact=True).click()
        expect(retry.locator(".error-box")).to_have_count(0)
        expect(retry.locator(".loading")).to_have_count(0)
        assert calls[0] >= 2
        retry.close()

        # Current cards are a separate fixture, since shipped predictions are historical.
        current = browser.new_page()
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        manifest = json.loads((ROOT / "web/frontend/public/data/manifest.json").read_text())
        manifest["gb_prediction_dates"] = [today, tomorrow]
        cards = []
        for stamp, source in zip([today, tomorrow], sorted((ROOT / "web/frontend/public/data/predictions").glob("*.json"))[:2]):
            card = json.loads(source.read_text())
            card["date"] = stamp
            for race in card["races"]:
                race["date"] = stamp
            cards.append(card)
        current.route("**/data/manifest.json", lambda route: route.fulfill(json=manifest))
        def current_card(route):
            stamp = route.request.url.rsplit("/", 1)[-1].removesuffix(".json")
            route.fulfill(json=cards[0] if stamp == today else cards[1])
        current.route("**/data/predictions/*.json", current_card)
        current.goto(url)
        settle_page(current)
        expect(current.get_by_role("combobox", name="Date", exact=True)).to_have_value("combined")
        races = current.get_by_role("combobox", name="Select race", exact=True).locator("option")
        expect(races).to_have_count(sum(len(card["races"]) for card in cards))
        assert today in races.first.inner_text() and tomorrow in races.last.inner_text()
        current.close()
        print("PASS fetch retry and combined Today & Tomorrow cards", flush=True)
        browser.close()
    (OUTPUT / "smoke-report.json").write_text(json.dumps({"timings": timings, "console_errors": errors, "failed_requests": failures}, indent=2))
    print("PASS mobile navigation and offline fallbacks; reports in web/test-results", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Read-only smoke test against a running app")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.url:
        exercise(args.url.rstrip("/"), writable=False)
        return
    runtime = ROOT / "web" / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=runtime) as folder:
        api_port, web_port = free_port(), free_port()
        env = {**os.environ, "HRP_STATE_DIR": folder, "VITE_API_PROXY_TARGET": f"http://127.0.0.1:{api_port}"}
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with open(OUTPUT / "servers.log", "w") as logs:
            api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", "web/backend", "--port", str(api_port)], cwd=ROOT, env=env, stdout=logs, stderr=logs, creationflags=flags)
            web = None
            try:
                wait_url(f"http://127.0.0.1:{api_port}/api/health", api)
                web = subprocess.Popen(["node", "node_modules/vite/bin/vite.js", "preview", "--host", "127.0.0.1", "--port", str(web_port), "--strictPort"], cwd=ROOT / "web/frontend", env=env, stdout=logs, stderr=logs, creationflags=flags)
                wait_url(f"http://127.0.0.1:{web_port}", web)
                exercise(f"http://127.0.0.1:{web_port}", writable=True)
            finally:
                for process in (web, api):
                    if process:
                        process.terminate()
                        process.wait(timeout=15)


if __name__ == "__main__":
    main()
