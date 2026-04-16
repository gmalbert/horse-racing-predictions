from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    br = p.chromium.launch(headless=True)
    ctx = br.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        locale="en-GB",
    )
    pg = ctx.new_page()
    pg.goto("https://www.racingpost.com/racecards/2026-04-14/", wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_timeout(5000)
    print("Title:", pg.title()[:80])
    src = pg.content()
    print("Contains 'racecard':", src.count("racecard"))
    print("Contains 'RC-meetingItem':", src.count("RC-meetingItem"))
    print("Contains 'CF-':", src.count("CF-"))
    print("Contains 'data-race-is-over':", src.count("data-race-is-over"))
    print("Contains 'blocked':", src.lower().count("blocked"))
    print("Contains 'captcha':", src.lower().count("captcha"))
    # Try querying links
    links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => a.href.includes('/racecards/')).slice(0,5).map(a => a.href)")
    print(f"\nRacecard links ({len(links)}):")
    for l in links:
        print(" ", l[:120])
    br.close()
