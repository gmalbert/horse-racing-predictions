#!/usr/bin/env python3
"""
Scrape odds from Racing Post odds-comparison pages using Playwright.

The `/odds-comparison` pages may be publicly accessible as they drive
affiliate traffic to bookmakers.

Install:
    pip install playwright playwright-stealth
    python -m playwright install chromium

Usage:
    python scripts/scrape_odds_playwright.py --date 2025-12-28
    python scripts/scrape_odds_playwright.py --date 2025-12-28 --headless
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    from playwright_stealth import Stealth
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("ERROR: Playwright not installed")
    print("Install with: pip install playwright playwright-stealth")
    print("Then run: python -m playwright install chromium")


async def scrape_odds_comparison_page(page, race_url, idx):
    """Scrape a single odds-comparison page"""
    
    # Convert to odds-comparison URL if needed
    if '/odds-comparison' not in race_url:
        race_url = race_url.rstrip('/') + '/odds-comparison'
    
    print(f"\n[{idx}] Visiting: {race_url}")
    
    try:
        # Navigate to odds-comparison page
        await page.goto(race_url, wait_until='domcontentloaded', timeout=20000)
        await page.wait_for_timeout(2000)
        
        # Handle cookies popup
        for selector in ['button:has-text("Accept")', 'button:has-text("Accept All")', '.truste-button1']:
            try:
                button = page.locator(selector).first
                if await button.count() > 0 and await button.is_visible():
                    await button.click()
                    await page.wait_for_timeout(1000)
                    break
            except:
                continue
        
        # Wait for odds table to load
        print("  Waiting for page to fully load...")
        await page.wait_for_timeout(8000)  # Give React plenty of time
        
        # Scroll to bottom to trigger lazy loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(2000)
        
        print("  ✓ Page loaded, extracting data...")
        
        # Extract race info
        course_name = "Unknown"
        try:
            course_elem = page.locator('h1, .RC-headerBox__courseNamelink, [class*="courseName"]').first
            if await course_elem.count() > 0:
                course_name = (await course_elem.inner_text()).strip()
        except:
            pass
        
        print(f"  Course: {course_name}")
        
        # Get the entire page text and parse it for horse names and odds
        page_text = await page.inner_text('body')
        
        # Save full page for debugging first race
        if idx == 1:
            await page.screenshot(path=f"debug_odds_comp_{idx}.png", full_page=True)
            with open(f"debug_odds_comp_{idx}.html", "w", encoding="utf-8") as f:
                f.write(await page.content())
            print(f"  Saved debug files for race {idx}")
        
        # Look for links to horses (these are the horse names)
        horse_links = await page.locator('a[href*="/horses/"]').all()
        
        horses = []
        for link in horse_links[:20]:  # Limit to first 20 horses
            try:
                horse_name = (await link.inner_text()).strip()
                
                # Skip if it's not a valid horse name
                if not horse_name or len(horse_name) < 3:
                    continue
                if re.match(r'^\d+:\d+$', horse_name):  # Skip times
                    continue
                
                # Find odds near this horse name in the page text
                # Look for the horse name in the full page text
                horse_pattern = re.escape(horse_name)
                # Look for fractional odds pattern near the horse name
                context_match = re.search(
                    rf'{horse_pattern}.{{0,200}}?(\d+/\d+)',
                    page_text,
                    re.IGNORECASE
                )
                
                odds = "N/A"
                if context_match:
                    odds = context_match.group(1)
                else:
                    # Try looking backwards too
                    context_match = re.search(
                        rf'(\d+/\d+).{{0,200}}?{horse_pattern}',
                        page_text,
                        re.IGNORECASE
                    )
                    if context_match:
                        odds = context_match.group(1)
                
                # Only add if we haven't seen this horse yet
                if not any(h['horse'] == horse_name for h in horses):
                    horses.append({
                        'horse': horse_name,
                        'odds': odds
                    })
                    print(f"    {horse_name}: {odds}")
            
            except Exception as e:
                continue
        
        if len(horses) >= 3:
            return {
                'course': course_name,
                'url': race_url,
                'horses': horses
            }
        
        print(f"  ⚠️  Only found {len(horses)} horses")
        return None
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


async def scrape_racing_post_odds(date_str, headless=False):
    """Main scraping function"""
    
    print(f"\nScraping Racing Post for {date_str}...")
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--window-size=1920,1080'
            ]
        )
        
        # Create context
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        )
        
        page = await context.new_page()
        
        # Apply stealth
        stealth = Stealth()
        await stealth.apply_stealth_async(page)
        
        # Navigate to main racecards page
        url = f"https://www.racingpost.com/racecards/{date_str}"
        print(f"Navigating to: {url}")
        
        try:
            await page.goto(url, wait_until='networkidle', timeout=30000)
        except PlaywrightTimeout:
            print("  Page load timeout - continuing")
        
        await page.wait_for_timeout(3000)
        
        # Handle cookies
        print("Checking for cookies...")
        for selector in ['button:has-text("Accept")', 'div:has-text("Accept") button']:
            try:
                button = page.locator(selector).first
                if await button.count() > 0 and await button.is_visible():
                    await button.click()
                    print(f"  ✓ Accepted cookies")
                    await page.wait_for_timeout(2000)
                    break
            except:
                continue
        
        # Wait for meetings
        try:
            await page.wait_for_selector('.RC-meetingItem', timeout=15000)
            print("✓ Meetings loaded")
        except PlaywrightTimeout:
            print("❌ Could not find meetings")
            await browser.close()
            return None
        
        # Find race URLs
        race_links = await page.locator('a[href*="/racecards/"]').all()
        race_urls = []
        for link in race_links:
            href = await link.get_attribute('href')
            if href:
                if href.startswith('/'):
                    href = f"https://www.racingpost.com{href}"
                parts = href.split('/racecards/')
                if len(parts) > 1 and '/' in parts[1] and len(parts[1].split('/')) >= 4:
                    if href not in race_urls and '/odds-comparison' not in href:
                        race_urls.append(href)
        
        print(f"\nFound {len(race_urls)} race URLs")
        
        if len(race_urls) == 0:
            await browser.close()
            return None
        
        all_odds = []
        
        # Scrape first 5 races
        for idx, race_url in enumerate(race_urls[:5], 1):
            race_data = await scrape_odds_comparison_page(page, race_url, idx)
            if race_data and len(race_data.get('horses', [])) > 0:
                all_odds.append(race_data)
        
        await browser.close()
        return all_odds


def save_odds(odds_data, date_str):
    """Save odds to JSON file"""
    if not odds_data:
        print("\n❌ No odds data to save")
        return
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"odds_racingpost_{date_str}.json"
    
    with open(output_file, 'w') as f:
        json.dump(odds_data, f, indent=2)
    
    print(f"\n✓ Saved odds to: {output_file}")
    print(f"  Total races: {len(odds_data)}")
    print(f"  Total horses: {sum(len(r['horses']) for r in odds_data)}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Racing Post odds using Playwright")
    parser.add_argument('--date', type=str, required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    
    args = parser.parse_args()
    
    if not PLAYWRIGHT_AVAILABLE:
        print("\n❌ Cannot run without Playwright")
        print("\nInstall requirements:")
        print("  pip install playwright playwright-stealth")
        print("  python -m playwright install chromium")
        return 1
    
    # Validate date
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("Expected: YYYY-MM-DD")
        return 1
    
    print("="*60)
    print("RACING POST ODDS SCRAPER (PLAYWRIGHT)")
    print("="*60)
    print("\nUsing odds-comparison pages for better access.")
    
    # Run scraper
    odds_data = asyncio.run(scrape_racing_post_odds(args.date, args.headless))
    
    if odds_data and len(odds_data) > 0:
        save_odds(odds_data, args.date)
        print("\n✅ SUCCESS!")
        return 0
    else:
        print("\n❌ FAILED - Could not scrape odds")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
