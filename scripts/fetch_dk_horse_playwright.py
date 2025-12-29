"""
Fetch DraftKings horse racing odds using Playwright with geo-spoofing.

DraftKings DK Horse platform provides horse racing odds but is geo-restricted to US.
This script uses Playwright to access the site and intercept API calls.

Note: DraftKings primarily focuses on US racing (not UK), so results may be limited.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from playwright.async_api import async_playwright, Page
from typing import Dict, List


class DKHorseOddsFetcher:
    """Fetch horse racing odds from DK Horse using Playwright."""
    
    def __init__(self):
        self.base_url = "https://www.dkhorse.com"
        self.api_responses = []
        self.uk_races = []
        
    async def setup_page(self, page: Page):
        """Set up page with API response interception."""
        
        # Intercept API responses
        async def handle_response(response):
            url = response.url
            
            # Look for racing-related API calls
            if any(keyword in url.lower() for keyword in ['race', 'horse', 'event', 'card', 'odds', 'api']):
                try:
                    if 'json' in response.headers.get('content-type', ''):
                        data = await response.json()
                        print(f"📡 API Call: {url[:80]}...")
                        self.api_responses.append({
                            'url': url,
                            'data': data,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Save interesting responses
                        if len(json.dumps(data)) > 1000:  # Substantial response
                            filename = f"draftkings_api_{len(self.api_responses)}.json"
                            self._save_response(data, filename)
                            
                except Exception as e:
                    pass
        
        page.on('response', handle_response)
        
        # Block unnecessary resources to speed up loading
        async def handle_route(route):
            if route.request.resource_type in ['image', 'stylesheet', 'font', 'media']:
                await route.abort()
            else:
                await route.continue_()
        
        await page.route('**/*', handle_route)
    
    async def fetch_todays_races(self):
        """Fetch today's horse races from DK Horse."""
        
        async with async_playwright() as p:
            # Launch browser with US geo-location
            browser = await p.chromium.launch(
                headless=False,  # Show browser to see what's happening
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                ]
            )
            
            # Create context with US geo-location (New York)
            context = await browser.new_context(
                locale='en-US',
                timezone_id='America/New_York',
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                permissions=['geolocation'],
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            )
            
            page = await context.new_page()
            await self.setup_page(page)
            
            print(f"🔄 Loading DK Horse...")
            
            try:
                # Go to DK Horse homepage
                await page.goto(self.base_url, wait_until='networkidle', timeout=30000)
                
                # Wait a bit for dynamic content and API calls
                await asyncio.sleep(3)
                
                # Take screenshot
                await page.screenshot(path='data/raw/dkhorse_screenshot.png')
                print("📸 Screenshot saved")
                
                # Save page HTML
                content = await page.content()
                with open('data/raw/dkhorse_page.html', 'w', encoding='utf-8') as f:
                    f.write(content)
                print("💾 HTML saved")
                
                # Try to navigate to today's races
                print("\n🔍 Looking for race cards...")
                
                # Look for links containing "races" or "today"
                race_links = await page.query_selector_all('a[href*="race"], a[href*="today"], a[href*="card"]')
                
                if race_links:
                    print(f"Found {len(race_links)} potential race links")
                    
                    # Click first race link
                    if len(race_links) > 0:
                        try:
                            await race_links[0].click(timeout=5000)
                            await asyncio.sleep(3)
                            
                            # Save this page too
                            await page.screenshot(path='data/raw/dkhorse_races.png')
                            
                        except Exception as e:
                            print(f"Could not click link: {e}")
                
                # Wait a bit more for any final API calls
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Error loading page: {e}")
            
            finally:
                await browser.close()
        
        # Analyze collected API responses
        self._analyze_responses()
    
    def _analyze_responses(self):
        """Analyze collected API responses for racing data."""
        
        print(f"\n📊 Analysis:")
        print(f"Total API responses captured: {len(self.api_responses)}")
        
        if self.api_responses:
            print("\nAPI Endpoints discovered:")
            for resp in self.api_responses[:10]:  # Show first 10
                print(f"  • {resp['url'][:100]}")
        else:
            print("⚠️  No API responses captured")
            print("   DK Horse may:")
            print("   - Require login/account")
            print("   - Be geo-blocked outside US")
            print("   - Load data differently than expected")
    
    def _save_response(self, data: Dict, filename: str):
        """Save API response to data/raw."""
        output_dir = "data/raw"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVED] {filepath}")


async def main():
    """Test DK Horse odds fetcher."""
    
    print("=" * 60)
    print("DRAFTKINGS HORSE RACING - PLAYWRIGHT FETCHER")
    print("=" * 60)
    print()
    print("⚠️  Note: DK Horse focuses on US racing")
    print("   UK races may not be available")
    print()
    
    fetcher = DKHorseOddsFetcher()
    await fetcher.fetch_todays_races()
    
    print("\n" + "=" * 60)
    print("Check data/raw/ for screenshots and captured API data")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
