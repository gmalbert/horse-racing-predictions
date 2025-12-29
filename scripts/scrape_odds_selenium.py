#!/usr/bin/env python3
"""
Scrape odds from Racing Post using Selenium.

Racing Post blocks requests library (HTTP 406), so we use Selenium 
to simulate a real browser and bypass anti-bot protection.

Requires: selenium and chromedriver (or geckodriver for Firefox)

Install:
    pip install selenium
    pip install webdriver-manager  # Auto-downloads chromedriver

Usage:
    python scripts/scrape_odds_selenium.py --date 2025-12-28
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("ERROR: Selenium not installed")
    print("Install with: pip install selenium webdriver-manager")


def setup_driver(headless=False):
    """Setup Chrome driver with enhanced anti-detection options"""
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument('--headless=new')  # New headless mode
    
    # Enhanced anti-detection settings
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--start-maximized')
    
    # More realistic user agent (latest Chrome)
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
    
    # Additional preferences
    prefs = {
        "profile.default_content_setting_values.notifications": 2,
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Setup driver with auto-downloaded chromedriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Override multiple automation detection flags
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
        '''
    })
    
    return driver


def scrape_racing_post_odds(date_str, course=None):
    """
    Scrape odds from Racing Post for a given date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        course: Optional course name filter (e.g., 'cheltenham')
    
    Returns:
        List of dicts with race and odds data
    """
    if not SELENIUM_AVAILABLE:
        return None
    
    print(f"\nScraping Racing Post for {date_str}...")
    
    driver = None
    try:
        driver = setup_driver(headless=False)  # Visible for debugging
        
        # Navigate to Racing Post racecards
        # Format: https://www.racingpost.com/racecards/YYYY-MM-DD
        url = f"https://www.racingpost.com/racecards/{date_str}"
        print(f"Navigating to: {url}")
        
        driver.get(url)
        
        # Wait for page to load with randomized delay
        print("Waiting for page to load...")
        time.sleep(3)  # Initial load
        
        # Simulate human behavior - scroll page
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Check if we got blocked
        page_source = driver.page_source.lower()
        if any(term in page_source for term in ["access denied", "403 forbidden", "cloudflare", "captcha"]):
            print("❌ Access denied - Racing Post detected automation")
            print(f"Page title: {driver.title}")
            # Save for debugging
            with open("debug_racing_post.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Saved page source to debug_racing_post.html")
            return None
        
        # Wait for React to render - look for meeting items
        try:
            print("Waiting for React to render content...")
            # Wait for meetings list to appear
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "RC-meetingItem"))
            )
            print("✓ Page loaded successfully")
            time.sleep(2)  # Give React time to finish rendering
        except Exception as e:
            print(f"⚠️  Could not find race cards - may need manual intervention: {e}")
            # Save page source for debugging
            with open("debug_racing_post.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Saved page source to debug_racing_post.html")
            return None
        
        # Extract meeting items (each course/meeting)
        meetings = driver.find_elements(By.CLASS_NAME, "RC-meetingItem")
        print(f"\nFound {len(meetings)} meetings/courses racing today")
        
        # Get all racecard links - filter for actual race pages
        race_links_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/racecards/']")
        race_urls = []
        for link in race_links_elements:
            href = link.get_attribute('href')
            # Filter for actual race pages (not just date pages)
            if href and '/' in href.split('/racecards/')[-1] and href not in race_urls:
                race_urls.append(href)
        
        print(f"Found {len(race_urls)} unique race URLs")
        
        all_odds = []
        
        # Visit first 3 races to test
        for idx, race_url in enumerate(race_urls[:3], 1):
            print(f"\n[{idx}/{min(3, len(race_urls))}] Visiting: {race_url}")
            
            try:
                driver.get(race_url)
                time.sleep(3)  # Wait for page to load
                
                # Wait for race card to load
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "RC-headerBox__raceInfo"))
                    )
                except:
                    print("  ⚠️  Race card didn't load")
                    continue
                
                # Extract race info
                try:
                    race_info = driver.find_element(By.CLASS_NAME, "RC-headerBox__raceInfo").text
                    course_elem = driver.find_element(By.CLASS_NAME, "RC-headerBox__courseNamelink")
                    course_name = course_elem.text if course_elem else "Unknown"
                    
                    print(f"  Course: {course_name}")
                    print(f"  Info: {race_info}")
                except:
                    course_name = "Unknown"
                    race_info = ""
                
                # Find runners table
                try:
                    # Modern Racing Post uses different structure
                    # Try multiple possible selectors
                    runners = driver.find_elements(By.CSS_SELECTOR, "[data-test-id='runner-row'], .RC-runnerRow, tr.RC-runner")
                    
                    if not runners:
                        # Try table rows
                        table = driver.find_element(By.CLASS_NAME, "RC-sc__table")
                        runners = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
                    
                    print(f"  Found {len(runners)} runners")
                    
                    race_odds = {
                        'course': course_name,
                        'info': race_info,
                        'horses': []
                    }
                    
                    # Extract each horse and odds
                    for runner_idx, runner in enumerate(runners[:10], 1):  # First 10 horses
                        try:
                            # Try different selectors for horse name
                            horse_name = None
                            for selector in [".RC-runnerInfo__name", "[data-test-id='runner-name']", ".ui-link_selectionName"]:
                                try:
                                    horse_name = runner.find_element(By.CSS_SELECTOR, selector).text
                                    break
                                except:
                                    continue
                            
                            if not horse_name:
                                horse_name = f"Horse {runner_idx}"
                            
                            # Try to find odds
                            odds = "N/A"
                            for selector in [".RC-odds__fractional", ".RC-odds__decimal", "[data-test-id='odds']", ".ui-link_oddsLink"]:
                                try:
                                    odds_elem = runner.find_element(By.CSS_SELECTOR, selector)
                                    odds = odds_elem.text
                                    break
                                except:
                                    continue
                            
                            race_odds['horses'].append({
                                'horse': horse_name,
                                'odds': odds
                            })
                            
                            print(f"    {horse_name}: {odds}")
                        
                        except Exception as e:
                            continue
                    
                    if race_odds['horses']:
                        all_odds.append(race_odds)
                
                except Exception as e:
                    print(f"  ❌ Could not extract runners: {e}")
                    continue
            
            except Exception as e:
                print(f"  ❌ Error loading race: {e}")
                continue
        
        return all_odds
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None
    
    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()


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
    parser = argparse.ArgumentParser(description="Scrape Racing Post odds using Selenium")
    parser.add_argument('--date', type=str, required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--course', type=str, help='Filter by course name (optional)')
    
    args = parser.parse_args()
    
    if not SELENIUM_AVAILABLE:
        print("\n❌ Cannot run without Selenium")
        print("\nInstall requirements:")
        print("  pip install selenium webdriver-manager")
        return 1
    
    # Validate date
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("Expected: YYYY-MM-DD")
        return 1
    
    print("="*60)
    print("RACING POST ODDS SCRAPER (SELENIUM)")
    print("="*60)
    print("\nNOTE: This will open a Chrome window to bypass bot detection.")
    print("Racing Post may still block automated access.\n")
    
    odds_data = scrape_racing_post_odds(args.date, args.course)
    
    if odds_data:
        save_odds(odds_data, args.date)
        print("\n✅ SUCCESS - Odds scraped successfully!")
        return 0
    else:
        print("\n❌ FAILED - Could not scrape odds")
        print("\nPossible issues:")
        print("  - Racing Post detected automation")
        print("  - Page structure changed")
        print("  - No races on this date")
        print("\nCheck debug_racing_post.html for page content")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
