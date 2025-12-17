"""
Fetch upcoming race fixtures from British Horseracing Authority using Selenium.

This script uses Selenium WebDriver to handle JavaScript-rendered content from the
BHA fixtures page. It waits for the page to load, extracts fixture data, and saves
to data/raw/ as CSV and JSON files.

Requirements:
- selenium
- Chrome browser installed
- chromedriver (will be auto-downloaded by selenium 4.6+)

Install: pip install selenium
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Configuration
TARGET_TRACKS = ["Ascot", "Newmarket", "Doncaster", "York"]
FIXTURES_URL = "https://www.britishhorseracing.com/racing/fixtures/upcoming/"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selenium settings
HEADLESS = True  # Set to False to see the browser
PAGE_LOAD_TIMEOUT = 30
IMPLICIT_WAIT = 10


def setup_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Set up Chrome WebDriver with options.
    
    Args:
        headless: Run browser in headless mode (no GUI)
        
    Returns:
        Configured Chrome WebDriver instance
    """
    print("Setting up Chrome WebDriver...")
    
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless=new")
    
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Selenium 4.6+ automatically downloads chromedriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.implicitly_wait(IMPLICIT_WAIT)
    
    return driver


def fetch_fixtures_selenium(url: str, tracks: list) -> list:
    """
    Fetch fixtures using Selenium to handle JavaScript rendering.
    
    Args:
        url: URL of the fixtures page
        tracks: List of track names to filter
        
    Returns:
        List of dicts containing fixture data
    """
    print(f"Fetching fixtures from {url}...")
    
    driver = None
    try:
        driver = setup_driver(headless=HEADLESS)
        driver.get(url)
        
        print("Waiting for page to load...")
        # Wait for AngularJS to finish rendering - longer wait for BHA
        time.sleep(5)  # Give AngularJS time to initialize and load data
        
        # Wait for body content to be present
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Additional wait for dynamic content
        time.sleep(3)
        
        # Common selectors for racing fixtures (BHA-specific and generic)
        possible_selectors = [
            "div.fixture",
            "div.meeting",
            "div.race-meeting",
            "div.fixture-item",
            "div.meeting-card",
            "div.race-card",
            "div[ng-repeat*='fixture']",
            "div[ng-repeat*='meeting']",
            "tr.fixture-row",
            "li.fixture",
            "li.meeting",
            ".fixtures-list .fixture",
            "[class*='fixture']",
            "[class*='meeting']",
            "article",
            "table tr",
        ]
        
        fixture_elements = []
        for selector in possible_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector: {selector}")
                    fixture_elements = elements
                    break
            except Exception:
                continue
        
        if not fixture_elements:
            print("Warning: No fixture elements found with known selectors.")
            print("\nSaving page source to debug.html for inspection...")
            debug_path = OUTPUT_DIR / "debug_page_source.html"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"Saved to: {debug_path}")
            
            print("\nPage source preview (first 1000 chars):")
            print(driver.page_source[:1000])
            
            print("\nTrying to extract any text containing track names...")
            
            # Fallback: search entire page for track names
            page_text = driver.find_element(By.TAG_NAME, "body").text
            data = []
            
            lines = page_text.split('\n')
            current_track = None
            current_date = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if line contains a track name
                for track in tracks:
                    if track.lower() in line.lower():
                        current_track = track
                        # Look at next few lines for date
                        for j in range(i, min(i+5, len(lines))):
                            next_line = lines[j].strip()
                            if any(month in next_line for month in ['January', 'February', 'March', 'April', 
                                                                      'May', 'June', 'July', 'August', 
                                                                      'September', 'October', 'November', 'December']) \
                               or any(day in next_line for day in ['Monday', 'Tuesday', 'Wednesday', 
                                                                     'Thursday', 'Friday', 'Saturday', 'Sunday']):
                                current_date = next_line
                                break
                        
                        data.append({
                            "Track": current_track,
                            "Date": current_date or "Unknown",
                            "Context": '\n'.join(lines[max(0,i-1):min(len(lines),i+4)]),
                            "Scraped_At": datetime.now().isoformat(),
                            "Source": "text_search"
                        })
                        current_date = None
                        break
            
            return data
        
        # Extract data from fixture elements
        data = []
        for element in fixture_elements:
            try:
                text = element.text.strip()
                
                # Try to find track name
                track = None
                for t in tracks:
                    if t.lower() in text.lower():
                        track = t
                        break
                
                if not track:
                    continue
                
                # Try to extract date (look for date patterns)
                date = "Unknown"
                lines = text.split('\n')
                for line in lines:
                    if any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                        date = line.strip()
                        break
                    elif any(day in line for day in ['Monday', 'Tuesday', 'Wednesday', 
                                                      'Thursday', 'Friday', 'Saturday', 'Sunday']):
                        date = line.strip()
                        break
                
                data.append({
                    "Track": track,
                    "Date": date,
                    "Raw_Text": text[:200],  # Save first 200 chars
                    "Scraped_At": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Error extracting data from element: {e}")
                continue
        
        print(f"Found {len(data)} fixtures for target tracks.")
        return data
        
    except Exception as e:
        print(f"Error during Selenium scraping: {e}")
        return []
        
    finally:
        if driver:
            driver.quit()
            print("Browser closed.")


def save_fixtures(data: list, output_dir: Path):
    """
    Save fixture data to CSV and JSON files.
    
    Args:
        data: List of fixture dicts
        output_dir: Directory to save files
    """
    if not data:
        print("No data to save.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / f"bha_fixtures_selenium_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = output_dir / f"bha_fixtures_selenium_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")
    
    # Display preview
    print("\nPreview of fetched data:")
    print(df.head(10))


def main():
    """Main execution function."""
    print("=" * 60)
    print("BHA Fixtures Scraper (Selenium)")
    print("=" * 60)
    print(f"Target tracks: {', '.join(TARGET_TRACKS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Headless mode: {HEADLESS}")
    print()
    
    # Fetch fixtures
    fixtures = fetch_fixtures_selenium(FIXTURES_URL, TARGET_TRACKS)
    
    if not fixtures:
        print("\nNo fixtures found.")
        return
    
    # Save results
    save_fixtures(fixtures, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
