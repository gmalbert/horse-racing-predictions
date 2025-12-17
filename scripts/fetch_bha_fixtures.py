"""
Fetch upcoming race fixtures from British Horseracing Authority or Racing Post.

This script scrapes upcoming race fixtures for selected UK racecourses and saves
the results to data/raw/ as CSV and JSON files.

IMPORTANT NOTES:
1. Web scraping may be subject to the website's terms of service and rate limits.
2. Many racing sites actively block scrapers. Consider using The Racing API instead.
3. The Racing API has upcoming fixtures: https://api.theracingapi.com/v1/racecards
   (requires authentication - see examples/api_example.py)
4. If scraping, you may need to:
   - Find the correct current URL (sites change frequently)
   - Use selenium/playwright for JavaScript-heavy pages
   - Add delays between requests
   - Respect robots.txt

This script provides a template. Update FIXTURES_URL with a working endpoint.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Configuration
TARGET_TRACKS = ["Ascot", "Newmarket", "Doncaster", "York"]
FIXTURES_URL = "https://www.britishhorseracing.com/racing/fixtures/upcoming/"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# User agent to avoid blocking - more complete headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}


def fetch_fixtures(url: str, tracks: list) -> list:
    """
    Fetch upcoming race fixtures from the Racing Post fixtures page.
    
    Args:
        url: URL of the fixtures page
        tracks: List of track names to filter
        
    Returns:
        List of dicts containing fixture data
    """
    print(f"Fetching fixtures from {url}...")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []
    
    # Check if page uses JavaScript rendering (AngularJS, React, etc.)
    if 'ng-app' in response.text or 'React' in response.text[:5000]:
        print("\nWARNING: This page uses JavaScript rendering (AngularJS detected).")
        print("BeautifulSoup cannot parse dynamically loaded content.")
        print("Recommended solutions:")
        print("1. Use Selenium or Playwright for browser automation")
        print("2. Look for an API endpoint (check Network tab in browser DevTools)")
        print("3. Use The Racing API: https://api.theracingapi.com/v1/racecards")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    data = []
    
    # NOTE: The HTML structure may change. Adjust selectors as needed.
    # This is a template; you may need to inspect the actual page structure.
    
    # Try multiple possible selectors for fixture cards
    fixture_containers = (
        soup.find_all("div", class_="fixture-card") or
        soup.find_all("div", class_=lambda x: x and "fixture" in x.lower()) or
        soup.find_all("article") or
        soup.find_all("li", class_=lambda x: x and "meeting" in x.lower())
    )
    
    if not fixture_containers:
        print("Warning: No fixture containers found. The page structure may have changed.")
        print("Inspect the HTML manually and update the selectors.")
        return []
    
    for fixture in fixture_containers:
        # Try to extract track name
        track_elem = (
            fixture.find("h3") or
            fixture.find("h2") or
            fixture.find(class_=lambda x: x and "course" in x.lower())
        )
        
        # Try to extract date
        date_elem = (
            fixture.find("span", class_="date") or
            fixture.find("time") or
            fixture.find(class_=lambda x: x and "date" in x.lower())
        )
        
        if not track_elem:
            continue
            
        track = track_elem.text.strip()
        date = date_elem.text.strip() if date_elem else "Unknown"
        
        # Filter by target tracks
        if any(t.lower() in track.lower() for t in tracks):
            data.append({
                "Track": track,
                "Date": date,
                "Scraped_At": datetime.now().isoformat()
            })
    
    print(f"Found {len(data)} fixtures for target tracks.")
    return data


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
    csv_path = output_dir / f"bha_fixtures_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = output_dir / f"bha_fixtures_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")
    
    # Display preview
    print("\nPreview of fetched data:")
    print(df.head(10))


def main():
    """Main execution function."""
    print("=" * 60)
    print("BHA/Racing Post Fixtures Scraper")
    print("=" * 60)
    print(f"Target tracks: {', '.join(TARGET_TRACKS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Fetch fixtures
    fixtures = fetch_fixtures(FIXTURES_URL, TARGET_TRACKS)
    
    if not fixtures:
        print("\nNo fixtures found. Possible reasons:")
        print("- The website structure has changed (update selectors)")
        print("- Rate limiting or blocking (check headers/user-agent)")
        print("- No upcoming fixtures for target tracks")
        return
    
    # Save results
    save_fixtures(fixtures, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
