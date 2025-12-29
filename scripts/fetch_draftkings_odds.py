"""
Fetch horse racing odds from DraftKings unofficial API.

DraftKings exposes racing data through their internal API endpoints.
This script discovers and fetches odds for today's UK horse races.

API Rate Limit: Unknown - use responsibly
Cost: Free (no API key required)
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class DraftKingsOddsFetcher:
    """Fetch horse racing odds from DraftKings API."""
    
    def __init__(self):
        # DraftKings has separate horse racing site: dkhorse.com
        self.base_url = "https://www.dkhorse.com"
        self.sportsbook_url = "https://sportsbook.draftkings.com"
        
        # Try multiple API base URLs
        self.api_bases = [
            "https://www.dkhorse.com/api",
            "https://api.dkhorse.com",
            "https://sportsbook-us-ny.draftkings.com/sites/US-NY-SB/api/v5",
            "https://sportsbook.draftkings.com/api",
        ]
        
        # Common headers to mimic browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Origin': 'https://www.dkhorse.com',
            'Referer': 'https://www.dkhorse.com/',
            'Connection': 'keep-alive',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_racing_events(self, date: Optional[str] = None) -> Dict:
        """
        Fetch racing events for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
        
        Returns:
            Dict containing racing events
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Try different endpoint patterns for horse racing
        endpoints_to_try = [
            # DK Horse specific endpoints
            "https://www.dkhorse.com/api/racecards",
            "https://www.dkhorse.com/api/races/today",
            "https://www.dkhorse.com/api/events",
            f"https://www.dkhorse.com/api/races/{date}",
            
            # Generic sportsbook endpoints
            "https://sportsbook.draftkings.com/api/sportscontent/dkus/v1/leagues/horse-racing",
            "https://sportsbook-us-ny.draftkings.com/api/sportscontent/dkusnj/v1/leagues/horse-racing",
            
            # Event group endpoints (horse racing might be event group ID)
            "https://sportsbook-us-ny.draftkings.com/sites/US-NY-SB/api/v5/eventgroups",
        ]
        
        for endpoint in endpoints_to_try:
            try:
                print(f"Trying: {endpoint}")
                response = self.session.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ SUCCESS! ({len(json.dumps(data))} bytes)")
                    
                    # Save raw response for inspection
                    self._save_response(data, f"draftkings_response_{int(time.time())}.json")
                    return data
                else:
                    print(f"  HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  Error: {str(e)[:50]}")
                continue
        
        return {}
    
    def get_race_odds(self, event_id: str) -> Dict:
        """
        Fetch odds for a specific race.
        
        Args:
            event_id: DraftKings event ID
        
        Returns:
            Dict containing race odds
        """
        endpoint = f"{self.api_base}/events/{event_id}"
        
        try:
            response = self.session.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to fetch event {event_id}: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error fetching event {event_id}: {e}")
            return {}
    
    def search_racing_markets(self) -> Dict:
        """Search for horse racing markets using DraftKings search/discovery."""
        
        # Try the leagues endpoint
        endpoints = [
            "https://sportsbook-us-ny.draftkings.com/sites/US-NY-SB/api/v5/leagues",
            "https://sportsbook-us-ny.draftkings.com/sites/US-NY-SB/api/v5/eventgroups",
        ]
        
        for endpoint in endpoints:
            try:
                print(f"\nSearching: {endpoint}")
                response = self.session.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Found data ({len(json.dumps(data))} bytes)")
                    
                    # Save for inspection
                    filename = f"draftkings_search_{endpoint.split('/')[-1]}.json"
                    self._save_response(data, filename)
                    
                    return data
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        return {}
    
    def _save_response(self, data: Dict, filename: str):
        """Save API response to data/raw for inspection."""
        output_dir = "data/raw"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVED] {filepath}")
    
    def extract_uk_races(self, data: Dict) -> List[Dict]:
        """
        Extract UK horse races from API response.
        
        Args:
            data: API response data
        
        Returns:
            List of UK race dictionaries with odds
        """
        races = []
        
        # This will need to be adapted based on actual API structure
        # Placeholder for now - will update after seeing real response
        
        return races


def main():
    """Test DraftKings odds fetcher."""
    
    print("=" * 60)
    print("DRAFTKINGS ODDS FETCHER - TEST")
    print("=" * 60)
    print()
    
    fetcher = DraftKingsOddsFetcher()
    
    # Try to discover racing markets
    print("Step 1: Searching for racing markets...")
    markets = fetcher.search_racing_markets()
    
    print("\nStep 2: Fetching today's racing events...")
    events = fetcher.get_racing_events()
    
    if events:
        print(f"\n✓ Successfully fetched data from DraftKings!")
        print(f"  Response keys: {list(events.keys())}")
        
        # Try to extract races
        races = fetcher.extract_uk_races(events)
        if races:
            print(f"\n✓ Found {len(races)} UK races")
            for race in races[:5]:  # Show first 5
                print(f"  - {race.get('course', 'Unknown')} at {race.get('time', 'Unknown')}")
        else:
            print("\n⚠️  No UK races found (may need to parse different structure)")
    else:
        print("\n❌ Could not fetch racing data")
        print("   DraftKings may require:")
        print("   - Geo-location (US only)")
        print("   - Cookies/session from actual browser")
        print("   - Different API endpoints")
    
    print("\n" + "=" * 60)
    print("Check data/raw/ for saved responses")
    print("=" * 60)


if __name__ == "__main__":
    main()
