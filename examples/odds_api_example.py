"""
Example script for fetching data from The Odds API.
This demonstrates proper authentication and API usage patterns.

Note: The Odds API provides LIVE odds only - no historical data available.
Limited to 500 API calls per month.
"""

import os
import requests
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
API_KEY = os.getenv('ODDS_API_KEY')

if not API_KEY:
    raise ValueError("ODDS_API_KEY not found. Check your .env file.")

BASE_URL = "https://api.the-odds-api.com/v4"


def get_sports():
    """
    Fetch list of available sports.
    This is useful to find the correct sport key for horse racing.
    
    Returns:
        List of available sports with their keys
    """
    endpoint = f"{BASE_URL}/sports"
    
    try:
        response = requests.get(
            endpoint,
            params={'apiKey': API_KEY},
            timeout=10
        )
        response.raise_for_status()
        
        # Check remaining requests in response headers
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        print(f"API Requests - Used: {used}, Remaining: {remaining}")
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sports: {e}")
        return None


def get_odds(sport_key='horse_racing_uk', regions='uk', markets='h2h'):
    """
    Fetch current odds for a specific sport.
    
    Args:
        sport_key: Sport identifier (e.g., 'horse_racing_uk', 'horse_racing_us')
        regions: Regions for odds (e.g., 'uk', 'us', 'au')
        markets: Betting markets (e.g., 'h2h' for head-to-head, 'spreads', 'totals')
    
    Returns:
        Response JSON data with current odds
    """
    endpoint = f"{BASE_URL}/sports/{sport_key}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': regions,
        'markets': markets,
        'oddsFormat': 'decimal',  # or 'american'
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(
            endpoint,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        # Check remaining requests in response headers
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        print(f"API Requests - Used: {used}, Remaining: {remaining}")
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def display_odds_summary(odds_data):
    """
    Display a summary of the odds data.
    
    Args:
        odds_data: JSON response from the odds API
    """
    if not odds_data:
        print("No odds data to display")
        return
    
    print(f"\n=== Odds Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"Total events: {len(odds_data)}\n")
    
    for event in odds_data[:5]:  # Show first 5 events
        print(f"Event: {event.get('home_team')} vs {event.get('away_team')}")
        print(f"Start time: {event.get('commence_time')}")
        print(f"Bookmakers: {len(event.get('bookmakers', []))}")
        
        # Show sample odds from first bookmaker
        if event.get('bookmakers'):
            bookmaker = event['bookmakers'][0]
            print(f"Sample odds ({bookmaker['key']}):")
            for market in bookmaker.get('markets', []):
                print(f"  Market: {market['key']}")
                for outcome in market.get('outcomes', []):
                    print(f"    {outcome['name']}: {outcome['price']}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("The Odds API - Example Usage")
    print("=" * 60)
    
    # Example 1: Get list of available sports
    print("\n1. Fetching available sports...")
    sports = get_sports()
    
    if sports:
        print(f"Found {len(sports)} sports available")
        # Look for horse racing sports
        horse_racing_sports = [s for s in sports if 'horse_racing' in s.get('key', '')]
        if horse_racing_sports:
            print("\nHorse racing sports available:")
            for sport in horse_racing_sports:
                print(f"  - {sport['key']}: {sport['title']}")
        else:
            print("\nNote: Horse racing may not be available in The Odds API")
            print("Check documentation for supported sports")
    
    # Example 2: Get odds for a specific sport
    # Note: Replace 'horse_racing_uk' with actual sport key from available sports
    print("\n2. Fetching odds (Note: horse racing may not be available)...")
    print("To use this API, first verify horse racing is available in the sports list")
    
    # Uncomment below if horse racing is available:
    # odds = get_odds('horse_racing_uk')
    # display_odds_summary(odds)
    
    print("\n" + "=" * 60)
    print("IMPORTANT REMINDERS:")
    print("- Limited to 500 API calls per month")
    print("- No historical data available (live odds only)")
    print("- Always check x-requests-remaining header")
    print("- Implement caching to minimize API calls")
    print("=" * 60)
