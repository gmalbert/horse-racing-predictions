"""
Example script for fetching data from The Racing API.
This demonstrates proper authentication and API usage patterns.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
USERNAME = os.getenv('RACING_API_USERNAME')
PASSWORD = os.getenv('RACING_API_PASSWORD')

if not USERNAME or not PASSWORD:
    raise ValueError("API credentials not found. Check your .env file.")

BASE_URL = "https://api.theracingapi.com/v1"


def fetch_races(region=None, date=None):
    """
    Fetch race data from The Racing API.
    
    Args:
        region: Optional region filter (e.g., 'GB', 'IE', 'US')
        date: Optional date filter (format: YYYY-MM-DD)
    
    Returns:
        Response JSON data
    """
    endpoint = f"{BASE_URL}/races"
    params = {}
    
    if region:
        params['region'] = region
    if date:
        params['date'] = date
    
    try:
        response = requests.get(
            endpoint,
            auth=(USERNAME, PASSWORD),
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching races: {e}")
        return None


if __name__ == "__main__":
    # Example: Fetch today's races
    print("Fetching race data...")
    races = fetch_races()
    
    if races:
        print(f"Successfully fetched {len(races.get('races', []))} races")
    else:
        print("Failed to fetch race data")
