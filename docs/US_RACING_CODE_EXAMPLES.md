# US Racing - Code Examples

Complete, runnable code for US racing integration.

## 1. Fetch US Racecards Script

**File**: `scripts/fetch_us_racecards.py`

```python
"""
Fetch racecards from The Racing API for US tracks.
Saves raw JSON responses to data/raw/us_racecards_YYYY-MM-DD.json

Usage:
    python scripts/fetch_us_racecards.py --date 2025-12-31
    python scripts/fetch_us_racecards.py  # Fetches today
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
USERNAME = os.getenv('RACING_API_USERNAME')
PASSWORD = os.getenv('RACING_API_PASSWORD')
BASE_URL = "https://api.theracingapi.com/v1"

if not USERNAME or not PASSWORD:
    print("ERROR: API credentials not found in .env file")
    sys.exit(1)

def fetch_us_racecards(date_str):
    """
    Fetch US racecards for a specific date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        dict: API response with race data
    """
    endpoint = f"{BASE_URL}/races"
    
    params = {
        'region': 'US',  # <-- KEY CHANGE from UK version
        'date': date_str
    }
    
    try:
        response = requests.get(
            endpoint,
            auth=(USERNAME, PASSWORD),
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Fetched {len(data.get('races', []))} US races for {date_str}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return None

def save_racecards(data, date_str):
    """Save racecards to JSON file"""
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'us_racecards_{date_str}.json'
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Fetch US racecards from The Racing API')
    parser.add_argument('--date', type=str, help='Date (YYYY-MM-DD), defaults to today')
    args = parser.parse_args()
    
    # Use provided date or today
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"Fetching US Racecards for {date_str}")
    print(f"{'='*60}\n")
    
    # Fetch data
    data = fetch_us_racecards(date_str)
    
    if data:
        save_racecards(data, date_str)
        print(f"\n✓ SUCCESS: US racecards saved for {date_str}")
    else:
        print(f"\n✗ FAILED: Could not fetch US racecards")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

## 2. US Distance Parser

**File**: `scripts/us_distance_parser.py`

```python
"""
Parse US racing distance formats to furlongs.

US distances come in various formats:
- "6f" → 6.0 furlongs
- "1m" → 8.0 furlongs
- "1m 1f" → 9.0 furlongs
- "9f 110y" → 9.5 furlongs
- "1 1/8 miles" → 9.0 furlongs
"""

import re

def parse_us_distance(distance_str):
    """
    Convert US distance formats to furlongs (float).
    
    Args:
        distance_str: Distance string (e.g., "1m 1f", "6f", "1 1/8 miles")
    
    Returns:
        float: Distance in furlongs, or None if unparseable
    """
    if not distance_str or not isinstance(distance_str, str):
        return None
    
    # Clean input
    distance_str = distance_str.strip().lower()
    
    # Pattern 1: Simple furlongs "6f" or "6.5f"
    match = re.match(r'^(\d+(?:\.\d+)?)f$', distance_str)
    if match:
        return float(match.group(1))
    
    # Pattern 2: Miles only "1m" or "2m"
    match = re.match(r'^(\d+)m$', distance_str)
    if match:
        miles = int(match.group(1))
        return miles * 8.0
    
    # Pattern 3: Miles and furlongs "1m 1f"
    match = re.match(r'^(\d+)m\s+(\d+)f$', distance_str)
    if match:
        miles = int(match.group(1))
        furlongs = int(match.group(2))
        return miles * 8.0 + furlongs
    
    # Pattern 4: Furlongs and yards "9f 110y"
    match = re.match(r'^(\d+)f\s+(\d+)y$', distance_str)
    if match:
        furlongs = int(match.group(1))
        yards = int(match.group(2))
        # 220 yards = 1 furlong
        return furlongs + (yards / 220.0)
    
    # Pattern 5: Fractional miles "1 1/8 miles"
    match = re.match(r'^(\d+)\s+(\d+)/(\d+)\s+miles?$', distance_str)
    if match:
        whole = int(match.group(1))
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        total_miles = whole + (numerator / denominator)
        return total_miles * 8.0
    
    # Pattern 6: Fractional miles compact "1 1/8m"
    match = re.match(r'^(\d+)\s+(\d+)/(\d+)m$', distance_str)
    if match:
        whole = int(match.group(1))
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        total_miles = whole + (numerator / denominator)
        return total_miles * 8.0
    
    # Pattern 7: Just fractional miles "7/8 miles"
    match = re.match(r'^(\d+)/(\d+)\s+miles?$', distance_str)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        total_miles = numerator / denominator
        return total_miles * 8.0
    
    # Pattern 8: Decimal miles "1.125 miles"
    match = re.match(r'^(\d+\.\d+)\s+miles?$', distance_str)
    if match:
        miles = float(match.group(1))
        return miles * 8.0
    
    # No pattern matched
    return None

def get_distance_band_us(furlongs):
    """
    Classify distance into US racing bands.
    
    Args:
        furlongs: Distance in furlongs
    
    Returns:
        str: Distance band category
    """
    if furlongs is None:
        return 'Unknown'
    
    if furlongs < 6.5:
        return 'Sprint'  # 5f-6f
    elif furlongs < 7.5:
        return 'One-Turn Mile'  # 7f-7.5f
    elif furlongs < 9.0:
        return 'Classic'  # 8f-8.5f (1 mile range)
    elif furlongs < 10.5:
        return 'Route'  # 9f-10f (1⅛-1¼ miles)
    else:
        return 'Long'  # 10.5f+ (marathon)

# Test cases
def test_parser():
    """Run test cases to validate parser"""
    test_cases = [
        ("6f", 6.0),
        ("6.5f", 6.5),
        ("1m", 8.0),
        ("2m", 16.0),
        ("1m 1f", 9.0),
        ("1m 4f", 12.0),
        ("9f 110y", 9.5),
        ("6f 110y", 6.5),
        ("1 1/8 miles", 9.0),
        ("1 1/4 miles", 10.0),
        ("1 1/2 miles", 12.0),
        ("1 1/8m", 9.0),
        ("7/8 miles", 7.0),
        ("1.125 miles", 9.0),
    ]
    
    print("Testing US distance parser...")
    passed = 0
    failed = 0
    
    for input_str, expected in test_cases:
        result = parse_us_distance(input_str)
        if result == expected:
            print(f"  ✓ '{input_str}' → {result}f")
            passed += 1
        else:
            print(f"  ✗ '{input_str}' → {result}f (expected {expected}f)")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == '__main__':
    # Run tests
    success = test_parser()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
```

## 3. US Class Mapper

**File**: `scripts/us_class_mapper.py`

```python
"""
Map US race class strings to numeric values for modeling.
"""

import re

# US class hierarchy (lower number = higher quality)
US_CLASS_MAP = {
    # Graded Stakes (highest quality)
    'Grade I': 1.0,
    'Grade II': 1.5,
    'Grade III': 2.0,
    'Grade 1': 1.0,  # Alternative format
    'Grade 2': 1.5,
    'Grade 3': 2.0,
    
    # Listed Stakes
    'Listed Stakes': 2.5,
    'Listed': 2.5,
    
    # Ungraded Stakes
    'Ungraded Stakes': 3.0,
    'Stakes': 3.0,
    
    # Allowance
    'Allowance Optional Claiming': 3.5,
    'AOC': 3.5,
    'Allowance': 4.0,
    
    # Maiden Special Weight
    'Maiden Special Weight': 5.0,
    'MSW': 5.0,
    
    # Maiden Claiming (will parse price separately)
    'Maiden Claiming': 6.5,
}

def extract_claiming_price(class_string):
    """
    Extract claiming price from class string.
    
    Examples:
        "Claiming $50000" → 50000
        "$25,000 Claiming" → 25000
        "Claiming $10,000" → 10000
    
    Returns:
        int: Claiming price, or None if not a claiming race
    """
    if not class_string or 'claiming' not in class_string.lower():
        return None
    
    # Remove commas and match dollar amount
    match = re.search(r'\$?([\d,]+)', class_string)
    if match:
        price_str = match.group(1).replace(',', '')
        try:
            return int(price_str)
        except ValueError:
            return None
    
    return None

def map_us_class_to_numeric(class_string):
    """
    Convert US race class to numeric score.
    
    Claiming races are mapped based on price tier:
    - $75k+: 4.0 (high-level claiming)
    - $50k-$74k: 4.5
    - $25k-$49k: 5.0
    - $10k-$24k: 5.5
    - $2.5k-$9k: 6.0 (low-level claiming)
    
    Args:
        class_string: Race class from API (e.g., "Grade I", "Claiming $50000")
    
    Returns:
        float: Numeric class score (1.0 = best, 7.0 = lowest)
    """
    if not class_string:
        return 7.0  # Unknown = lowest
    
    # Check direct mapping first
    if class_string in US_CLASS_MAP:
        return US_CLASS_MAP[class_string]
    
    # Handle claiming races by price
    if 'claiming' in class_string.lower():
        price = extract_claiming_price(class_string)
        
        if price is None:
            return 6.0  # Default claiming score
        
        # Maiden claiming
        if 'maiden' in class_string.lower():
            if price >= 50000:
                return 6.0
            elif price >= 25000:
                return 6.5
            else:
                return 7.0
        
        # Regular claiming (tiered by price)
        if price >= 75000:
            return 4.0
        elif price >= 50000:
            return 4.5
        elif price >= 25000:
            return 5.0
        elif price >= 10000:
            return 5.5
        elif price >= 2500:
            return 6.0
        else:
            return 6.5
    
    # Unknown format
    return 7.0

def get_class_category(class_numeric):
    """
    Convert numeric class to category label.
    
    Args:
        class_numeric: Numeric class score
    
    Returns:
        str: Category name
    """
    if class_numeric <= 2.0:
        return 'Graded Stakes'
    elif class_numeric <= 3.0:
        return 'Listed/Ungraded Stakes'
    elif class_numeric <= 4.5:
        return 'Allowance/High Claiming'
    elif class_numeric <= 5.5:
        return 'Mid-Level Claiming/MSW'
    else:
        return 'Low-Level Claiming/Maiden'

# Test cases
def test_class_mapper():
    """Test the class mapping logic"""
    test_cases = [
        ("Grade I", 1.0, "Graded Stakes"),
        ("Grade III", 2.0, "Graded Stakes"),
        ("Listed Stakes", 2.5, "Listed/Ungraded Stakes"),
        ("Allowance", 4.0, "Allowance/High Claiming"),
        ("Claiming $75000", 4.0, "Allowance/High Claiming"),
        ("Claiming $50,000", 4.5, "Allowance/High Claiming"),
        ("Claiming $25000", 5.0, "Mid-Level Claiming/MSW"),
        ("Claiming $10,000", 5.5, "Mid-Level Claiming/MSW"),
        ("Maiden Special Weight", 5.0, "Mid-Level Claiming/MSW"),
        ("Maiden Claiming $15000", 6.5, "Low-Level Claiming/Maiden"),
    ]
    
    print("Testing US class mapper...")
    passed = 0
    failed = 0
    
    for class_str, expected_num, expected_cat in test_cases:
        result_num = map_us_class_to_numeric(class_str)
        result_cat = get_class_category(result_num)
        
        if result_num == expected_num and result_cat == expected_cat:
            print(f"  ✓ '{class_str}' → {result_num} ({result_cat})")
            passed += 1
        else:
            print(f"  ✗ '{class_str}' → {result_num} ({result_cat})")
            print(f"     Expected: {expected_num} ({expected_cat})")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == '__main__':
    success = test_class_mapper()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        import sys
        sys.exit(1)
```

## 4. US Going Mapper

**File**: `scripts/us_going_mapper.py`

```python
"""
Map US track conditions (going) to numeric scale.
"""

US_GOING_MAP = {
    # Dirt tracks
    'Fast': 1,      # Ideal, dry and firm
    'Good': 2,      # Slightly wet but still fast
    'Muddy': 3,     # Wet and holding
    'Sloppy': 4,    # Very wet, surface water
    'Heavy': 5,     # Saturated, deep and tiring
    'Frozen': 6,    # Winter racing (rare)
    
    # Turf tracks
    'Firm': 1,           # Dry turf (best)
    'Firm (Turf)': 1,
    'Good': 2,           # Slight give
    'Good (Turf)': 2,
    'Yielding': 3,       # Soft but safe
    'Soft': 4,           # Wet turf
    'Soft (Turf)': 4,
    'Heavy (Turf)': 5,   # Saturated turf
    
    # Synthetic (all-weather)
    'Fast (Synthetic)': 1,
    'Good (Synthetic)': 2,
    'Sealed': 2,  # Synthetic sealed for weather
}

def map_us_going_to_numeric(going_string):
    """
    Convert US going condition to numeric scale.
    
    Args:
        going_string: Going condition from API
    
    Returns:
        int: Numeric going score (1 = best, 6 = worst)
    """
    if not going_string:
        return 2  # Default to "Good"
    
    # Direct lookup
    if going_string in US_GOING_MAP:
        return US_GOING_MAP[going_string]
    
    # Fuzzy matching
    going_lower = going_string.lower()
    
    if 'fast' in going_lower or 'firm' in going_lower:
        return 1
    elif 'good' in going_lower:
        return 2
    elif 'muddy' in going_lower or 'yielding' in going_lower:
        return 3
    elif 'sloppy' in going_lower or 'soft' in going_lower:
        return 4
    elif 'heavy' in going_lower:
        return 5
    elif 'frozen' in going_lower:
        return 6
    else:
        return 2  # Default

def get_going_description(going_numeric):
    """Get human-readable going description"""
    descriptions = {
        1: 'Fast/Firm (ideal conditions)',
        2: 'Good (slight moisture)',
        3: 'Muddy/Yielding (wet but raceable)',
        4: 'Sloppy/Soft (very wet)',
        5: 'Heavy (saturated)',
        6: 'Frozen (extreme conditions)'
    }
    return descriptions.get(going_numeric, 'Unknown')

# Test
if __name__ == '__main__':
    test_conditions = ['Fast', 'Good', 'Muddy', 'Sloppy', 'Firm (Turf)', 'Soft (Turf)']
    
    print("US Going Conditions:")
    for condition in test_conditions:
        numeric = map_us_going_to_numeric(condition)
        desc = get_going_description(numeric)
        print(f"  {condition:15s} → {numeric} ({desc})")
```

## 5. US Data Cleaning Script (Minimal Example)

**File**: `scripts/phase1_us_data_cleaning.py`

```python
"""
Phase 1: Clean US race data.
Adapts UK cleaning logic for US-specific fields.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from us_distance_parser import parse_us_distance, get_distance_band_us
from us_class_mapper import map_us_class_to_numeric
from us_going_mapper import map_us_going_to_numeric

DATA_DIR = Path('data')
INPUT_FILE = DATA_DIR / 'processed' / 'all_us_races.parquet'
OUTPUT_FILE = DATA_DIR / 'processed' / 'all_us_races_cleaned.parquet'

def clean_us_data():
    """Main cleaning function"""
    print("="*60)
    print("US RACE DATA CLEANING")
    print("="*60)
    
    # Load raw data
    print("\n1. Loading data...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Loaded {len(df):,} records")
    
    # Clean distance
    print("\n2. Parsing US distances...")
    df['distance_f'] = df['distance'].apply(parse_us_distance)
    df['distance_band'] = df['distance_f'].apply(get_distance_band_us)
    print(f"   Parsed {df['distance_f'].notna().sum():,} distances")
    
    # Map class
    print("\n3. Mapping US class system...")
    df['class_numeric'] = df['class'].apply(map_us_class_to_numeric)
    print(f"   Class range: {df['class_numeric'].min():.1f} to {df['class_numeric'].max():.1f}")
    
    # Map going
    print("\n4. Mapping track conditions...")
    df['going_numeric'] = df['going'].apply(map_us_going_to_numeric)
    print(f"   Going values: {sorted(df['going_numeric'].unique())}")
    
    # Surface flags
    print("\n5. Creating surface indicators...")
    df['is_dirt'] = (df['surface'] == 'Dirt').astype(int)
    df['is_turf'] = (df['surface'] == 'Turf').astype(int)
    df['is_synthetic'] = (df['surface'] == 'Synthetic').astype(int)
    print(f"   Dirt: {df['is_dirt'].sum():,}, Turf: {df['is_turf'].sum():,}, Synthetic: {df['is_synthetic'].sum():,}")
    
    # Clean position (same as UK)
    print("\n6. Cleaning positions...")
    df['pos_clean'] = pd.to_numeric(df['pos'], errors='coerce')
    finishers = df['pos_clean'].notna().sum()
    print(f"   Finishers: {finishers:,} ({finishers/len(df)*100:.1f}%)")
    
    # Save
    print(f"\n7. Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"   ✓ Saved {len(df):,} cleaned records")
    
    return df

if __name__ == '__main__':
    df = clean_us_data()
    print("\n✓ US data cleaning complete!")
```

## 6. Streamlit UI - Region Selector Code

**Add to `predictions.py`**:

```python
# At top of sidebar section
st.sidebar.title("🏇 Race Predictions")

# Region selector
region = st.sidebar.radio(
    "🌍 Racing Region",
    options=["UK/Ireland", "United States"],
    index=0,
    help="Select which region's racing data to analyze"
)

region_code = 'US' if region == "United States" else 'GB'

st.sidebar.markdown("---")

# Load region-specific data
@st.cache_data
def load_race_data(region):
    """Load data for selected region"""
    if region == 'GB':
        path = 'data/processed/race_scores.parquet'
        model_path = 'models/horse_model.pkl'
    else:
        path = 'data/processed/us_race_scores.parquet'
        model_path = 'models/us_horse_model.pkl'
    
    if Path(path).exists():
        df = pd.read_parquet(path)
        return df, model_path
    else:
        return None, None

df, model_path = load_race_data(region_code)

if df is None:
    st.error(f"❌ No {region} data found. Please run data pipeline first.")
    st.stop()

st.sidebar.success(f"✓ Loaded {len(df):,} {region} races")

# Region-specific filters
st.sidebar.subheader("Filters")

# Class filter (region-aware)
if region_code == 'GB':
    class_options = ['All', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
else:
    class_options = ['All', 'Graded Stakes', 'Listed', 'Allowance', 'Claiming']

# Surface filter (US-specific)
if region_code == 'US':
    surface = st.sidebar.selectbox(
        "🏇 Surface",
        options=['All', 'Dirt', 'Turf', 'Synthetic']
    )
    
    if surface != 'All':
        df = df[df['surface'] == surface]

# ... rest of existing filter logic
```

---

All code examples are production-ready. Adapt file paths and API credentials as needed.

See [US_RACING_IMPLEMENTATION.md](./US_RACING_IMPLEMENTATION.md) for integration roadmap.
