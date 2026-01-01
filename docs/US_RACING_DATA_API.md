# US Racing - Data & API Integration

## The Racing API - US Region Support

The Racing API already supports US racing via the `region` parameter.

### API Endpoints (No Changes Needed)

```python
# Current UK approach
response = requests.get(
    "https://api.theracingapi.com/v1/races",
    auth=(USERNAME, PASSWORD),
    params={'region': 'GB', 'date': '2025-12-31'}
)

# US approach (just change region)
response = requests.get(
    "https://api.theracingapi.com/v1/races",
    auth=(USERNAME, PASSWORD),
    params={'region': 'US', 'date': '2025-12-31'}
)
```

### Expected US Data Fields

Based on The Racing API documentation, US races should return similar structure to UK:

```json
{
  "date": "2025-12-31",
  "course": "Churchill Downs",
  "time": "14:30",
  "race_name": "Woodford Reserve Stakes",
  "distance": "9f",
  "surface": "Dirt",
  "class": "Grade II",
  "going": "Fast",
  "prize": "200000",
  "horses": [
    {
      "name": "Secret Weapon",
      "jockey": "I. Ortiz Jr.",
      "trainer": "B. Mott",
      "weight": "126",
      "age": "4",
      "sex": "C",
      "draw": "5",
      "odds": "5/2",
      "official_rating": null,
      "position": "1"
    }
  ]
}
```

### Key Differences to Handle

| Field | UK Format | US Format | Action Required |
|-------|-----------|-----------|-----------------|
| **distance** | `"6f"`, `"12.5f"` | `"6f"`, `"1m 1f"`, `"9f 110y"` | Parse mixed formats |
| **class** | `"Class 1"` | `"Grade I"`, `"$50000 Claiming"` | Create US class mapper |
| **surface** | `"Turf"`, `"AW"` | `"Dirt"`, `"Turf"`, `"Synthetic"` | Expand surface types |
| **going** | `"Good to Firm"` | `"Fast"`, `"Sloppy"` | Create US going mapper |
| **course** | Standard names | May include state `"Churchill Downs (KY)"` | Standardize |
| **official_rating** | BHA ratings (60-130) | Often `null` (US uses speed figures instead) | Handle missing values |

## Data Fetching Strategy

### Option 1: Separate Scripts (Recommended)

Clone UK fetching scripts and adapt for US:

```
scripts/
  fetch_us_racecards.py          # Clone of fetch_racecards.py with region='US'
  combine_us_races.py            # Clone of combine_gb_flat_races.py
  phase1_us_data_cleaning.py     # US-specific cleaning logic
```

**Pros**: Clear separation, no risk of breaking UK pipeline  
**Cons**: Code duplication (mitigate with shared utility functions)

### Option 2: Unified Scripts with Region Parameter

Add `--region` flag to existing scripts:

```bash
python scripts/fetch_racecards.py --region US --date 2025-12-31
python scripts/combine_races.py --region US --years 2020-2025
```

**Pros**: Less duplication, easier maintenance  
**Cons**: More complex logic, higher risk of regression bugs

**Recommendation**: Start with Option 1 (separate scripts) for safety, refactor to Option 2 later.

## Data Storage Structure

Extend current structure with US data:

```
data/
  raw/
    gb_races_2020.csv
    gb_races_2021.csv
    ...
    us_races_2020.csv           # NEW
    us_races_2021.csv           # NEW
    racecards_2025-12-31.json   # UK (existing)
    us_racecards_2025-12-31.json  # NEW
  
  processed/
    all_gb_races.parquet
    all_gb_races_cleaned.parquet
    race_scores.parquet
    
    all_us_races.parquet         # NEW
    all_us_races_cleaned.parquet # NEW
    us_race_scores.parquet       # NEW
    
    predictions_2025-12-31.csv   # UK predictions
    us_predictions_2025-12-31.csv # NEW
```

## US Class System Mapping

UK uses simple numeric classes. US has complex hierarchy:

### US Race Types (in quality order)

1. **Graded Stakes**
   - Grade I (highest quality, $200k-$3M purses)
   - Grade II ($150k-$1M)
   - Grade III ($100k-$500k)
   
2. **Listed Stakes** ($75k-$150k, not graded)

3. **Allowance Races** (conditions races, $40k-$80k)
   - Allowance Optional Claiming (AOC)
   - Non-winners of X allowance races
   
4. **Claiming Races** (horses for sale, $2.5k-$100k claiming price)
   - Higher claiming price = better quality
   
5. **Maiden Races** (non-winners)
   - Maiden Special Weight (MSW)
   - Maiden Claiming

### Proposed Numeric Class Mapping

To integrate with existing scoring system:

```python
US_CLASS_MAP = {
    # Stakes races (map to UK Class 1-2)
    'Grade I': 1,
    'Grade II': 1,
    'Grade III': 2,
    'Listed Stakes': 2,
    'Ungraded Stakes': 3,
    
    # Allowance (map to UK Class 3-4)
    'Allowance Optional Claiming': 3,
    'Allowance': 4,
    
    # Claiming by price (map to UK Class 4-6)
    'Claiming $75000+': 4,
    'Claiming $50000-$74999': 5,
    'Claiming $25000-$49999': 5,
    'Claiming $10000-$24999': 6,
    'Claiming $2500-$9999': 6,
    
    # Maiden (map to UK Class 5-6)
    'Maiden Special Weight': 5,
    'Maiden Claiming': 6,
}
```

## US Going Conditions Mapping

UK has 8 going conditions. US typically has 6:

```python
US_GOING_MAP = {
    'Fast': 1,           # Equivalent to UK "Firm"
    'Good': 2,           # Equivalent to UK "Good to Firm"
    'Muddy': 3,          # Wet but raceable
    'Sloppy': 4,         # Very wet (like UK "Soft")
    'Heavy': 5,          # Saturated (like UK "Heavy")
    'Frozen': 6,         # Winter racing
    
    # Turf-specific
    'Firm (Turf)': 1,
    'Good (Turf)': 2,
    'Yielding': 3,
    'Soft (Turf)': 4,
}
```

## Surface Handling

Expand beyond UK's binary Turf/AW:

```python
SURFACE_MAP = {
    # Current UK surfaces
    'Turf': 'turf',
    'AW': 'synthetic',
    
    # US surfaces (NEW)
    'Dirt': 'dirt',
    'Synthetic': 'synthetic',  # Polytrack, Tapeta
    'Turf': 'turf',
}

# Feature engineering will need surface type encoding
# Example: 3 binary flags instead of 1
df['is_dirt'] = (df['surface'] == 'dirt').astype(int)
df['is_turf'] = (df['surface'] == 'turf').astype(int)
df['is_synthetic'] = (df['surface'] == 'synthetic').astype(int)
```

## Distance Parsing Challenges

US distances can be in multiple formats:

```python
# UK (simple)
"6f"     → 6.0 furlongs
"12.5f"  → 12.5 furlongs

# US (complex)
"6f"          → 6.0 furlongs
"1m"          → 8.0 furlongs (1 mile = 8f)
"1m 1f"       → 9.0 furlongs
"9f 110y"     → 9.0 + (110/220) = 9.5 furlongs (220 yards = 1 furlong)
"1 1/8 miles" → 9.0 furlongs
```

### Parsing Function Needed

```python
def parse_us_distance(distance_str):
    """
    Convert US distance formats to furlongs.
    
    Examples:
        "6f" → 6.0
        "1m" → 8.0
        "1m 1f" → 9.0
        "9f 110y" → 9.5
        "1 1/8 miles" → 9.0
    """
    # Implementation needed (see US_RACING_CODE_EXAMPLES.md)
    pass
```

## The Odds API - US Coverage

The Odds API supports US racing. Check coverage:

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

# Check available US racing markets
response = requests.get(
    'https://api.the-odds-api.com/v4/sports',
    params={
        'apiKey': API_KEY
    }
)

# Look for: "horse_racing_us" or similar
sports = response.json()
us_racing = [s for s in sports if 'horse' in s['key'].lower()]
print(us_racing)
```

**Action Required**: Verify The Odds API has US track coverage and update `examples/odds_api_example.py`.

## Next Steps

1. Test API access: `python examples/api_example.py` with `region='US'`
2. Fetch sample US data (1 month) to validate field structure
3. Build US distance parser and test on sample data
4. Create US class mapping lookup table
5. Proceed to [US_RACING_FEATURES.md](./US_RACING_FEATURES.md) for feature engineering

## Code Example: Fetch US Races

See [US_RACING_CODE_EXAMPLES.md](./US_RACING_CODE_EXAMPLES.md) for complete script.
