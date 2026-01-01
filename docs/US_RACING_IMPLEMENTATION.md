# US Racing - Implementation Roadmap

## Prerequisites

- [ ] Verify The Racing API has US data (test `region='US'`)
- [ ] Confirm historical depth (need 2+ years for model training)
- [ ] Check The Odds API for US track coverage
- [ ] Review US class/going terminology in API responses

## Phase 1: Data Collection & Validation (Week 1)

### Step 1.1: Fetch Sample US Data

**Goal**: Validate API structure and data quality

```bash
# Create US-specific fetch script
cp scripts/fetch_racecards.py scripts/fetch_us_racecards.py
```

**Modifications**:
- Change `region='GB'` to `region='US'`
- Update output path: `data/raw/us_racecards_YYYY-MM-DD.json`
- Save to `scripts/fetch_us_racecards.py`

**Test**:
```bash
python scripts/fetch_us_racecards.py --date 2025-12-31
```

**Validation Checklist**:
- [ ] Data fetched successfully
- [ ] `class` field contains US formats (Grade I, Claiming $X, etc.)
- [ ] `surface` field has Dirt/Turf/Synthetic
- [ ] `going` field has US conditions (Fast, Sloppy, etc.)
- [ ] `distance` field parseable

### Step 1.2: Historical Data Pull

**Goal**: Get 2 years of US race history for model training

```bash
# Clone UK historical fetch script
cp scripts/combine_gb_flat_races.py scripts/combine_us_races.py
```

**Changes**:
- Loop through 2023-2025 (or whatever API supports)
- Set `region='US'`
- Output: `data/raw/us_races_2023.csv`, etc.

**Estimated API Calls**: ~1,000 (500 days × 2 calls/day)  
**Rate Limit**: 500/month → spread over 2 months

### Step 1.3: Data Validation

Run validation on sample US data:

```bash
cp scripts/phase1_data_validation.py scripts/phase1_us_data_validation.py
# Adapt expected values (US class names, surfaces, etc.)
python scripts/phase1_us_data_validation.py
```

**Expected Outputs**:
- Row count: 50k-100k races (US has ~3,000 races/month)
- Unique courses: 50-80 (major US tracks)
- Class distribution: Stakes 10%, Allowance 15%, Claiming 50%, Maiden 25%
- Surface: Dirt 70%, Turf 25%, Synthetic 5%

## Phase 2: Data Cleaning & Standardization (Week 1-2)

### Step 2.1: Create US Lookup Tables

**File**: `scripts/phase1_create_us_lookups.py`

**Creates**:
1. **US Course Tiers**: `data/processed/lookups/us_course_tiers.csv`
   ```csv
   course,tier,state
   Churchill Downs,Premium,KY
   Belmont Park,Premium,NY
   Santa Anita,Premium,CA
   Keeneland,Premium,KY
   Saratoga,Premium,NY
   Del Mar,Major,CA
   ...
   ```

2. **US Class Mapping**: `data/processed/lookups/us_class_numeric.csv`
   ```csv
   class_string,class_numeric,class_category
   Grade I,1,Stakes
   Grade II,1,Stakes
   Grade III,2,Stakes
   Listed Stakes,2,Stakes
   Allowance Optional Claiming,3,Allowance
   Claiming $75000+,4,Claiming
   ...
   ```

3. **US Going Mapping**: `data/processed/lookups/us_going_numeric.csv`
   ```csv
   going_string,going_numeric
   Fast,1
   Good,2
   Muddy,3
   Sloppy,4
   ...
   ```

### Step 2.2: Distance Parser

**File**: `scripts/us_distance_parser.py`

**Function**:
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
    import re
    
    # Remove extra whitespace
    distance_str = distance_str.strip().lower()
    
    # Pattern 1: "6f" (simple furlongs)
    if re.match(r'^\d+(\.\d+)?f$', distance_str):
        return float(distance_str[:-1])
    
    # Pattern 2: "1m" or "2m" (miles)
    if re.match(r'^\d+m$', distance_str):
        miles = int(distance_str[:-1])
        return miles * 8.0
    
    # Pattern 3: "1m 1f" (miles and furlongs)
    match = re.match(r'^(\d+)m\s+(\d+)f$', distance_str)
    if match:
        miles = int(match.group(1))
        furlongs = int(match.group(2))
        return miles * 8.0 + furlongs
    
    # Pattern 4: "9f 110y" (furlongs and yards)
    match = re.match(r'^(\d+)f\s+(\d+)y$', distance_str)
    if match:
        furlongs = int(match.group(1))
        yards = int(match.group(2))
        return furlongs + (yards / 220.0)  # 220 yards = 1 furlong
    
    # Pattern 5: "1 1/8 miles" (fractional miles)
    match = re.match(r'^(\d+)\s+(\d+)/(\d+)\s+miles?$', distance_str)
    if match:
        whole = int(match.group(1))
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        total_miles = whole + (numerator / denominator)
        return total_miles * 8.0
    
    # Pattern 6: "1 1/8m" (fractional miles, compact)
    match = re.match(r'^(\d+)\s+(\d+)/(\d+)m$', distance_str)
    if match:
        whole = int(match.group(1))
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        total_miles = whole + (numerator / denominator)
        return total_miles * 8.0
    
    # If no pattern matches, return None
    return None
```

**Test Cases**:
```python
assert parse_us_distance("6f") == 6.0
assert parse_us_distance("1m") == 8.0
assert parse_us_distance("1m 1f") == 9.0
assert parse_us_distance("9f 110y") == 9.5
assert parse_us_distance("1 1/8 miles") == 9.0
assert parse_us_distance("1 1/4m") == 10.0
```

### Step 2.3: US Data Cleaning Pipeline

**File**: `scripts/phase1_us_data_cleaning.py`

**Tasks**:
1. Parse distances using `parse_us_distance()`
2. Map class strings to numeric using lookup table
3. Map going conditions using lookup table
4. Standardize course names (remove state suffixes if present)
5. Create surface type flags (is_dirt, is_turf, is_synthetic)
6. Handle missing official ratings (US doesn't use BHA ratings)
7. Clean position values (same logic as UK)

**Output**: `data/processed/all_us_races_cleaned.parquet`

**Run**:
```bash
python scripts/phase1_us_data_cleaning.py
```

## Phase 3: Race Scoring (Week 2)

### Step 3.1: US Race Scoring Logic

**File**: `scripts/phase2_score_us_races.py`

**Scoring Criteria** (adapted from UK version):

```python
def score_us_race(race_row, course_tier, historical_stats):
    """
    Score US races 0-100 based on predictability.
    
    Factors:
    - Premium track (+15): Churchill, Belmont, Santa Anita, Keeneland, Saratoga
    - Major track (+5): Del Mar, Gulfstream, Oaklawn
    - Stakes race (+20): Grade I/II/III
    - Listed stakes (+15)
    - High-value allowance (+10): >$75k purse
    - Field size (8-12 ideal): +10
    - Classic distance (8f-10f): +10
    - Dirt sprint at premium track (+5): Speed bias, predictable
    """
    score = 0
    
    # Course quality
    if course_tier == 'Premium':
        score += 15
    elif course_tier == 'Major':
        score += 5
    
    # Race class
    if 'Grade' in race_row['class']:
        score += 20
    elif 'Listed' in race_row['class']:
        score += 15
    elif 'Allowance' in race_row['class'] and race_row['prize'] > 75000:
        score += 10
    
    # Distance
    distance_f = race_row['distance_f']
    if 8.0 <= distance_f <= 10.0:  # Classic distances
        score += 10
    elif 6.0 <= distance_f <= 7.0:  # Sprints
        score += 5
    
    # Field size
    field_size = race_row['field_size']
    if 8 <= field_size <= 12:
        score += 10
    elif field_size > 16:
        score -= 5  # Too chaotic
    
    # Surface: Dirt is more predictable than turf in US
    if race_row['surface'] == 'Dirt':
        score += 5
    
    return min(score, 100)  # Cap at 100
```

**Output**: `data/processed/us_race_scores.parquet`

## Phase 4: Feature Engineering (Week 2-3)

### Step 4.1: Extend Feature Engineering Scripts

**Option A**: Create `scripts/phase3_build_us_horse_model.py` (recommended)
**Option B**: Add `--region US` flag to existing script

**Key Changes from UK**:
1. Add surface-specific form tracking (`cd_surface_key`)
2. Add surface switch features (`turf_to_dirt`, `dirt_to_turf`)
3. Use US class numeric mapping
4. Adjust recency thresholds (30 days vs. 60 days)
5. Use US going numeric mapping
6. Create 3-way surface encoding (dirt/turf/synthetic)

**Run**:
```bash
python scripts/phase3_build_us_horse_model.py
```

**Output**:
- `data/processed/us_race_features.parquet` (all features)
- `models/us_feature_columns.txt` (feature list)

## Phase 5: Model Training (Week 3)

### Step 5.1: Train US Model

**File**: `scripts/train_us_model.py` (clone from `phase3_build_horse_model.py`)

**Key Settings**:
```python
# XGBoost parameters (may need tuning for US data)
us_model_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42
}

# Train/test split
# Use 2023-2024 for training, 2025 for testing
train_df = us_df[us_df['date'] < '2025-01-01']
test_df = us_df[us_df['date'] >= '2025-01-01']
```

**Validation**:
- Check ROC AUC > 0.65 (comparable to UK model)
- Review feature importance (surface features should be top 5)
- Test on held-out 2025 data

**Output**: `models/us_horse_model.pkl`

### Step 5.2: Compare UK vs. US Models

**Analysis Script**: `scripts/compare_uk_us_models.py`

```python
import joblib
import pandas as pd

# Load both models
uk_model = joblib.load('models/horse_model.pkl')
us_model = joblib.load('models/us_horse_model.pkl')

# Load feature importance
uk_importance = pd.read_csv('models/feature_importance.csv')
us_importance = pd.read_csv('models/us_feature_importance.csv')

# Compare
comparison = pd.merge(
    uk_importance, us_importance,
    on='feature', suffixes=('_uk', '_us')
)

print(comparison.sort_values('importance_uk', ascending=False))
```

**Expected Findings**:
- US model relies more on surface features
- US model weights jockey higher
- UK model relies more on CD form (course specialists)

## Phase 6: Prediction Pipeline (Week 3-4)

### Step 6.1: US Prediction Script

**File**: `scripts/predict_us_todays_races.py`

**Logic**:
1. Fetch today's US racecards (via `fetch_us_racecards.py`)
2. Clean and standardize
3. Engineer features
4. Load US model
5. Predict win probabilities
6. Convert to odds (decimal and fractional)
7. Save: `data/processed/us_predictions_YYYY-MM-DD.csv`

**Run**:
```bash
python scripts/predict_us_todays_races.py
# Or with date:
python scripts/predict_us_todays_races.py --date 2025-12-31
```

### Step 6.2: Unified Prediction Script (Optional)

**File**: `scripts/predict_todays_races.py` (extend existing)

Add `--region` flag:
```bash
python scripts/predict_todays_races.py --region US --date 2025-12-31
python scripts/predict_todays_races.py --region GB --date 2025-12-31
```

**Implementation**:
```python
def main(region='GB', date=None):
    if region == 'US':
        model_path = 'models/us_horse_model.pkl'
        fetch_script = fetch_us_racecards
        class_map = US_CLASS_MAP
    else:
        model_path = 'models/horse_model.pkl'
        fetch_script = fetch_gb_racecards
        class_map = UK_CLASS_MAP
    
    # ... rest of prediction logic
```

## Phase 7: UI Integration (Week 4)

### Step 7.1: Add Region Selector to Streamlit

**File**: `predictions.py`

**Changes**:

1. **Sidebar - Add Region Toggle**:
```python
st.sidebar.title("🏇 Race Predictions")

# NEW: Region selector
region = st.sidebar.radio(
    "Racing Region",
    options=["UK/Ireland", "United States"],
    index=0
)

# Set internal region code
region_code = 'US' if region == "United States" else 'GB'
```

2. **Load Region-Specific Data**:
```python
@st.cache_data
def load_data(region='GB'):
    if region == 'GB':
        df = pd.read_parquet('data/processed/race_scores.parquet')
        model_path = 'models/horse_model.pkl'
    else:
        df = pd.read_parquet('data/processed/us_race_scores.parquet')
        model_path = 'models/us_horse_model.pkl'
    
    return df, model_path

df, model_path = load_data(region_code)
```

3. **Update Filters for US**:
```python
# Class filter (region-aware)
if region_code == 'GB':
    class_options = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
else:
    class_options = ['Grade I', 'Grade II', 'Grade III', 'Listed', 'Allowance', 'Claiming']

selected_classes = st.sidebar.multiselect("Race Class", class_options)

# Surface filter (US-specific)
if region_code == 'US':
    surface_options = ['All', 'Dirt', 'Turf', 'Synthetic']
    selected_surface = st.sidebar.selectbox("Surface", surface_options)
    
    if selected_surface != 'All':
        df = df[df['surface'] == selected_surface]
```

4. **Region-Specific Course Lists**:
```python
# Premium courses (region-aware)
if region_code == 'GB':
    premium_courses = ['Ascot', 'Newmarket', 'York', 'Doncaster', 'Goodwood']
else:
    premium_courses = ['Churchill Downs', 'Belmont Park', 'Santa Anita', 'Keeneland', 'Saratoga']
```

5. **Update Tabs**:
```python
tab1, tab2, tab3 = st.tabs([
    "📊 Data Explorer",
    "🔮 ML Predictions",
    "💰 Value Bets"
])

# In each tab, show region-specific data
with tab1:
    st.header(f"{region} Racing Data")
    # ... existing data explorer logic
```

### Step 7.2: Today's Predictions Tab

**Add region-specific prediction loading**:
```python
@st.cache_data
def load_todays_predictions(region='GB'):
    today = datetime.now().strftime('%Y-%m-%d')
    
    if region == 'GB':
        pred_file = f'data/processed/predictions_{today}.csv'
    else:
        pred_file = f'data/processed/us_predictions_{today}.csv'
    
    if Path(pred_file).exists():
        return pd.read_csv(pred_file)
    else:
        return None

predictions = load_todays_predictions(region_code)

if predictions is not None:
    st.success(f"✅ {region} predictions loaded for {today}")
    st.dataframe(predictions)
else:
    st.info(f"No {region} predictions available for today. Generate using prediction script.")
```

## Phase 8: Testing & Validation (Week 4)

### Test Checklist

**Data Pipeline**:
- [ ] US data fetches correctly from API
- [ ] Distance parser handles all US formats
- [ ] Class mapping produces sensible numeric values
- [ ] Surface flags correctly assigned (dirt/turf/synthetic)

**Feature Engineering**:
- [ ] Career stats calculate correctly
- [ ] Surface-specific form tracks properly
- [ ] Class step logic works for claiming → stakes progression
- [ ] No data leakage (future data in past features)

**Model Training**:
- [ ] US model ROC AUC > 0.65
- [ ] Feature importance makes sense (surface features prominent)
- [ ] No overfitting (test set performance close to training)

**Predictions**:
- [ ] US predictions generate for sample date
- [ ] Odds conversion works (decimal and fractional)
- [ ] Value bets identified correctly

**UI**:
- [ ] Region selector works
- [ ] Filters update based on region
- [ ] Data loads for both UK and US
- [ ] No errors when switching regions

## Deployment Checklist

- [ ] Update `README.md` with US racing support
- [ ] Update `.env.example` with any new variables
- [ ] Document US-specific workflows in `.github/copilot-instructions.md`
- [ ] Add US course tier CSV to repo
- [ ] Add US class/going mapping CSVs
- [ ] Update `requirements.txt` if new packages needed
- [ ] Run full pipeline on test data
- [ ] Validate predictions on historical data (backtesting)

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Data Collection | 3-5 days | API access validated |
| 2. Data Cleaning | 3-5 days | Phase 1 complete |
| 3. Race Scoring | 1-2 days | Phase 2 complete |
| 4. Feature Engineering | 3-5 days | Phase 2 complete |
| 5. Model Training | 2-3 days | Phase 4 complete |
| 6. Prediction Pipeline | 2-3 days | Phase 5 complete |
| 7. UI Integration | 3-4 days | Phase 6 complete |
| 8. Testing | 2-3 days | All phases |

**Total**: 3-4 weeks (with testing)

## Risk Mitigation

**Risk**: US data quality lower than UK  
**Mitigation**: Validate on sample month before full historical pull

**Risk**: Feature engineering breaks UK pipeline  
**Mitigation**: Use separate scripts initially, merge later

**Risk**: US model performs poorly  
**Mitigation**: Benchmark against simple baselines (favorite win rate, odds-derived probabilities)

**Risk**: Odds API doesn't cover US tracks well  
**Mitigation**: Research alternative sources (DraftKings API, FanDuel, TVG)

## Next Steps

1. Complete Phase 1 (data collection) and validate API access
2. See [US_RACING_CODE_EXAMPLES.md](./US_RACING_CODE_EXAMPLES.md) for ready-to-use code
3. Track progress in project task list
4. Document learnings and edge cases as you encounter them
