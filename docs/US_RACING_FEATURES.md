# US Racing - Feature Engineering Adaptations

## Core Principle

**Don't break UK features** - Add parallel US-specific logic using region flag.

## Region-Aware Feature Engineering

### Approach 1: Separate Feature Functions (Recommended Initially)

```python
def engineer_career_features_uk(df):
    """UK-specific career stats"""
    # Existing logic from phase3_build_horse_model.py
    pass

def engineer_career_features_us(df):
    """
    US-specific career stats.
    
    Key differences:
    - US horses race more frequently (monthly vs. UK every 3-6 weeks)
    - Adjust recency windows: last 3 months instead of last 3 runs
    - Class progression: claiming → allowance → stakes
    """
    pass

def engineer_career_features(df, region='GB'):
    """Route to region-specific implementation"""
    if region == 'US':
        return engineer_career_features_us(df)
    else:
        return engineer_career_features_uk(df)
```

### Approach 2: Unified Functions with Region Column (Long-Term)

```python
def engineer_career_features(df):
    """
    Region-aware career feature engineering.
    Requires df to have 'region' column.
    """
    # Sort by horse and date
    df = df.sort_values(['horse', 'date']).copy()
    
    # Universal features (work for both)
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    df['career_runs'] = df.groupby('horse').cumcount()
    
    # Region-specific recency calculation
    df['days_since_last'] = df.groupby('horse')['date'].diff().dt.days
    
    # US: More frequent racing, use 90-day window for "recent"
    # UK: Less frequent, use last 3 runs
    df['is_recent_run_us'] = (df['days_since_last'] <= 90)
    df['is_recent_run_uk'] = (df.groupby('horse').cumcount() <= 3)
    
    df['is_recent_run'] = np.where(
        df['region'] == 'US',
        df['is_recent_run_us'],
        df['is_recent_run_uk']
    )
    
    return df
```

**Recommendation**: Start with Approach 1 (cleaner separation), migrate to Approach 2 when US features are validated.

## Feature Adaptations by Category

### 1. Career Statistics

**UK Implementation** (current):
```python
df['career_runs'] = df.groupby('horse').cumcount()
df['career_win_rate'] = df['career_wins'] / df['career_runs']
df['career_earnings'] = df.groupby('horse')['prize_numeric'].cumsum()
```

**US Adaptation** (new):
```python
# US horses have higher volume, may need minimum runs threshold
df['career_win_rate_us'] = np.where(
    df['career_runs'] >= 5,  # US: 5 run minimum (vs. UK 3)
    df['career_wins'] / df['career_runs'],
    0  # Insufficient data
)

# US earnings more variable (claiming vs. stakes), use log scale
df['career_earnings_log_us'] = np.log1p(df['career_earnings'])
```

### 2. Course-Distance Form

**UK Implementation**:
```python
df['cd_key'] = df['course'] + '_' + df['distance_band']
df['cd_win_rate'] = df.groupby(['horse', 'cd_key'])['won'].mean()
```

**US Adaptation** - Add Surface to CD:
```python
# In US, dirt vs. turf specialists are critical
df['cd_surface_key'] = (
    df['course'] + '_' + 
    df['distance_band'] + '_' + 
    df['surface']
)

# Example: "Churchill Downs_Mile_Dirt" vs. "Churchill Downs_Mile_Turf"
df['cds_win_rate'] = df.groupby(['horse', 'cd_surface_key'])['won'].mean()

# Also track surface-only form
df['surface_runs'] = df.groupby(['horse', 'surface']).cumcount()
df['surface_win_rate'] = df.groupby(['horse', 'surface'])['won'].mean()
```

### 3. Class Step Detection

**UK Implementation**:
```python
CLASS_ORDER_UK = {
    'Class 1': 1,
    'Class 2': 2,
    'Class 3': 3,
    'Class 4': 4,
    'Class 5': 5,
    'Class 6': 6,
    'Class 7': 7
}

df['class_numeric'] = df['class'].map(CLASS_ORDER_UK)
df['class_last_run'] = df.groupby('horse')['class_numeric'].shift(1)
df['class_step'] = df['class_numeric'] - df['class_last_run']
# Negative = stepping up (harder race)
# Positive = dropping down (easier race)
```

**US Adaptation**:
```python
CLASS_ORDER_US = {
    'Grade I': 1,
    'Grade II': 1.5,
    'Grade III': 2,
    'Listed Stakes': 2.5,
    'Ungraded Stakes': 3,
    'Allowance Optional Claiming': 3.5,
    'Allowance': 4,
    'Claiming $75000+': 4.5,
    'Claiming $50000-$74999': 5,
    'Claiming $25000-$49999': 5.5,
    'Claiming $10000-$24999': 6,
    'Claiming $2500-$9999': 6.5,
    'Maiden Special Weight': 7,
    'Maiden Claiming': 7.5
}

# Map US class strings to numeric
df['class_numeric_us'] = df['class'].map(CLASS_ORDER_US)

# Same logic as UK
df['class_last_run'] = df.groupby('horse')['class_numeric_us'].shift(1)
df['class_step'] = df['class_numeric_us'] - df['class_last_run']
```

**Key Insight**: US class drops (e.g., stakes → claiming) are often deliberate trainer moves to find easier spots. This is a stronger signal than in UK.

### 4. Surface Switch Penalty (US-Specific)

**New Feature for US**:
```python
# Track last surface run
df['surface_last'] = df.groupby('horse')['surface'].shift(1)

# Flag surface switches
df['surface_switch'] = (
    (df['surface'] != df['surface_last']) & 
    (df['surface_last'].notna())
).astype(int)

# Specific switch types
df['turf_to_dirt'] = (
    (df['surface'] == 'dirt') & 
    (df['surface_last'] == 'turf')
).astype(int)

df['dirt_to_turf'] = (
    (df['surface'] == 'turf') & 
    (df['surface_last'] == 'dirt')
).astype(int)

# Note: Turf→Dirt often signals form decline; Dirt→Turf can be strategic
```

### 5. Going/Track Condition

**UK Implementation**:
```python
GOING_MAP_UK = {
    'Hard': 1, 'Firm': 2, 'Good to Firm': 3, 'Good': 4,
    'Good to Soft': 5, 'Soft': 6, 'Heavy': 7
}
df['going_numeric'] = df['going'].map(GOING_MAP_UK)
```

**US Adaptation**:
```python
GOING_MAP_US = {
    'Fast': 1,
    'Good': 2,
    'Muddy': 3,
    'Sloppy': 4,
    'Heavy': 5,
    'Frozen': 6,
    # Turf
    'Firm (Turf)': 1,
    'Good (Turf)': 2,
    'Yielding': 3,
    'Soft (Turf)': 4
}
df['going_numeric_us'] = df['going'].map(GOING_MAP_US)

# US: Track condition interacts with surface more
# Create combined feature
df['surface_going'] = df['surface'] + '_' + df['going']
# Example: "Dirt_Fast", "Turf_Yielding"
```

### 6. Race Frequency (US-Specific Adjustment)

**US horses race more often**:
```python
# Days since last race
df['days_since_last'] = df.groupby('horse')['date'].diff().dt.days

# In UK: 30-60 days typical, >90 days = layoff concern
# In US: 14-30 days typical, >60 days = layoff concern

df['is_fresh_uk'] = (df['days_since_last'] <= 60)
df['is_fresh_us'] = (df['days_since_last'] <= 30)

df['is_fresh'] = np.where(
    df['region'] == 'US',
    df['is_fresh_us'],
    df['is_fresh_uk']
)

# Layoff flag
df['is_layoff_uk'] = (df['days_since_last'] > 90)
df['is_layoff_us'] = (df['days_since_last'] > 60)
```

### 7. Jockey & Trainer Stats

**UK Implementation** (current):
```python
df['jockey_win_rate'] = df.groupby('jockey')['won'].transform('mean')
df['trainer_win_rate'] = df.groupby('trainer')['won'].transform('mean')
```

**US Adaptation** - Same logic, but US has stronger jockey effect:
```python
# In US, top jockeys (Ortiz, Rosario, Prat) have 20%+ win rates
# In UK, spread is tighter (top jockeys ~18%)

# Consider adding jockey-trainer combo (US barns often pair)
df['jockey_trainer_combo'] = df['jockey'] + '_' + df['trainer']
df['combo_win_rate'] = df.groupby('jockey_trainer_combo')['won'].transform('mean')

# Only use if sufficient sample size
df['combo_sample_size'] = df.groupby('jockey_trainer_combo').cumcount()
df['combo_win_rate'] = np.where(
    df['combo_sample_size'] >= 10,
    df['combo_win_rate'],
    df['jockey_win_rate']  # Fall back to jockey-only if small sample
)
```

### 8. Speed Figures (US Opportunity)

**UK**: Uses Official Ratings (BHA ratings, 60-130 scale)  
**US**: Uses Beyer Speed Figures, Timeform, Equibase figures (not in basic API)

**If available in API or via scraping**:
```python
# UK
df['official_rating_normalized'] = df['official_rating'] / 130

# US equivalent (if Beyer figures available)
df['beyer_normalized'] = df['beyer_speed_figure'] / 120

# Track best recent figure
df['best_beyer_last3'] = (
    df.groupby('horse')['beyer_speed_figure']
       .rolling(3, min_periods=1)
       .max()
       .reset_index(0, drop=True)
)
```

**Note**: Speed figures may not be in base Racing API. Explore external data sources or scraping.

## Feature Importance - Expected Changes

Based on US racing dynamics:

**UK Model Feature Importance** (current):
1. Career win rate (18%)
2. Class step (12%)
3. CD win rate (11%)
4. Recent form (10%)
5. Days since last race (8%)
6. Official rating (8%)
7. Jockey win rate (7%)

**Expected US Model Feature Importance**:
1. **Surface form** (15-20%) - Dirt/turf specialists
2. Class step (12-15%) - Claiming drops very predictive
3. Career win rate (10-12%)
4. Jockey win rate (10-12%) - Stronger jockey effect
5. **Surface switch penalty** (8-10%) - New feature
6. Recent form (8-10%)
7. CD form (6-8%) - Less important (horses ship more)
8. Days since last race (5-7%) - Different threshold

## Distance Bands - US Version

**UK Distance Bands** (current):
```python
DISTANCE_BANDS_UK = {
    (5.0, 6.5): 'Sprint',    # 5f-6f
    (6.5, 8.5): 'Mile',      # 7f-8f
    (8.5, 12.5): 'Middle',   # 9f-12f
    (12.5, 20.0): 'Long'     # 13f+
}
```

**US Distance Bands** (adapted for common US distances):
```python
DISTANCE_BANDS_US = {
    (5.0, 6.5): 'Sprint',         # 5f-6f (common)
    (6.5, 7.5): 'One-Turn Mile',  # 7f-7.5f (one-turn tracks)
    (7.5, 9.0): 'Classic',        # 8f-9f (1 mile - 1⅛ miles, most common)
    (9.0, 10.5): 'Route',         # 9.5f-10f (1⅛ - 1¼ miles)
    (10.5, 14.0): 'Long'          # 10.5f+ (rare, marathon races)
}
```

**Key Differences**:
- US has more 8f-9f races (classic distances)
- One-turn vs. two-turn mile is significant in US
- Very long distances (14f+) are rare in US compared to UK

## Model Training Considerations

### Separate Models Required

**Why?**
- Feature distributions differ (surface types, class ranges, race frequency)
- Different optimal hyperparameters likely
- Risk of US data overwhelming UK signals (if combined)

**Approach**:
```python
# Train separate models
uk_model = XGBClassifier(**uk_params)
uk_model.fit(uk_features, uk_targets)

us_model = XGBClassifier(**us_params)
us_model.fit(us_features, us_targets)

# Save separately
joblib.dump(uk_model, 'models/uk_horse_model.pkl')
joblib.dump(us_model, 'models/us_horse_model.pkl')
```

### Shared Feature Schema

Keep feature column names consistent:
```python
SHARED_FEATURES = [
    'career_runs', 'career_win_rate', 'career_place_rate',
    'cd_win_rate', 'class_step', 'days_since_last',
    'field_size', 'going_numeric', 'distance_f',
    'jockey_win_rate', 'trainer_win_rate',
    # Region-specific binary flags
    'is_dirt', 'is_turf', 'is_synthetic',
    'surface_switch', 'turf_to_dirt', 'dirt_to_turf'
]
```

Even if UK doesn't use dirt/turf flags (all 0), keep columns for unified prediction pipeline.

## Next Steps

1. Review [US_RACING_IMPLEMENTATION.md](./US_RACING_IMPLEMENTATION.md) for integration roadmap
2. See [US_RACING_CODE_EXAMPLES.md](./US_RACING_CODE_EXAMPLES.md) for complete code samples
3. Validate feature engineering on sample US data before full pipeline run
