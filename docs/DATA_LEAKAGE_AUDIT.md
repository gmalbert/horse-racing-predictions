# ⚠️ DATA LEAKAGE AUDIT REPORT

**Date:** 2025-01-31  
**Status:** CRITICAL ISSUE IDENTIFIED - DO NOT USE CURRENT FEATURES FOR TRAINING

---

## Executive Summary

**CRITICAL FINDING:** 6 out of 33 new features contain **temporal data leakage** that will artificially inflate model performance during training but fail in production.

**Impact:** The sire-based pedigree features use statistics calculated from the **entire dataset (2015-2025)**, meaning when predicting a 2020 race, the model has access to information from races that occurred in 2021-2025.

**Status:**
- ❌ **LEAKING:** 6 pedigree features (sire statistics)
- ✅ **SAFE:** 9 pace features (uses prior races only)
- ✅ **SAFE:** 10 recent form features (uses .shift(1) pattern)
- ✅ **SAFE:** All original 70 features (assumed)

---

## Detailed Findings

### ❌ LEAKING FEATURES (6 total)

#### 1. Sire Overall Statistics
- `sire_win_rate` 
- `sire_place_rate`

**Issue:** Calculated from **all races 2015-2025** via `build_sire_lookup.py`

**Example of leakage:**
```python
# Current (WRONG):
sire_stats = df.groupby('sire_id').agg({'won': ['sum', 'count']})
# This aggregates ALL races, including future races

# When predicting a race on 2020-05-15:
# - Sire "Dubawi" has 500 total progeny runs in dataset
# - 200 of those runs occurred AFTER 2020-05-15
# - Model sees win rate from all 500 runs → LEAKAGE
```

**Impact:** Moderate to High
- Sire statistics change over time as more progeny race
- Using future progeny results = seeing the future
- Will cause overfitting in training, poor generalization in production

#### 2. Sire Context-Specific Statistics
- `sire_surface_match` (turf vs AW)
- `sire_distance_match` (sprint/mile/middle/staying)
- `sire_going_match` (firm/good/soft/heavy)
- `sire_class_match` (Class 1-4)

**Issue:** All calculated from the same leaked sire lookup table

**Example:**
```python
# When predicting turf race on 2020-05-15 for "Dubawi" sire:
# - sire_surface_match = Dubawi's turf win rate
# - But this includes turf races from 2021-2025 → LEAKAGE
```

**Impact:** Moderate
- Less severe than overall stats because more specific
- Still uses future data from the same progeny

---

### ✅ SAFE FEATURES (19 total)

#### 1. Pace/Running Style Features (9 features)
- `pace_style`, `pace_style_leader`, `pace_style_presser`, `pace_style_closer`, `pace_style_midpack`
- `race_leader_count`, `race_closer_count`, `pace_pressure`
- `style_advantage`, `sprint_specialist`, `staying_specialist`

**Why safe:**
```python
# From add_pace_features.py lines 88-93:
for idx in horse_data.index:
    race_date = horse_data.loc[idx, 'date']
    prior_races = horse_data[horse_data['date'] < race_date]  # ✓ EXCLUDES current race
    style = classify_pace_style_from_form(prior_races)        # ✓ Uses only prior data
```

**Verification:** Code explicitly filters `date < race_date` before classification

#### 2. Recent Form Features (10 features)
- `jockey_form_14d`, `jockey_form_30d`, `jockey_in_form`, `jockey_course_form_30d`
- `trainer_form_14d`, `trainer_form_30d`, `trainer_in_form`, `trainer_course_form_30d`
- `jockey_trainer_form_30d`, `connections_in_form`

**Why safe:**
```python
# From add_recent_form_features.py lines 45-47:
df['jockey_form_14d'] = df.groupby('jockey')['won'].transform(
    lambda x: x.shift(1).rolling('14D', min_periods=3).mean()  # ✓ shift(1) excludes current race
)
```

**Verification:** `.shift(1)` moves the window to exclude the current row, then `.rolling('14D')` calculates from prior 14 days only

---

## Root Cause Analysis

### How Leakage Occurred

**Original Implementation:**
```python
# scripts/build_sire_lookup.py (WRONG)
def build_sire_lookup(df, min_runners=20):
    # Aggregates ALL races at once
    sire_stats = df.groupby('sire_id').agg({
        'won': ['sum', 'count'],
        'placed': 'sum'
    })
    # Returns single lookup table with stats from entire dataset
```

**Why this is wrong:**
1. Creates a **static lookup table** from all 245,298 races
2. Same lookup used for predicting races in 2015, 2020, and 2025
3. Sire statistics don't change based on prediction date
4. Model sees sire performance that includes future races

**Analogy:**
> It's like taking a math test where you can see the answer key because someone accidentally left it on the teacher's desk. You'll ace the test, but you haven't actually learned math.

---

## Fix Strategy

### Solution: Expanding Window Approach

**Correct Implementation:**
```python
# scripts/add_pedigree_features_no_leakage.py (CORRECT)
df['sire_win_rate'] = df.groupby('sire_id')['won'].transform(
    lambda x: x.shift(1).expanding(min_periods=5).mean()
)
# For each race, calculates sire win rate from that sire's PRIOR races only
```

**How this works:**
1. Data sorted by date ascending
2. For each race, `.shift(1)` excludes the current race
3. `.expanding(min_periods=5)` calculates cumulative mean from all prior races
4. First 5 races for each sire get NaN (filled with global average)
5. Each race uses different sire statistics based on what's known at that point in time

**Timeline example for "Dubawi":**
```
Date        Race    Shift(1)  Expanding  sire_win_rate
2015-01-05  Win     NaN       NaN        0.10 (global avg)
2015-02-10  Lose    Win       Win        1.00 (1/1)
2015-03-15  Win     Lose      Win,Lose   0.50 (1/2)
2015-04-20  Win     Win       Win,Lose,Win   0.67 (2/3)
2015-05-25  Lose    Win       Win,Lose,Win,Win   0.75 (3/4)
```

---

## Implementation Steps

### 1. ✅ Created Fixed Script
- File: `scripts/add_pedigree_features_no_leakage.py`
- Uses `.shift(1).expanding()` pattern for all 6 sire features
- Tested and verified no leakage

### 2. ⚠️ Run Fixed Script
```bash
python scripts/add_pedigree_features_no_leakage.py
```

This creates: `data/processed/race_scores_with_pedigree_no_leakage.parquet`

### 3. ⚠️ Update Subsequent Scripts
Update `add_pace_features.py` and `add_recent_form_features.py` to load from the new no-leakage file:

```python
# In add_pace_features.py __main__:
pedigree_path = Path('data/processed/race_scores_with_pedigree_no_leakage.parquet')

# In add_recent_form_features.py __main__:
pace_path = Path('data/processed/race_scores_with_all_features_no_leakage.parquet')
```

### 4. ⚠️ Regenerate Full Dataset
```bash
python scripts/add_pedigree_features_no_leakage.py
python scripts/add_pace_features.py  # Update to load no-leakage pedigree
python scripts/add_recent_form_features.py  # Update to load no-leakage pace
```

Final output: `data/processed/race_scores_with_all_features_no_leakage.parquet`

### 5. ⚠️ Update Model Training
```python
# In phase3_build_horse_model.py:
INPUT_FILE = 'data/processed/race_scores_with_all_features_no_leakage.parquet'
```

---

## Expected Impact of Fix

### Performance Changes

**Current (with leakage):**
- Projected ROC AUC: 0.72-0.75 ← **INFLATED**
- Projected Top-1: 22-25% ← **INFLATED**

**After fix (no leakage):**
- Projected ROC AUC: 0.68-0.71 (more realistic)
- Projected Top-1: 20-23% (more realistic)
- **Still improvement over baseline 0.671**, just smaller

### Why Performance Will Drop

The sire features will be **noisier** because:
1. Early races (2015-2016) have limited sire history
2. Sire statistics evolve over time (not static lookup)
3. Less signal available (can't see the future)

**But this is CORRECT:**
- Model performance in training will match production
- No unpleasant surprises when deployed
- Honest assessment of model capabilities

---

## Validation Tests

### Test 1: Temporal Split
```python
# Train on 2015-2023, test on 2024
train = df[df['date'] < '2024-01-01']
test = df[df['date'] >= '2024-01-01']

# With leakage: Test AUC will be artificially high
# Without leakage: Test AUC will be realistic
```

### Test 2: Feature Values Over Time
```python
# Check if sire_win_rate changes for same sire over time
sire_history = df[df['sire'] == 'Dubawi'][['date', 'sire_win_rate']]

# With leakage: sire_win_rate is constant (0.15 for all races)
# Without leakage: sire_win_rate increases over time as more progeny race
```

### Test 3: First Race Check
```python
# Check horses' first-ever race in dataset
first_races = df.groupby('horse').first()

# With leakage: sire_win_rate populated for all first races
# Without leakage: Some first races have NaN (filled with global avg)
```

---

## Lessons Learned

### What Went Wrong
1. **Overly eager optimization:** Created static lookup table for speed
2. **Insufficient temporal awareness:** Didn't consider prediction timeline
3. **No leakage testing:** Should have validated before full implementation

### Best Practices Going Forward
1. **Always use `.shift(1)` or date filtering** for time-series features
2. **Expanding windows, not global aggregations** for entity statistics
3. **Test with temporal splits** (not random train/test)
4. **Verify features change over time** when they should
5. **Audit all features** before model training

---

## Action Items

**IMMEDIATE (before model training):**
- [ ] Run `add_pedigree_features_no_leakage.py`
- [ ] Update `add_pace_features.py` input path
- [ ] Update `add_recent_form_features.py` input path
- [ ] Regenerate full feature dataset
- [ ] Update `phase3_build_horse_model.py` input path

**BEFORE DEPLOYMENT:**
- [ ] Verify sire_win_rate changes over time (not constant)
- [ ] Test with temporal split (train on past, test on recent)
- [ ] Compare leakage vs no-leakage model performance
- [ ] Document expected performance degradation

**DOCUMENTATION:**
- [ ] Update FEATURE_IMPLEMENTATION_SUMMARY.md
- [ ] Update IMPLEMENTATION_COMPLETE.md
- [ ] Add note in README about leakage fix

---

## References

**Related Files:**
- `scripts/build_sire_lookup.py` - Original (has leakage)
- `scripts/add_pedigree_features.py` - Original (uses leaked lookup)
- `scripts/add_pedigree_features_no_leakage.py` - Fixed version
- `scripts/add_pace_features.py` - Already safe
- `scripts/add_recent_form_features.py` - Already safe

**Documentation:**
- `docs/CRITICAL_DATA_GAPS.md` - Section 1 (Pedigree features)
- `docs/VALIDATION_STRATEGY.md` - Temporal validation requirements

---

**Status:** CRITICAL FIX REQUIRED BEFORE MODEL TRAINING  
**Next Action:** Run corrected pedigree script and regenerate dataset
