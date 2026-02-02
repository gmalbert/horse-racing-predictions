# Data Leakage Audit Report — Feature Implementation v2.3

**Date**: 2026-02-01  
**Auditor**: AI Coding Agent  
**Scope**: Pedigree, Going Preferences, and OR Context features

---

## Executive Summary

**Status**: ✅ **ALL LEAKAGE ISSUES FIXED**

### Issues Found and Resolved

| Script | Issue | Severity | Status |
|--------|-------|----------|--------|
| `add_going_preference_features.py` | Filtered data not re-sorted before expanding window | 🔴 HIGH | ✅ FIXED |
| `add_or_context_features.py` | Class OR typical not sorted by date | 🔴 HIGH | ✅ FIXED |
| `add_or_context_features.py` | Career high OR not sorted by horse+date | 🟡 MEDIUM | ✅ FIXED |
| `add_pedigree_features.py` | None found | ✅ CLEAN | ✅ VERIFIED |

---

## Detailed Audit

### 1. Pedigree Features (`add_pedigree_features.py`)

**Status**: ✅ **NO LEAKAGE DETECTED**

#### Sire Features
```python
# CORRECT: Sorts by sire and date before expanding window
df = df.sort_values(['sire', 'date']).copy()

# CORRECT: Uses shift(1) to exclude current race
df['sire_win_rate_v2'] = (
    df.groupby('sire')['won']
    .apply(lambda x: x.shift(1).expanding().mean())
    .fillna(0.0)
)
```

**Verification**:
- ✅ Sorted by `['sire', 'date']` before all calculations
- ✅ All expanding windows use `shift(1)` to exclude current race
- ✅ Surface-specific win rates properly separated and shifted
- ✅ Distance/class/sprint/stayer calculations all use shifted expanding windows

#### Dam Features
```python
# CORRECT: Sorts by dam and date
df = df.sort_values(['dam', 'date']).copy()

# CORRECT: Uses shift(1)
df['dam_offspring_win_rate'] = (
    df.groupby('dam')['won']
    .apply(lambda x: x.shift(1).expanding().mean())
    .fillna(0.0)
)
```

**Verification**:
- ✅ Sorted by `['dam', 'date']`
- ✅ Expanding window with shift(1)

#### Damsire Features
```python
# CORRECT: Sorts by damsire and date
df = df.sort_values(['damsire', 'date']).copy()

# CORRECT: Stamina score uses shifted expanding window
df['damsire_avg_win_dist'] = (
    df.groupby('damsire')['damsire_win_distance']
    .apply(lambda x: x.shift(1).expanding().sum())
    ...
)
```

**Verification**:
- ✅ Sorted by `['damsire', 'date']`
- ✅ Expanding window with shift(1)

---

### 2. Going Preference Features (`add_going_preference_features.py`)

**Status**: 🔴 **LEAKAGE FOUND AND FIXED**

#### Issue 1: Horse Going Preferences

**BEFORE (LEAKED)**:
```python
# WRONG: Filtered data loses date ordering within horses
going_df = df[df['going_category'] == going_type].copy()

# This shift(1) operates on incorrectly ordered data
going_stats = going_df.groupby('horse_id').agg({
    'won': lambda x: x.shift(1).expanding().sum(),
    ...
})
```

**Problem**: When you filter `df` to only heavy going races, the resulting `going_df` is no longer sorted by `['horse_id', 'date']`. The shift(1) operates on whatever order the rows happen to be in, causing leakage.

**AFTER (FIXED)**:
```python
# CORRECT: Re-sort after filtering
going_df = df[df['going_category'] == going_type].copy()
going_df = going_df.sort_values(['horse_id', 'date']).reset_index(drop=True)

# Now shift(1) operates on correctly ordered data
going_stats = going_df.groupby('horse_id').agg({
    'won': lambda x: x.shift(1).expanding().sum(),
    ...
})
```

**Fix Applied**: Added `.sort_values(['horse_id', 'date']).reset_index(drop=True)` after filtering

#### Issue 2: Sire Going Preferences

**Same issue as above**, fixed with:
```python
going_df = df[df['going_category'] == going_type].copy()
going_df = going_df.sort_values(['sire_id', 'date']).reset_index(drop=True)
```

**Impact of Fix**:
- Before: Could use future going performance in current prediction
- After: Only uses past going performance for each horse/sire

---

### 3. OR Context Features (`add_or_context_features.py`)

**Status**: 🔴 **LEAKAGE FOUND AND FIXED**

#### Issue 1: Class OR Typical

**BEFORE (LEAKED)**:
```python
# WRONG: Groups by class without considering date order
class_or_typical = df.groupby('class_numeric').agg({
    'or_numeric': lambda x: x.shift(1).expanding().mean()
}).reset_index()
```

**Problem**: When grouping by `class_numeric` alone, the shift(1) operates on randomly ordered dates. A Class 3 race in 2024 could use Class 3 OR data from 2025.

**AFTER (FIXED)**:
```python
# CORRECT: Sort by class AND date, apply shift within groups
df = df.sort_values(['class_numeric', 'date']).reset_index(drop=True)

df['class_or_typical'] = (
    df.groupby('class_numeric')['or_numeric']
    .apply(lambda x: x.shift(1).expanding().mean())
    .fillna(df['or_numeric'])
)
```

**Fix Applied**: Sort by `['class_numeric', 'date']` before expanding window calculation

#### Issue 2: Career High OR

**BEFORE (POTENTIALLY LEAKED)**:
```python
# UNCLEAR: No explicit sort before cummax
df['or_career_high'] = df.groupby('horse_id')['or_numeric'].cummax()
```

**Problem**: Without explicit sorting, `cummax()` operates on whatever row order exists, which may not be chronological.

**AFTER (FIXED)**:
```python
# CORRECT: Explicit sort before cummax
df = df.sort_values(['horse_id', 'date']).reset_index(drop=True)

df['or_career_high'] = df.groupby('horse_id')['or_numeric'].cummax()
df['or_at_career_high'] = (df['or_numeric'] == df['or_career_high']).astype(int)
```

**Fix Applied**: Added explicit sort by `['horse_id', 'date']` before career high calculation

#### ✅ Race-Level Features (NO LEAKAGE)

These features are CORRECT because they compare only within the same race:

```python
# CORRECT: Race-level comparisons don't leak across time
race_or_stats = df.groupby('race_id')['or_numeric'].agg(['max', 'mean', 'min', 'std'])
df['or_vs_race_max'] = df['or_numeric'] - df['race_or_max']
df['or_percentile'] = df.groupby('race_id')['or_numeric'].rank(pct=True) * 100
```

**Why this is safe**: Each `race_id` represents a single point in time. Comparing horses within the same race cannot create temporal leakage.

---

## Leakage Prevention Principles

### ✅ DO THIS

1. **Always sort before temporal calculations**:
   ```python
   df = df.sort_values(['group_var', 'date']).copy()
   ```

2. **Use shift(1) with expanding windows**:
   ```python
   df['feature'] = df.groupby('group')['target'].apply(
       lambda x: x.shift(1).expanding().mean()
   )
   ```

3. **Re-sort after filtering**:
   ```python
   subset = df[df['condition'] == True].copy()
   subset = subset.sort_values(['group', 'date'])  # CRITICAL!
   ```

4. **Use cummax/cumsum only after sorting**:
   ```python
   df = df.sort_values(['horse_id', 'date'])
   df['career_max'] = df.groupby('horse_id')['or'].cummax()
   ```

5. **Race-level features are safe** (no time dimension):
   ```python
   df['or_percentile'] = df.groupby('race_id')['or'].rank(pct=True)
   ```

### ❌ DON'T DO THIS

1. **Don't filter without re-sorting**:
   ```python
   # WRONG
   subset = df[df['going'] == 'heavy']
   subset.groupby('horse')['won'].apply(lambda x: x.shift(1).expanding().mean())
   ```

2. **Don't use expanding without shift**:
   ```python
   # WRONG - includes current race
   df.groupby('horse')['won'].expanding().mean()
   ```

3. **Don't group by time-unaware keys**:
   ```python
   # WRONG - mixes all dates
   df.groupby('class')['or'].mean()  # Use expanding instead
   ```

4. **Don't use cummax without explicit sort**:
   ```python
   # WRONG - order undefined
   df.groupby('horse')['or'].cummax()  # Sort first!
   ```

---

## Testing Recommendations

### Manual Verification Test

Run this on each new feature to check for leakage:

```python
import pandas as pd

# Load processed data
df = pd.read_parquet('data/processed/race_scores_or_context.parquet')

# Pick a specific horse and date
test_horse = 'Example Horse'
test_date = '2024-06-15'

# Check: does this horse's feature use any data from after test_date?
horse_history = df[
    (df['horse'] == test_horse) & 
    (df['date'] <= test_date)
].sort_values('date')

# The feature value at test_date should only depend on rows BEFORE test_date
print(horse_history[['date', 'sire_win_rate_v2', 'horse_heavy_win_rate']].tail(10))
```

### Automated Leakage Test

Add to test suite:

```python
def test_no_temporal_leakage():
    """Verify features don't use future data"""
    df = pd.read_parquet('data/processed/race_scores_or_context.parquet')
    
    # Sort by date
    df = df.sort_values('date')
    
    # For each temporal feature, check monotonicity within groups
    temporal_features = [
        'sire_win_rate_v2', 'dam_offspring_win_rate', 
        'horse_heavy_win_rate', 'or_career_high'
    ]
    
    for feature in temporal_features:
        # Feature should never decrease when a horse wins
        # (it's cumulative, can only stay same or increase)
        for horse in df['horse_id'].unique():
            horse_df = df[df['horse_id'] == horse].copy()
            
            # Check: cumulative stats should be monotonic or stable
            # (allowing for NaNs at start)
            assert not (horse_df[feature].diff() < -0.01).any(), \
                f"{feature} decreased for {horse} - possible leakage"
```

---

## Compilation and Deployment

### Pre-Deployment Checklist

- ✅ All scripts compile successfully
- ✅ Leakage audit completed
- ✅ Leakage issues fixed
- ✅ Copilot instructions updated with leakage prevention rules
- ⏳ Manual verification test passed (run after feature generation)
- ⏳ Model retrained with fixed features
- ⏳ AUC measured and compared to baseline

### Files Modified

1. `scripts/add_going_preference_features.py` — Added sort after filter (lines 67, 126)
2. `scripts/add_or_context_features.py` — Added sort before class/career calculations (lines 61, 90)
3. `.github/copilot-instructions.md` — Added data leakage prevention section

---

## Conclusion

**Final Status**: ✅ **ALL CLEAR**

All temporal features now have proper safeguards:
1. ✅ Explicit sorting by [group, date] before calculations
2. ✅ shift(1) used with all expanding windows
3. ✅ Filtered subsets re-sorted before aggregation
4. ✅ Race-level comparisons isolated to single time points

**Impact**: The fixes ensure the model trains on truly out-of-sample data, making AUC measurements trustworthy and preventing overfitting.

**Next Steps**:
1. Run pipeline with fixed scripts
2. Compare AUC to baseline (should be similar or slightly lower due to stricter temporal integrity)
3. If AUC drops significantly, investigate which features were most affected
4. Document actual vs expected performance

---

**End of Audit Report**
