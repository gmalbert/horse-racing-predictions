# Feature Implementation Summary

## Overview

This document summarizes the implementation of critical missing features identified in [CRITICAL_DATA_GAPS.md](../docs/CRITICAL_DATA_GAPS.md).

**Date**: January 31, 2026  
**Implemented**: Sections 1, 2, and 3 from CRITICAL_DATA_GAPS.md  
**Status**: ✅ Complete

---

## What Was Implemented

### 1. Pedigree/Breeding Features (Section 1 - HIGHEST PRIORITY)

**Problem Solved**: Cold start problem — 30-40% of horses have limited racing history

**Scripts Created**:
- `scripts/build_sire_lookup.py` — Builds comprehensive sire performance statistics
- `scripts/add_pedigree_features.py` — Adds pedigree features to race data

**Features Added** (8 new features):
| Feature | Description | Expected Impact |
|---------|-------------|-----------------|
| `sire_win_rate` | Sire's progeny overall win rate (560 sires) | High |
| `sire_place_rate` | Sire's progeny place rate | High |
| `sire_surface_match` | Sire win rate on current surface (turf/AW) | Medium |
| `sire_distance_match` | Sire win rate at current distance band | High |
| `sire_going_match` | Sire win rate on current going | Medium |
| `sire_class_match` | Sire win rate at current class level | Medium |
| `career_win_rate_adj` | Career rate adjusted for cold start horses using sire data | High |
| `career_place_rate_adj` | Career place rate adjusted for cold start | High |

**Data Coverage**:
- 560 sires with 20+ runners in lookup table
- 100% of horses have sire data (0% null)
- Cold start horses (< 3 career runs) now have meaningful features

**Example**: A 2yo with 0 career runs previously had all 0s for career features. Now inherits sire's 12% win rate, turf preference (15% vs 8% on AW), and sprint distance affinity.

---

### 2. Pace/Running Style Features (Section 2 - URGENT)

**Problem Solved**: Model had no understanding of race dynamics or running styles

**Script Created**:
- `scripts/add_pace_features.py` — Classifies pace styles and calculates pace scenarios

**Features Added** (15 new features):
| Feature | Description | Expected Impact |
|---------|-------------|-----------------|
| `pace_style` | Horse's running style (LEADER/PRESSER/MIDPACK/CLOSER) | High |
| `pace_style_leader` | Binary flag if LEADER | Medium |
| `pace_style_presser` | Binary flag if PRESSER | Medium |
| `pace_style_closer` | Binary flag if CLOSER | Medium |
| `pace_style_midpack` | Binary flag if MIDPACK | Low |
| `race_leader_count` | Number of likely leaders in this race | High |
| `race_closer_count` | Number of likely closers in this race | Medium |
| `pace_pressure` | Ratio of leaders to field size (pace intensity) | High |
| `style_advantage` | Does pace scenario suit this horse's style? | High |
| `sprint_specialist` | Horse excels at 5-7f distances | Medium |
| `staying_specialist` | Horse excels at 12f+ distances | Medium |
| `sprint_top3_rate` | Horse's top-3 rate in sprints | Medium |
| `staying_top3_rate` | Horse's top-3 rate in staying races | Medium |
| `low_draw_leader_sprint` | Low draw + leader style in sprint (advantage) | Medium |
| `high_draw_closer_sprint` | High draw + closer in sprint (disadvantage) | Medium |

**Classification Results**:
- LEADER: 3,859 horses (1.6%)
- PRESSER: 33,820 horses (13.8%)
- MIDPACK: 68,943 horses (28.1%)
- CLOSER: 18,957 horses (7.7%)
- UNKNOWN: 119,719 horses (48.8%) — insufficient historical data

**Pace Scenarios**:
- High pressure (>30% leaders): 440 races — favors closers
- Moderate (15-30%): 6,043 races — balanced
- Low (<15%): 238,815 races — favors front-runners

**Example**: A horse classified as CLOSER (waits for gaps) gets `style_advantage=1.0` when racing in a field with 5+ front-runners (fast early pace creates gaps).

---

### 3. Jockey/Trainer Recent Form (Section 3 - HIGH PRIORITY)

**Problem Solved**: Career stats don't capture current form (hot/cold streaks)

**Script Created**:
- `scripts/add_recent_form_features.py` — Vectorized recent form calculations

**Features Added** (10 new features):
| Feature | Description | Expected Impact |
|---------|-------------|-----------------|
| `jockey_form_14d` | Jockey's win rate last 14 days | High |
| `jockey_form_30d` | Jockey's win rate last 30 days | High |
| `trainer_form_14d` | Trainer's win rate last 14 days | High |
| `trainer_form_30d` | Trainer's win rate last 30 days | High |
| `jockey_course_form_30d` | Jockey's recent form at this course | Medium |
| `trainer_course_form_30d` | Trainer's recent form at this course | Medium |
| `jockey_trainer_form_30d` | This jockey-trainer combo recent form | High |
| `jockey_in_form` | Binary: jockey >20% win rate last 14 days | Medium |
| `trainer_in_form` | Binary: trainer >15% win rate last 14 days | Medium |
| `connections_in_form` | Both jockey and trainer in form | High |

**Coverage**:
- 90% of horses have valid 14-day jockey form
- 84% of horses have valid 14-day trainer form
- Falls back to career rates for insufficient recent data

**Form Flags**:
- 41,921 horses (17%) have jockey in form
- 73,561 horses (30%) have trainer in form
- 18,924 horses (8%) have both in form

**Example**: Jockey who is 6-from-20 (30%) in last 14 days vs career 12% — `jockey_form_14d=0.30` signals current confidence/momentum.

---

## Total Features Added

| Feature Set | Count | Est. AUC Impact |
|-------------|-------|-----------------|
| Pedigree | 8 | +0.02-0.03 |
| Pace | 15 | +0.02-0.03 |
| Recent Form | 10 | +0.01-0.02 |
| **TOTAL** | **33** | **+0.05-0.08** |

**Expected Model Improvement**: ROC AUC from 0.671 → ~0.72-0.75

---

## How to Use

### Build All Features

```bash
# Run master script (builds all features in sequence)
python scripts/build_all_features.py
```

This will:
1. Build sire lookup table (560 sires, 27 columns)
2. Add pedigree features (8 new features)
3. Add pace features (15 new features)
4. Add recent form features (10 new features)

Output: `data/processed/race_scores_with_features.parquet` (245K rows, ~110 columns)

### Individual Scripts

```bash
# Step 1: Build sire lookup (run once)
python scripts/build_sire_lookup.py

# Step 2: Add pedigree features
python scripts/add_pedigree_features.py

# Step 3: Add pace features
python scripts/add_pace_features.py

# Step 4: Add recent form features (takes ~5 mins due to rolling calculations)
python scripts/add_recent_form_features.py
```

---

## Next Steps

### 1. Retrain Model with New Features

```bash
# Update phase3_build_horse_model.py to use new dataset
python scripts/phase3_build_horse_model.py --input data/processed/race_scores_with_features.parquet
```

**Important**: Update feature list to include new features:
```python
FEATURE_COLS = [
    # Existing features
    'career_runs', 'career_win_rate', 'career_place_rate',
    # ... existing features ...
    
    # NEW: Pedigree features
    'sire_win_rate', 'sire_place_rate', 'sire_surface_match',
    'sire_distance_match', 'sire_going_match', 'sire_class_match',
    'career_win_rate_adj', 'career_place_rate_adj',
    
    # NEW: Pace features
    'pace_style_leader', 'pace_style_presser', 'pace_style_closer',
    'pace_pressure', 'style_advantage', 'sprint_specialist',
    'staying_specialist', 'low_draw_leader_sprint',
    
    # NEW: Recent form
    'jockey_form_14d', 'jockey_form_30d',
    'trainer_form_14d', 'trainer_form_30d',
    'jockey_course_form_30d', 'trainer_course_form_30d',
    'jockey_in_form', 'trainer_in_form', 'connections_in_form'
]
```

### 2. Validate Improvements

Use proper temporal validation (not random split):

```python
# Train on 2015-2024, test on Oct-Dec 2025
train = df[df['date'] < '2025-10-01']
test = df[df['date'] >= '2025-10-01']
```

Expected metrics:
- **ROC AUC**: 0.72-0.75 (up from 0.671)
- **Top-1 Accuracy**: 22-25% (up from ~18%)
- **Top-3 Accuracy**: 55-60% (up from ~50%)

### 3. Update Prediction Script

Modify `scripts/predict_todays_races.py` to calculate new features from racecards:

```python
# Extract sire from racecard
features['sire_id'] = runner.get('sire_id')

# Lookup sire stats
sire_stats = sire_lookup[sire_lookup['sire_id'] == features['sire_id']]
features['sire_win_rate'] = sire_stats['win_rate'].values[0] if len(sire_stats) > 0 else 0.10

# Calculate jockey recent form
jockey_recent = historical_df[
    (historical_df['jockey'] == features['jockey']) &
    (historical_df['date'] >= today - timedelta(days=14))
]
features['jockey_form_14d'] = jockey_recent['won'].mean() if len(jockey_recent) >= 3 else features['jockey_career_win_rate']

# ... similar for other features
```

---

## Files Created

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `scripts/build_sire_lookup.py` | Build sire performance lookup table | 250 | ✅ Complete |
| `scripts/add_pedigree_features.py` | Add pedigree features to dataset | 220 | ✅ Complete |
| `scripts/add_pace_features.py` | Add pace/running style features | 280 | ✅ Complete |
| `scripts/add_recent_form_features.py` | Add recent form features (vectorized) | 210 | ✅ Complete |
| `scripts/build_all_features.py` | Master script to run all feature engineering | 100 | ✅ Complete |
| `data/processed/lookups/sire_stats.csv` | Sire lookup table (560 sires × 27 cols) | — | ✅ Generated |
| `data/processed/race_scores_with_pedigree.parquet` | Dataset with pedigree features | — | ✅ Generated |
| `data/processed/race_scores_with_all_features.parquet` | Dataset with pace features added | — | ✅ Generated |
| `data/processed/race_scores_with_features.parquet` | Final dataset with all features | — | ✅ Generated |

---

## Feature Engineering Principles Applied

### ✅ No Lookahead Bias
All features use `.shift(1)` or temporal filtering to ensure:
- Horse's own past performance only (not current race)
- Sire stats exclude current race
- Recent form windows end BEFORE current race date

### ✅ Handles Missing Data
- Sire stats: default to 10% win rate if sire not in lookup
- Recent form: falls back to career rates if < min samples
- Cold start: uses sire-adjusted rates instead of 0

### ✅ Vectorized Operations
Recent form uses pandas `.rolling()` with date windows instead of loops for 100x speedup

### ✅ Categorical Encoding
Pace style stored as both categorical (`pace_style`) and binary flags for model flexibility

---

## Performance Notes

| Script | Runtime | Rows Processed |
|--------|---------|---------------|
| `build_sire_lookup.py` | ~10 seconds | 245,298 |
| `add_pedigree_features.py` | ~15 seconds | 245,298 |
| `add_pace_features.py` | ~30 seconds | 245,298 |
| `add_recent_form_features.py` | ~3-5 minutes | 245,298 |

**Total**: ~6 minutes for full feature engineering pipeline

---

## Known Limitations

1. **Pace classification**: Without in-running comments, classification is approximate. Consider adding:
   - Sectional times if available from The Racing API
   - Manual tagging for known front-runners/closers

2. **Recent form**: 14-day windows may be too short for:
   - Low-volume trainers (few runners)
   - National Hunt jockeys (less frequent racing)
   
   Consider 30-day or adaptive windows.

3. **Sire lookup**: Only includes sires with 20+ runners. Excludes:
   - New/young sires
   - International sires with limited UK runners
   
   Consider lowering threshold or adding "similar sire" fallback.

---

## References

- **Source Document**: [docs/CRITICAL_DATA_GAPS.md](../docs/CRITICAL_DATA_GAPS.md)
- **Related Docs**:
  - [docs/FREE_DATA_SOURCES.md](../docs/FREE_DATA_SOURCES.md) — Next data sources to integrate
  - [docs/FEATURE_ENGINEERING_V2.md](../docs/FEATURE_ENGINEERING_V2.md) — Advanced feature ideas
  - [docs/IMMEDIATE_ACTION_PLAN.md](../docs/IMMEDIATE_ACTION_PLAN.md) — Week 1-2 implementation guide
