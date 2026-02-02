# Feature Implementation v2.2: Going Preferences + OR Context

**Date**: 2025-01-31  
**Model Version**: v2.2 (pending retrain)  
**Feature Count**: 91 → 110 features (+19 features)

---

## Overview

This document tracks the implementation of **going preference** and **OR context** features to address critical data gaps identified in [CRITICAL_DATA_GAPS.md](CRITICAL_DATA_GAPS.md).

---

## New Features Added

### 1. Going Preference Features (+6 features)

**Script**: `scripts/add_going_preference_features.py`

| Feature | Description | Temporal Integrity |
|---------|-------------|-------------------|
| `horse_heavy_win_rate` | Horse's win rate on heavy going (0-2.49) | ✅ Expanding window |
| `horse_soft_win_rate` | Horse's win rate on soft going (2.5-4.49) | ✅ Expanding window |
| `horse_good_win_rate` | Horse's win rate on good going (4.5-6.49) | ✅ Expanding window |
| `horse_firm_win_rate` | Horse's win rate on firm going (6.5+) | ✅ Expanding window |
| `going_match_score` | Preference match: 1.0 exact, 0.5 adjacent, 0.0 opposite | ✅ Calculated |
| `sire_going_pref` | Sire's progeny best going category (0=heavy, 3=firm) | ✅ Calculated |

**Going Categorization**:
```python
0-2.49:  heavy (worst conditions)
2.5-4.49: soft
4.5-6.49: good (standard)
6.5+:    firm (fast ground)
```

**Match Score Logic**:
- Same category = 1.0
- Adjacent category = 0.5
- 2+ categories apart = 0.0

**Expected Impact**: +0.01-0.02 AUC (important for horses with strong going preferences)

---

### 2. Official Rating Context Features (+13 features)

**Script**: `scripts/add_or_context_features.py`

#### Race-Level Comparisons (+7 features)
| Feature | Description |
|---------|-------------|
| `or_vs_race_max` | Difference from highest-rated horse in race |
| `or_vs_race_avg` | Difference from race average OR |
| `or_vs_race_median` | Difference from race median OR |
| `or_percentile` | Horse's OR percentile within race (0-100) |
| `or_advantage` | Positive advantage over avg (0 if below avg) |
| `is_top_rated` | Boolean: has highest OR in race |
| `or_gap_to_top` | Gap to highest-rated (always ≤ 0) |

#### Career Context (+4 features)
| Feature | Description |
|---------|-------------|
| `or_vs_class_typical` | OR vs typical for this race class |
| `is_well_handicapped` | Boolean: OR below class average (undervalued) |
| `or_career_high` | Boolean: currently at career-best OR |
| `or_career_range` | Career OR range (max - min) |

#### Interaction Features (+2 features)
| Feature | Description |
|---------|-------------|
| `or_utilization` | Current OR / career high (deterioration indicator) |
| `or_relative_strength` | or_advantage × or_percentile (combined strength) |

**Expected Impact**: +0.01 AUC (improves handicap race predictions)

---

## Pipeline Updates

### 1. Feature Generation Pipeline

**regenerate_all.ps1** now has **9 steps** (was 6):

```powershell
Step 1: Data Cleaning           (phase1_data_cleaning.py)
Step 2: Race Scoring            (phase2_score_races.py)
Step 3: Enhanced Form           (add_enhanced_form_features.py)
Step 4: Connections V2          (add_connections_form_v2.py)
Step 5: Going Preferences       (add_going_preference_features.py)  ← NEW
Step 6: OR Context              (add_or_context_features.py)        ← NEW
Step 7: Model Training          (phase3_build_horse_model.py)
Step 8: Predictions Generation  (batch_generate_predictions.py)
Step 9: Betting Strategy        (apply_betting_strategy.py)
```

### 2. GitHub Actions Workflow

**`.github/workflows/weekly_model_training.yml`** updated to include:
- Step: Add going preference features
- Step: Add OR context features

Both steps run before model training in the weekly Monday 07:00 UTC job.

### 3. Model Training Script

**`scripts/phase3_build_horse_model.py`** updated to prefer:
1. `race_scores_or_context.parquet` (all features)
2. `race_scores_going_pref.parquet` (going only)
3. `race_scores_connections_v2.parquet` (v2.1 baseline)

---

## Data Flow Diagram

```
all_gb_races.parquet (245K rows)
  ↓
race_scores.parquet (Phase 2 scoring)
  ↓
race_scores_enhanced_form.parquet (+6 features)
  ↓
race_scores_connections_v2.parquet (+13 features) ← v2.1 baseline (91 features)
  ↓
race_scores_going_pref.parquet (+6 features)      ← NEW
  ↓
race_scores_or_context.parquet (+13 features)     ← NEW (110 features total)
  ↓
horse_win_predictor.json (XGBoost model)
  ↓
predictions_YYYY-MM-DD.csv (daily predictions)
```

---

## Testing & Validation

### Compilation Status
- ✅ `add_going_preference_features.py` — compiles successfully
- ✅ `add_or_context_features.py` — compiles successfully
- ✅ `phase3_build_horse_model.py` — updated data loader
- ✅ `regenerate_all.ps1` — 9-step pipeline structure
- ✅ `weekly_model_training.yml` — workflow updated

### Next Steps
1. ⏳ **Run feature generation**: Execute Steps 5-6 on existing data
2. ⏳ **Retrain model**: Run phase3_build_horse_model.py with 110 features
3. ⏳ **Measure impact**: Compare v2.2 AUC to v2.1 baseline (0.706)
4. ⏳ **Update gaps doc**: Record actual AUC improvement in CRITICAL_DATA_GAPS.md

---

## Expected Performance

### Model v2.1 Baseline
- **Features**: 91
- **AUC**: 0.706
- **Improvement over v2.0**: +0.021 (connections form)

### Model v2.2 Target
- **Features**: 110 (+19)
- **Expected AUC**: 0.716-0.726
- **Expected Improvement**: +0.01-0.02 (going + OR features)

### Cumulative Progress
- **v2.0**: 0.685 AUC (72 features)
- **v2.1**: 0.706 AUC (+21 enhanced form + connections)
- **v2.2**: 0.716-0.726 AUC (target, +19 going + OR features)
- **Remaining potential**: +0.02-0.03 (pedigree + pace features)

---

## Feature Quality Checks

### Going Preference Features
- ✅ Temporal integrity: Uses `shift(1).expanding()` to avoid lookahead
- ✅ Going categorization: 4 categories (heavy/soft/good/firm)
- ✅ Sire integration: Aggregates progeny performance by going type
- ✅ Match scoring: Quantifies preference alignment (0.0-1.0)

### OR Context Features
- ✅ Race-level comparisons: Real-time field analysis
- ✅ Career context: Historical OR tracking
- ✅ Handicap assessment: Identifies value bets (well-handicapped)
- ✅ Interaction features: Combined strength metrics

---

## Documentation Updates

- ✅ [CRITICAL_DATA_GAPS.md](CRITICAL_DATA_GAPS.md) — marked items 4-5 as completed
- ✅ [regenerate_all.ps1](../regenerate_all.ps1) — updated to 9 steps
- ✅ [weekly_model_training.yml](../.github/workflows/weekly_model_training.yml) — added new steps
- ✅ [phase3_build_horse_model.py](../scripts/phase3_build_horse_model.py) — updated data loader
- ✅ This document — feature implementation tracking

---

## Remaining Priority Items

From [CRITICAL_DATA_GAPS.md](CRITICAL_DATA_GAPS.md):

| Priority | Feature Set | Est. Impact | Status |
|----------|-------------|-------------|--------|
| 1 | Pedigree/breeding data | +0.02-0.03 | ❌ Not implemented |
| 2 | Pace/running style | +0.02-0.03 | ❌ Not implemented |
| 3 | Equipment changes fix | +0.005-0.01 | ❌ Not implemented |

**Total remaining potential**: +0.04-0.07 AUC (to ~0.75-0.79)

---

## Author Notes

This implementation completes the **medium-priority data gaps** (going preferences + OR context) identified in the original gap analysis. 

**Key design decisions**:
1. **Temporal integrity**: All features use expanding windows to prevent data leakage
2. **Modular pipeline**: Each feature set is a separate script for maintainability
3. **Backward compatibility**: Model training script prefers latest data but falls back gracefully
4. **Going categorization**: Simple 4-category system balances granularity with sample size

**Next priorities**:
1. Retrain model to measure actual v2.2 impact
2. Implement pedigree features (highest remaining ROI)
3. Implement pace analysis (high complexity, high impact)

---

**End of Document**
