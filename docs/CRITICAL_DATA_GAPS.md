# Critical Data Gaps

Analysis of current predictions reveals several critical data gaps that significantly impact prediction accuracy.

---

## 1. Pedigree/Breeding Data ✅ **COMPLETED**

### Why This Matters
- **Cold start problem**: 30-40% of predictions are for horses with limited form (< 3 runs)
- Current model falls back to defaults when `career_runs=0`
- Sire/dam statistics provide CRITICAL signal for unraced or lightly-raced horses
- UK flat racing heavily influenced by breeding (sprinter sires, stayer sires, etc.)

### Data Available from The Racing API
```json
{
  "horse": "Example Horse",
  "sire": "Frankel",
  "sire_id": "abc123",
  "dam": "Example Dam",
  "dam_id": "def456",
  "damsire": "Galileo",
  "damsire_id": "ghi789"
}
```

### ✅ **IMPLEMENTED FEATURES** (Pedigree V2)
| Feature | Description | Status |
|---------|-------------|--------|
| `sire_win_rate_v2` | Sire's progeny overall win rate (temporal) | ✅ Added |
| `sire_place_rate_v2` | Sire's progeny place rate (temporal) | ✅ Added |
| `sire_turf_win_rate` | Sire win rate on turf specifically | ✅ Added |
| `sire_aw_win_rate` | Sire win rate on all-weather specifically | ✅ Added |
| `sire_surface_pref` | Surface preference (turf_wr - aw_wr) | ✅ Added |
| `sire_avg_win_dist` | Average winning distance of sire's winners | ✅ Added |
| `sire_sprint_pct` | Percentage of sire wins at sprint (<7f) | ✅ Added |
| `sire_stayer_pct` | Percentage of sire wins at staying (>10f) | ✅ Added |
| `sire_class_avg` | Average class of sire's winners | ✅ Added |
| `dam_offspring_count` | Number of previous offspring from dam | ✅ Added |
| `dam_offspring_win_rate` | Win rate of dam's previous offspring | ✅ Added |
| `damsire_stamina_score` | Damsire stamina index (avg win dist / 10) | ✅ Added |

### Implementation Details
- **Temporal integrity**: Uses expanding windows grouped by sire/dam/damsire
- **No data leakage**: All stats use shift(1) to exclude current race
- **Rich sire data**: 1,467 unique sires with progeny performance
- **Distance specialization**: Sprint/stayer percentages for sire profiling
- **Surface specialization**: Turf vs AW preference calculations
- **Dam influence**: Previous offspring performance tracking

### Implementation Priority: 🔴 URGENT (Week 1) ✅ **DONE**

---

## 2. Pace/Running Style Analysis ✅ **COMPLETED**

### Why This Matters
- Racing is about race dynamics, not just individual ability
- A race with 5 front-runners creates different conditions vs 5 closers
- "Pace makes the race" — fundamental racing principle
- Current model has NO pace features

### ✅ **IMPLEMENTED FEATURES** (Pace Analysis)
| Feature | Description | Importance | Rank |
|---------|-------------|------------|------|
| `pace_style_leader` | Horse classified as front-runner | 0.007653 | 42/120 |
| `pace_style_closer` | Horse classified as closer | 0.007300 | 53/120 |
| `pace_style_presser` | Horse classified as presser | 0.005444 | 108/120 |
| `pace_style_midpack` | Horse classified as mid-pack runner | 0.006559 | 90/120 |
| `race_leader_count` | Number of likely leaders in race | 0.006936 | 70/120 |
| `race_closer_count` | Number of likely closers in race | 0.006225 | 101/120 |
| `pace_pressure` | Ratio of leaders to field size | - | - |
| `style_advantage` | Pace scenario suits this horse's style | 0.005075 | 110/120 |
| `sprint_specialist` | Excels at sprint distances | 0.036000 | 2/120 |
| `staying_specialist` | Excels at staying distances | 0.026600 | 5/120 |

### Implementation Details
- **Running Style Classification**: Derived from historical finishing patterns, draw preferences, and consistency
- **Race-Level Pace Analysis**: Counts leaders/closers per race, calculates pace pressure
- **Style Advantage**: Front-runners benefit from slow paces, closers from fast paces
- **Distance Specialization**: Sprint (<8f) and staying (≥12f) specialists identified
- **Temporal Integrity**: All classifications use only prior race history (no data leakage)

### Classification Logic
```python
PACE_STYLES = {
    'LEADER': Sprint races with low draw + good finishes,
    'PRESSER': Consistent top-3 finishes, low variance,
    'CLOSER': High variance in finishing positions,
    'MIDPACK': Default category
}
```

### Implementation Priority: 🔴 URGENT (Week 1-2) ✅ **DONE**

---

## 3. Jockey/Trainer Form (Current vs Career) ✅ **COMPLETED**

### Problem with Current Features
- Current model uses `jockey_career_runs` and `jockey_career_win_rate`
- Career stats are **static** — don't capture current form
- A jockey in poor form riding a good horse is a red flag

### ✅ **IMPLEMENTED FEATURES** (Connections Form V2)
| Feature | Description | Status |
|---------|-------------|--------|
| `jockey_runs_14d_v2` | Jockey rides in last 14 days | ✅ Added |
| `jockey_runs_30d_v2` | Jockey rides in last 30 days | ✅ Added |
| `jockey_win_rate_14d_v2` | Jockey win rate last 14 days | ✅ Added |
| `jockey_win_rate_30d_v2` | Jockey win rate last 30 days | ✅ Added |
| `trainer_runs_14d_v2` | Trainer runners in last 14 days | ✅ Added |
| `trainer_runs_30d_v2` | Trainer runners in last 30 days | ✅ Added |
| `trainer_win_rate_14d_v2` | Trainer win rate last 14 days | ✅ Added |
| `trainer_win_rate_30d_v2` | Trainer win rate last 30 days | ✅ Added |
| `combo_runs_30d_v2` | Jockey-trainer combo recent runs | ✅ Added |
| `combo_win_rate_30d_v2` | Jockey-trainer combo win rate | ✅ Added |

### Implementation Details
- **Temporal integrity**: Uses proper date filtering (`date < race_date`)
- **Windowed aggregations**: 14d and 30d rolling windows
- **No data leakage**: Features calculated from past races only
- **Impact**: +2.10% AUC improvement in model v2.1

### Implementation Priority: 🟠 HIGH (Week 2) ✅ **DONE**

---

## 4. Going/Ground Preference Analysis ✅ **COMPLETED**

### Current State
- `going_numeric` is a single numeric encoding
- ✅ **NEW**: Horse-specific going preference added

### ✅ **IMPLEMENTED FEATURES** (Going Preferences)
| Feature | Description | Status |
|---------|-------------|--------|
| `horse_heavy_win_rate` | Horse win rate on heavy going | ✅ Added |
| `horse_soft_win_rate` | Horse win rate on soft going | ✅ Added |
| `horse_good_win_rate` | Horse win rate on good going | ✅ Added |
| `horse_firm_win_rate` | Horse win rate on firm going | ✅ Added |
| `going_match_score` | How close is today's going to best? | ✅ Added |
| `sire_going_pref` | Sire progeny best going category | ✅ Added |

### Implementation Details
- **Temporal integrity**: Uses proper expanding windows
- **Going categorization**: heavy (0-2.49), soft (2.5-4.49), good (4.5-6.49), firm (6.5+)
- **Match score**: 1.0 for exact match, 0.5 for adjacent, 0.0 for opposite
- **Sire stats**: Aggregated from progeny performance

### Implementation Priority: 🟠 HIGH (Week 2) ✅ **DONE**

---

## 5. Official Rating (OR) Context ✅ **COMPLETED**

### Current Features
- `or_numeric` — raw OR value
- `or_change` — change from last run
- `or_trend_3` — 3-race OR trend

### ✅ **IMPLEMENTED FEATURES** (OR Context)
| Feature | Description | Status |
|---------|-------------|--------|
| `or_vs_race_max` | Horse OR vs highest rated in race | ✅ Added |
| `or_vs_race_avg` | Horse OR vs race average | ✅ Added |
| `or_vs_race_median` | Horse OR vs race median | ✅ Added |
| `or_percentile` | Where horse sits in OR distribution | ✅ Added |
| `or_advantage` | Positive difference from avg (0 if below) | ✅ Added |
| `is_top_rated` | Has highest OR in race | ✅ Added |
| `or_gap_to_top` | OR difference to highest rated | ✅ Added |
| `or_vs_class_typical` | OR vs typical for this class | ✅ Added |
| `is_well_handicapped` | OR below class average (value bet) | ✅ Added |
| `or_career_high` | Is current OR career best? | ✅ Added |
| `or_career_range` | Career OR range (max - min) | ✅ Added |
| `or_utilization` | Current OR / career high ratio | ✅ Added |
| `or_relative_strength` | OR advantage * percentile score | ✅ Added |

### Implementation Details
- **Race-level comparisons**: Real-time comparisons within each race field
- **Career context**: Historical OR tracking for improvement potential
- **Handicap assessment**: Identifies potentially well-handicapped horses
- **Combined metrics**: Interaction features (or_relative_strength)

### Implementation Priority: 🟡 MEDIUM (Week 3) ✅ **DONE**

---

## 6. Equipment Changes ✅ **COMPLETED**

### Current State
- `has_blinkers`, `has_visor` features now show non-zero importance
- `gear_changed`, `first_time_blinkers` also have importance

### ✅ **IMPLEMENTED FEATURES** (Equipment Features)
| Feature | Description | Importance | Rank |
|---------|-------------|------------|------|
| `has_blinkers` | Horse currently wearing blinkers | 0.007776 | 38/120 |
| `has_visor` | Horse currently wearing visor | 0.005953 | 106/120 |
| `first_time_blinkers` | First time wearing blinkers | 0.006426 | 97/120 |
| `gear_changed` | Headgear changed from last run | 0.006988 | 68/120 |

### Root Cause & Fix
- **Problem**: `engineer_gear_features()` was looking for 'headgear' column but dataset had 'hg'
- **Fix**: Updated function to use correct column name (`hg` from historical data)
- **Impact**: Equipment features now contribute to model predictions

### Implementation Priority: 🟡 MEDIUM (Week 2) ✅ **DONE**

---

## 7. Weight Analysis Improvements ✅ **PARTIALLY COMPLETED**

### ✅ **IMPLEMENTED FEATURES** (Basic Weight Features)
| Feature | Description | Status |
|---------|-------------|--------|
| `weight_lbs` | Weight carried in lbs | ✅ Added |
| `weight_vs_avg` | Weight vs race average | ✅ Added |
| `is_top_weight` | Is this horse top weight? | ✅ Added |
| `weight_change` | Weight change from last run | ✅ Added |

### Missing Features (Still Needed)
| Feature | Description | Impact |
|---------|-------------|--------|
| `weight_for_age` | WFA-adjusted weight | High |
| `weight_trend` | Weight carried trend over last 3 runs | Medium |
| `lb_per_length` | Historical lengths beaten per lb | Medium |
| `handicap_efficiency` | Wins per lb carried above minimum | Medium |

### Implementation Priority: 🟡 MEDIUM (Week 3) ✅ **BASIC FEATURES DONE**

---

## Summary: Priority Order

| Priority | Gap | Est. Impact on AUC | Effort | Status |
|----------|-----|-------------------|--------|--------|
| 1 | **Pedigree data** | **+0.02-0.03** | **Medium** | **✅ COMPLETED** |
| 2 | **Pace analysis** | **+0.02-0.03** | **Medium** | **✅ COMPLETED** |
| 3 | **Jockey/trainer form** | **+0.01-0.02** | **Low** | **✅ COMPLETED** |
| 4 | **Going preferences** | **+0.01-0.02** | **Low** | **✅ COMPLETED** |
| 5 | **OR context** | **+0.01** | **Low** | **✅ COMPLETED** |
| 6 | **Equipment fixes** | **+0.005-0.01** | **Low** | **✅ COMPLETED** |
| 7 | **Weight improvements** | **+0.005-0.01** | **Low** | **✅ PARTIAL** |

**Total Estimated AUC Improvement**: +0.05 to +0.08 (to ~0.72-0.75)
**Actual Improvement Achieved**:
- Baseline (v1.0): ~0.685 AUC
- v2.1 (Jockey/Trainer Form): 0.706 AUC (+0.021)
- v2.3 (Pedigree + Going + OR Context): 0.710 AUC (+0.004)
- v2.4 (Equipment Fixed + Pace Analysis): 0.710 AUC (pace features are top-5 important)

**Performance Verification (Temporal Split)**:
- Train AUC: 0.845
- Test AUC: 0.710 ✅
- Test Accuracy: 88.96%
- Features: 120 total
- Top Feature: `sprint_specialist` (pace) at rank #2

**Data Leakage Verification**: ✅ **PASSED**
- Run `python scripts/verify_no_leakage.py` after ANY feature changes
- ALWAYS use temporal split (not random split) for model evaluation
- Random split gives inflated AUC of 0.823 due to temporal leakage

**Next Steps**:
1. ✅ Equipment features fixed and retrained (has_blinkers rank 38)
2. ✅ Pace analysis implemented (sprint_specialist rank 2, staying_specialist rank 5)
3. ⏳ Measure combined AUC improvement from equipment + pace features
4. ❌ Any remaining low-hanging fruit features
