# RacingEdgeUK Signal Framework — Enhancements for the European Model

**Source**: Analysis of [RacingEdgeUK methodology post](https://old.reddit.com/r/RacingEdgeUK/comments/1teprnx/) and the [Alfred Boucher case study](https://old.reddit.com/r/RacingEdgeUK/comments/1tephr9/) (May 2026).

---

## Overview

The RacingEdgeUK posts describe a five-signal selection framework (S1–S5) plus a stake-sizing confidence/warning layer (C1–C11, W1–W10). The methodology surfaces horses the market systematically underestimates by anchoring on career-best performance **in today's specific conditions** (course, distance, going, class, weight) rather than on headline recent form.

After mapping the framework against the current codebase, **eight concrete feature gaps** were identified where implementing the framework's logic would add signal the XGBoost model currently cannot see. Each gap maps to one or more of the five signals.

---

## Gap Analysis: What Is Missing

| # | Signal | What's Missing | Current Closest Feature |
|---|--------|---------------|------------------------|
| 1 | S1, S3 | Horse's own peak figure split by **today's surface** (turf vs AW/dirt) | `sire_surface_pref` (sire only) |
| 2 | S1 | Peak figure over last 12 months **ranked against the field** | `is_highest_rated` (OR only, no 12m window) |
| 3 | S2 | **LTO figure vs career peak** ratio (how close last run was to ceiling) | `or_below_career_high` (OR context, not race figure) |
| 4 | S5 | **Multi-condition fit score** — count of distance/going/class/weight where horse leads field | None |
| 5 | S3 | **Surface-suppression flag** — recent runs on wrong surface pulling headline figures down | None |
| 6 | S2/B | **Second-run-after-break** flag — first run back is often a prep, second is live | `days_since_last` (no look-back DLR) |
| 7 | B | **Trainer RTF (Run-to-Form)** — place rate %, better than win% for low-volume yards | `trainer_form_14d` (win% only) |
| 8 | S4 | **Improver vs field guard** — improver is still well short of race best (avoid flagging weak improvers in strong fields) | `form_trend` (no race-relative guard) |

---

## Proposed Implementations

All code follows the project's leakage-prevention conventions:
- `shift(1)` before any cumulative/expanding aggregation
- Sort by `[group_var, date]` before calculations
- Pure functions where possible
- After adding, run `python scripts/verify_no_leakage.py`

---

### 1. Horse Surface Peak Figure (S1 / S3 — Return-to-Surface)

The core of the Alfred Boucher case: his AW RPR sequence (89-94-88) was buried under poor turf runs (28-70-74). No existing feature captures this at the **horse** level.

Add to `scripts/add_going_preference_features.py` or create `scripts/add_surface_peak_features.py`:

```python
import pandas as pd
import numpy as np

def add_horse_surface_peak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute horse's career-peak figure split by surface (turf vs AW/all-weather).
    
    S1/S3 signal: if a horse's peak on today's surface is close to its overall ceiling
    but recent runs were on the wrong surface, this catches the suppression effect.
    
    Temporal integrity: uses shift(1) before expanding().max() to prevent leakage.
    Sort df by [horse, date] before calling.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()

    # Carrier figure: use 'rpr' if available, fall back to 'or_numeric'
    fig_col = 'rpr' if 'rpr' in df.columns else 'or_numeric'

    # Surface-specific expanding peak (exclude current race via shift)
    df['_fig_if_turf'] = df[fig_col].where(df['is_turf'] == 1)
    df['_fig_if_aw']   = df[fig_col].where(df['is_turf'] == 0)

    df['horse_turf_peak'] = (
        df.groupby('horse')['_fig_if_turf']
          .transform(lambda x: x.shift(1).expanding().max())
    )
    df['horse_aw_peak'] = (
        df.groupby('horse')['_fig_if_aw']
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Overall career peak figure (prior races only)
    df['horse_overall_peak'] = (
        df.groupby('horse')[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Surface return advantage:
    #   positive = horse's peak on today's surface >= 90% of overall ceiling
    #   (horse is near its best when surface conditions match)
    today_surface_peak = df['horse_turf_peak'].where(df['is_turf'] == 1, df['horse_aw_peak'])
    df['surface_peak_ratio'] = today_surface_peak / df['horse_overall_peak'].replace(0, np.nan)
    df['surface_peak_ratio'] = df['surface_peak_ratio'].fillna(0.5)

    # Surface suppression flag (S3 signal):
    #   recent run was on opposite surface, and horse has a significantly higher
    #   peak on today's surface than on the surface it just ran on
    df['_prev_is_turf'] = df.groupby('horse')['is_turf'].shift(1)
    surface_switched = df['is_turf'] != df['_prev_is_turf']
    df['surface_switch'] = surface_switched.astype(int)

    # Surface suppression: switched back to preferred surface AND surface_peak_ratio high
    df['surface_return_advantage'] = (
        (df['surface_switch'] == 1) & (df['surface_peak_ratio'] >= 0.90)
    ).astype(int)

    # Recent form figure on "wrong" surface vs career peak on today's surface
    # (large gap = possible suppression effect)
    df['_lto_fig'] = df.groupby('horse')[fig_col].shift(1)
    df['_prev_was_wrong_surface'] = (
        (df['_prev_is_turf'] != df['is_turf']) & df['_lto_fig'].notna()
    )
    df['surface_suppression_flag'] = (
        df['_prev_was_wrong_surface'] & (today_surface_peak - df['_lto_fig'] >= 8)
    ).astype(int)

    # Clean up temp columns
    df.drop(columns=['_fig_if_turf', '_fig_if_aw', '_prev_is_turf', '_lto_fig',
                     '_prev_was_wrong_surface'], inplace=True, errors='ignore')

    return df
```

**Features added**: `horse_turf_peak`, `horse_aw_peak`, `horse_overall_peak`, `surface_peak_ratio`, `surface_switch`, `surface_return_advantage`, `surface_suppression_flag`

---

### 2. Peak-12m Figure Ranked Against the Field (S1 — Peak Speed Leader)

The S1 signal requires a horse to hold the **highest peak figure in the race over the last 12 months** AND be **top rated at today's course** by the same metric. Currently `is_highest_rated` only covers the current OR, not a rolling 12-month peak.

Add to `scripts/add_or_context_features.py` or as a new post-processing step:

```python
def add_peak_figure_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute race-relative ranking of each horse's peak figure (12-month window)
    and course-specific peak figure.

    S1 fires when:
      - peak_12m_rank_in_race == 1  (best in field over last 12m)
      - peak_course_rank_in_race == 1  (best at today's course)
      - peak_12m_lead_over_2nd >= threshold (configurable, default 3 OR points)

    Temporal integrity: peak figures are pre-computed with shift(1); race-relative
    rank is computed within the race group (same date/course/off), so no cross-race
    leakage.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'rpr' if 'rpr' in df.columns else 'or_numeric'
    PEAK_LEAD_THRESHOLD = 3  # minimum points above second-best to fire S1

    # 12-month rolling peak (prior races only)
    df['horse_peak_12m'] = (
        df.groupby('horse')[fig_col]
          .transform(lambda x: x.shift(1).rolling('365D', min_periods=1).max())
    )

    # Course-specific career peak (prior races at this course only)
    df['_fig_at_course'] = df[fig_col].where(
        df.groupby('horse')['course_clean'].shift(1) == df['course_clean']
    )
    df['horse_peak_at_course'] = (
        df.groupby(['horse', 'course_clean'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Race-relative ranking (within race = same date/course/off)
    race_key = ['date_dt', 'course_clean', 'off']
    df['peak_12m_rank_in_race'] = (
        df.groupby(race_key)['horse_peak_12m']
          .rank(ascending=False, method='min')
    )
    df['is_peak_12m_leader'] = (df['peak_12m_rank_in_race'] == 1).astype(int)

    df['peak_course_rank_in_race'] = (
        df.groupby(race_key)['horse_peak_at_course']
          .rank(ascending=False, method='min', na_option='bottom')
    )
    df['is_peak_course_leader'] = (df['peak_course_rank_in_race'] == 1).astype(int)

    # Gap over second-best in field (S1 requires a minimum gap)
    race_second_best = (
        df.groupby(race_key)['horse_peak_12m']
          .transform(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else x.max())
    )
    df['peak_12m_lead_over_2nd'] = (df['horse_peak_12m'] - race_second_best).clip(lower=0)
    df['s1_peak_leader_signal'] = (
        (df['is_peak_12m_leader'] == 1) &
        (df['is_peak_course_leader'] == 1) &
        (df['peak_12m_lead_over_2nd'] >= PEAK_LEAD_THRESHOLD)
    ).astype(int)

    df.drop(columns=['_fig_at_course'], inplace=True, errors='ignore')
    return df
```

**Features added**: `horse_peak_12m`, `horse_peak_at_course`, `peak_12m_rank_in_race`, `is_peak_12m_leader`, `peak_course_rank_in_race`, `is_peak_course_leader`, `peak_12m_lead_over_2nd`, `s1_peak_leader_signal`

---

### 3. LTO Figure vs Career Peak Gap (S2 — High LTO + Strong Conditions)

S2 requires the last-time-out figure to sit within ~5 OR points of the horse's own career peak. This is different from `or_below_career_high` (which measures the **current** OR against peak, not the **last run's performance**).

Add to `scripts/phase3_build_horse_model.py` near the existing LTO form features:

```python
def add_lto_vs_peak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the gap between the horse's LTO figure and its career peak figure.
    
    S2 fires when LTO is within the configured window of the career peak
    (i.e., the horse ran close to its best last time and is in current form).
    
    Temporal integrity: career peak uses shift(1) on the current race; the LTO
    figure is shift(1) of the performance figure, so both exclude today's race.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'rpr' if 'rpr' in df.columns else 'or_numeric'
    LTO_NEAR_PEAK_WINDOW = 5  # points within career peak to qualify as "near peak"

    df['lto_figure'] = df.groupby('horse')[fig_col].shift(1)
    df['horse_career_peak_fig'] = (
        df.groupby('horse')[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    df['lto_vs_peak_gap'] = df['horse_career_peak_fig'] - df['lto_figure']
    df['lto_near_peak'] = (df['lto_vs_peak_gap'] <= LTO_NEAR_PEAK_WINDOW).astype(int)

    # LTO figure percentile within the race it was run in (was it a quality run?)
    # This is a static lookup, safe to compute after shift already applied.
    df['lto_figure_pct_career'] = (
        df['lto_figure'] / df['horse_career_peak_fig'].replace(0, np.nan)
    ).fillna(0)

    return df
```

**Features added**: `lto_figure`, `horse_career_peak_fig`, `lto_vs_peak_gap`, `lto_near_peak`, `lto_figure_pct_career`

---

### 4. Multi-Condition Fit Score ("Gold Pills" Count — S5 / Alfred Boucher)

The single most actionable insight from the Alfred Boucher case study. Four gold pills (distance/going/class/weight all field-best) identified the 28/1 winner that every "recency bias" punter overlooked. This composite feature directly encodes how many conditions a horse dominates in.

Add to `scripts/add_or_context_features.py` after race-level OR context is built, or as a new `scripts/add_condition_fit_score.py`:

```python
def add_condition_fit_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 'gold pill' multi-condition fit score — how many of the five
    key race conditions (distance, going, class, weight, course) this horse
    leads the field on, based on career-best figures in each condition.

    A horse scoring 4-5 here is the S5 / Alfred Boucher pattern:
    consistent across all conditions without necessarily dominating any one metric.

    Temporal integrity: all condition-specific career bests use shift(1).
    Race-relative ranks are within-race group, no cross-race leakage.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'rpr' if 'rpr' in df.columns else 'or_numeric'
    race_key = ['date_dt', 'course_clean', 'off']

    # --- Career-best figure at today's distance band (prior races only) ---
    df['horse_peak_at_distance'] = (
        df.groupby(['horse', 'distance_band'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # --- Career-best figure at today's going category (prior races only) ---
    df['going_category'] = df.get('going_category',
        pd.Categorical(df['going_numeric'].round().astype(str)))
    df['horse_peak_at_going'] = (
        df.groupby(['horse', 'going_category'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # --- Career-best figure at today's class (prior races only) ---
    df['horse_peak_at_class'] = (
        df.groupby(['horse', 'class_num'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # --- Weight-adjusted peak figure (proxy: used or_below_career_high context) ---
    # "Best in field on weight" = horse ran its peak under a comparable weight burden
    # Use `weight_vs_avg`: horse's peak when it was NOT the top weight, or within 2lb of today's burden
    # Simplification: use horse_peak_at_course from feature set 2, or fall back to overall peak
    df['horse_peak_at_weight_band'] = df.get(
        'horse_peak_at_course',
        df.groupby('horse')[fig_col].transform(lambda x: x.shift(1).expanding().max())
    )

    # --- Race-relative ranks for each condition ---
    for condition_col, rank_col in [
        ('horse_peak_at_distance', 'dist_peak_rank'),
        ('horse_peak_at_going',    'going_peak_rank'),
        ('horse_peak_at_class',    'class_peak_rank'),
        ('horse_peak_at_course',   'course_peak_rank'),   # requires feature set 2
        ('horse_peak_at_weight_band', 'weight_peak_rank'),
    ]:
        if condition_col in df.columns:
            df[rank_col] = (
                df.groupby(race_key)[condition_col]
                  .rank(ascending=False, method='min', na_option='bottom')
            )

    # --- Gold pill count: how many of these 5 conditions the horse is #1 in field ---
    rank_cols = ['dist_peak_rank', 'going_peak_rank', 'class_peak_rank',
                 'course_peak_rank', 'weight_peak_rank']
    available_rank_cols = [c for c in rank_cols if c in df.columns]
    df['gold_pill_count'] = (df[available_rank_cols] == 1).sum(axis=1)

    # Top-2 pill count (field-best or second-best across conditions)
    df['top2_pill_count'] = (df[available_rank_cols] <= 2).sum(axis=1)

    # S5 multi-condition leader flag: 3+ gold pills (strong multi-condition fit)
    df['s5_multi_condition_leader'] = (df['gold_pill_count'] >= 3).astype(int)

    return df
```

**Features added**: `horse_peak_at_distance`, `horse_peak_at_going`, `horse_peak_at_class`, `dist_peak_rank`, `going_peak_rank`, `class_peak_rank`, `weight_peak_rank`, `gold_pill_count`, `top2_pill_count`, `s5_multi_condition_leader`

---

### 5. Second-Run-After-Break Flag (Phase B Confidence Chip C-equivalent)

The posts explicitly note the distinction: *"second run after a break (the live one) vs first (the prep)"*. No current feature encodes whether a horse is exiting its first comeback run, which is a known trainer pattern signal.

Add to `scripts/phase3_build_horse_model.py` near the `days_since_last` block:

```python
def add_second_run_after_break_feature(df: pd.DataFrame, break_threshold_days: int = 60) -> pd.DataFrame:
    """
    Flag horses running for the second time after an extended layoff.
    
    In UK/Irish racing, many trainers use the first run back as a fitness run
    (the "prep"), with the second run being the intended target.
    
    break_threshold_days: minimum gap that qualifies as a "break" (default 60 days).
    
    Temporal integrity: uses shift(1) and shift(2) — no current race data used.
    Sort df by [horse, date_dt] before calling.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()

    df['days_since_last']    = (
        df['date_dt'] - df.groupby('horse')['date_dt'].shift(1)
    ).dt.days.fillna(60)

    df['days_before_prev']   = (
        df.groupby('horse')['date_dt'].shift(1)
        - df.groupby('horse')['date_dt'].shift(2)
    ).dt.days

    # First run back: current race follows a long break
    df['is_first_run_after_break'] = (
        df['days_since_last'] >= break_threshold_days
    ).astype(int)

    # Second run back: previous race was a comeback (days_before_prev was a long gap)
    df['prev_was_first_run_back'] = (
        df['days_before_prev'] >= break_threshold_days
    ).fillna(False).astype(int)

    df['is_second_run_after_break'] = (
        (df['is_first_run_after_break'] == 0) &  # not still in first run
        (df['prev_was_first_run_back'] == 1)      # but previous run WAS first run back
    ).astype(int)

    return df
```

**Features added**: `is_first_run_after_break`, `prev_was_first_run_back`, `is_second_run_after_break`

---

### 6. Trainer RTF (Run-to-Form / Place Rate) for Low-Volume Yards

The posts explicitly call out that raw 14-day win% is misleading for low-volume trainers. A trainer like Owen Burrows running 5 horses a fortnight with a 57% RTF (runners finishing in the frame) is far from "cold" even after a 14-day winless streak.

Add to `scripts/add_recent_form_features.py` alongside the existing trainer win-rate logic:

```python
def add_trainer_rtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trainer Run-to-Form (RTF) metrics alongside win rate.
    
    RTF = proportion of runners finishing in the top 3 (or top 4 in fields of 8+).
    More stable than win% for low-volume yards; avoids cold/hot misclassification.
    
    Temporal integrity: shift(1) applied inside rolling window.
    Sort df by [trainer, date_dt] before calling.
    """
    df = df.sort_values(['trainer', 'date_dt']).copy()

    # Binary: finished in frame (top 3, or top 4 in big fields)
    df['in_frame'] = (
        (df['pos_clean'] <= 3) |
        ((df['field_size'] >= 8) & (df['pos_clean'] <= 4))
    ).astype(int)

    # 14-day and 30-day RTF with a minimum of 3 runners to be meaningful
    df['trainer_rtf_14d'] = (
        df.groupby('trainer')['in_frame']
          .transform(lambda x: x.shift(1).rolling('14D', min_periods=3).mean())
    )
    df['trainer_rtf_30d'] = (
        df.groupby('trainer')['in_frame']
          .transform(lambda x: x.shift(1).rolling('30D', min_periods=5).mean())
    )

    # Volume-weighted quality: trainers with <10 runners in 30d use career RTF
    df['trainer_runner_count_30d'] = (
        df.groupby('trainer')['in_frame']
          .transform(lambda x: x.shift(1).rolling('30D', min_periods=1).count())
    )
    df['trainer_career_rtf'] = (
        df.groupby('trainer')['in_frame']
          .transform(lambda x: x.shift(1).expanding().mean())
    )
    # Best estimate: recent if enough volume, career otherwise
    df['trainer_rtf_best'] = df['trainer_rtf_30d'].where(
        df['trainer_runner_count_30d'] >= 10,
        df['trainer_career_rtf']
    )

    # RTF in-form flag: >40% runners in frame in last 30d
    df['trainer_rtf_in_form'] = (df['trainer_rtf_best'] >= 0.40).astype(int)

    return df
```

**Features added**: `in_frame`, `trainer_rtf_14d`, `trainer_rtf_30d`, `trainer_runner_count_30d`, `trainer_career_rtf`, `trainer_rtf_best`, `trainer_rtf_in_form`

---

### 7. Improver vs Field Guard (S4 — Improvers with Race-Best Check)

The S4 signal has an explicit guard rail: *"don't flag improvers who are still well off the race best."* The current `form_trend` feature has no such guard, meaning an improving maiden could be flagged in a Group 1.

Add to `scripts/add_enhanced_form_features.py` alongside `form_trend`:

```python
def add_improver_with_guard_features(df: pd.DataFrame, 
                                      min_gain: float = 5.0,
                                      race_best_max_gap: float = 15.0) -> pd.DataFrame:
    """
    S4 Improver signal with race-best guard rail.
    
    Fires when:
      1. Horse shows upward figure trend over recent window (min_gain over last 3 races)
      2. Horse is NOT still well below the race best (within race_best_max_gap points)
    
    min_gain: minimum OR gain over 3-race window to qualify as an improver (default 5)
    race_best_max_gap: if horse is more than this many points below the race peak 12m figure,
                       the improver flag is suppressed (default 15)
    
    Temporal integrity: all window calcs use shift(1). Race-relative gap uses
    horse_peak_12m from feature set 2 (must be computed first).
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'rpr' if 'rpr' in df.columns else 'or_numeric'

    # Recent figure trend over last 3 prior races
    df['_fig_3_races_ago'] = df.groupby('horse')[fig_col].shift(3)
    df['_fig_lto']         = df.groupby('horse')[fig_col].shift(1)
    df['recent_fig_gain']  = (df['_fig_lto'] - df['_fig_3_races_ago']).fillna(0)
    df['is_improving']     = (df['recent_fig_gain'] >= min_gain).astype(int)

    # Sub-condition for lightly raced horses (< 5 career runs): lower threshold
    lightly_raced = df['career_runs'] < 5
    df['is_improving'] = df['is_improving'].where(
        ~lightly_raced,
        (df['recent_fig_gain'] >= min_gain * 0.5).astype(int)
    )

    # Race-best guard: suppress if still too far from field leader
    # Requires horse_peak_12m from add_peak_figure_rank_features
    if 'horse_peak_12m' in df.columns and 'peak_12m_rank_in_race' in df.columns:
        race_key = ['date_dt', 'course_clean', 'off']
        race_best_fig = df.groupby(race_key)['horse_peak_12m'].transform('max')
        df['gap_to_race_best'] = (race_best_fig - df['horse_peak_12m']).clip(lower=0)
        below_race_best = df['gap_to_race_best'] > race_best_max_gap
        df['s4_improver_signal'] = (df['is_improving'] == 1) & (~below_race_best)
        df['s4_improver_signal'] = df['s4_improver_signal'].astype(int)
    else:
        df['s4_improver_signal'] = df['is_improving']

    df.drop(columns=['_fig_3_races_ago', '_fig_lto'], inplace=True, errors='ignore')
    return df
```

**Features added**: `recent_fig_gain`, `is_improving`, `gap_to_race_best`, `s4_improver_signal`

---

## Integration Priority

Implement in this order (highest expected lift first):

| Priority | Feature Set | Expected Signal Lift | Effort |
|----------|------------|---------------------|--------|
| 1 | Gold pills / multi-condition fit score (§4) | High — directly encodes Alfred Boucher pattern | Medium |
| 2 | Horse surface peak split (§1) | High — S3 return-to-surface is missed entirely | Medium |
| 3 | Peak-12m rank vs field (§2) | High — S1 leader signal | Low |
| 4 | LTO vs career peak gap (§3) | Medium — S2 quantifies "near ceiling" | Low |
| 5 | Trainer RTF (§6) | Medium — more stable than 14d win% | Low |
| 6 | Second-run-after-break (§5) | Medium — known trainer pattern | Low |
| 7 | Improver guard (§7) | Low-Medium — prevents false positives | Low |

---

## What These Features Are NOT Replacing

The RacingEdgeUK framework is a **manual selection framework**. In the ML model, these features feed the XGBoost classifier as numeric inputs; the model learns the non-linear interactions between them. The following caveats from the original posts still apply:

- **Unknown fitness**: the model cannot know if Alfred Boucher was physically ready
- **Ride quality**: jockey partnership with an unfamiliar horse
- **Trainer intent**: prep vs target — partially addressed by second-run-after-break flag but never fully knowable from public data
- **Pace data at scale**: the posts note this is their own weakest point; the project's `add_pace_features.py` partially addresses this

---

## Leakage Verification

After implementing any of the above, run:

```bash
python scripts/verify_no_leakage.py
```

Key checks to add to the verifier for these new features:
- `horse_turf_peak`, `horse_aw_peak`, `horse_overall_peak` — must be NaN on row 0 per horse
- `horse_peak_12m` — must not exceed the `or_numeric` value for the same race row
- `gold_pill_count` — rank columns must be computed within the same race day group only
- `s4_improver_signal` — `recent_fig_gain` must use `shift(1)` and `shift(3)`, never the current race's figure
