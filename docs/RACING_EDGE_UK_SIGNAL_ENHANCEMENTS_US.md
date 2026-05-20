# RacingEdgeUK Signal Framework — Enhancements for the US Model

**Source**: Analysis of [RacingEdgeUK methodology post](https://old.reddit.com/r/RacingEdgeUK/comments/1teprnx/) and the [Alfred Boucher case study](https://old.reddit.com/r/RacingEdgeUK/comments/1tephr9/) (May 2026).

**See also**: [RACING_EDGE_UK_SIGNAL_ENHANCEMENTS.md](RACING_EDGE_UK_SIGNAL_ENHANCEMENTS.md) for the European model version of the same analysis. This document focuses on US-specific adaptations and the differences caused by US racing structure.

---

## How the Framework Maps to US Racing

The RacingEdgeUK framework was built for UK/Irish conditions racing. US racing has structural differences that make some signals stronger and others require adaptation:

| Framework Concept | UK/Irish | US Equivalent | Notes |
|-------------------|----------|---------------|-------|
| Peak figure metric | RPR / Official Rating | Beyer Speed Figure | Beyer is the standard US equivalent |
| Surface split | Turf vs AW (all-weather) | Dirt / Turf / Synthetic (3-way) | Surface switches are far more common and dramatic in the US |
| Going categories | Firm/Good/Soft/Heavy | Fast/Good/Yielding/Firm (turf) + Fast/Sloppy/Muddy (dirt) | Dirt going and turf going are independent dimensions |
| Class | UK handicap ratings (0-115+) | Grade I–III / Allowance / Claiming / Maiden levels | `us_class_num` (1.0–7.0) already exists |
| Course form | Every UK course is highly specific (Epsom, Ascot, etc.) | US tracks are broadly standardised but oval shape/circumference varies | Track-specific Beyer variant accounts for some of this |
| Weight | Major factor in UK handicaps | Minor factor; weight-for-age scale; claiming weights vary | Weight features matter less for US claiming/allowance |
| Draw | Critical in UK (rail bias, watering patterns) | Less dominant; post position matters more in sprint routes | Draw bias exists but is less universal |
| Trainer patterns | UK yards are highly specific | US barns have tighter "condition book" patterns | Condition-book targeting is a stronger trainer signal in US |

---

## Gap Analysis: US-Specific Missing Features

On top of the seven gaps identified in the European model document, the US model has three additional gaps and two amplified gaps:

| # | Signal | What's Missing | US Context |
|---|--------|----------------|------------|
| 1 | S3 | **Three-way surface split Beyer peaks** (dirt/turf/synthetic separately) | Stronger than UK 2-way split; horses often have zero runs on one surface |
| 2 | S3 | **Synthetic → dirt transition** signal | Many US horses race on synthetic tracks then move to dirt; transition figures unreliable |
| 3 | S1 | **Beyer-based peak-12m rank in race** | `beyer_best` exists but is not race-ranked against the field |
| 4 | S4 | **Workout-adjusted improver signal** — accelerating workout pattern ahead of the race | Workouts are public in the US; fast workouts in the week before a big target are a known signal |
| 5 | B | **Condition-book pattern** — trainer returning to a specific condition type they target | US trainers often angle horses toward specific condition book slots |

---

## Proposed US-Specific Implementations

### 1. Three-Way Surface Split Beyer Peaks (S3)

The UK model uses a binary turf/AW split. US racing has three surfaces — dirt, turf, and synthetic (Polytrack/Tapeta) — and the suppression effect is much stronger: a turf specialist run on dirt will post dramatically lower Beyer figures, causing the market to underestimate them when they return to grass.

Add to `scripts/train_us_model.py` or a new `scripts/add_us_surface_features.py`:

```python
import pandas as pd
import numpy as np

SURFACES = ['dirt', 'turf', 'synthetic']

def add_us_surface_peak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute horse's Beyer Speed Figure peak split across three US surfaces:
    dirt, turf, and synthetic.

    S3 signal (Return to Surface): if a horse's peak on today's surface is close
    to its overall Beyer ceiling, but recent runs were on a different surface,
    the headline figure is suppressed and the horse is underrated.

    Temporal integrity: shift(1) before expanding().max() on all aggregations.
    Sort df by [horse, date_dt] before calling.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'beyer_last' if 'beyer_last' in df.columns else 'or_numeric'

    # Normalise surface to one of the three categories
    if 'surface' in df.columns:
        df['surface_cat'] = df['surface'].str.lower().map(
            lambda x: 'synthetic' if any(s in str(x) for s in ['poly', 'tapeta', 'synth', 'all'])
                      else ('turf' if 'turf' in str(x) else 'dirt')
        )
    else:
        df['surface_cat'] = 'dirt'  # fallback

    # Career peak Beyer on each surface (prior races only)
    for surf in SURFACES:
        mask_col = f'_fig_if_{surf}'
        peak_col = f'horse_{surf}_beyer_peak'
        df[mask_col] = df[fig_col].where(df['surface_cat'] == surf)
        df[peak_col] = (
            df.groupby('horse')[mask_col]
              .transform(lambda x: x.shift(1).expanding().max())
        )
        df.drop(columns=[mask_col], inplace=True, errors='ignore')

    # Overall career peak Beyer (prior races only)
    df['horse_overall_beyer_peak'] = (
        df.groupby('horse')[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Today's surface peak (whichever surface applies today)
    def get_today_surface_peak(row):
        return row.get(f"horse_{row.get('surface_cat', 'dirt')}_beyer_peak", np.nan)

    df['horse_today_surface_beyer_peak'] = df.apply(get_today_surface_peak, axis=1)

    # Surface peak ratio: how close is today's surface peak to overall ceiling
    df['us_surface_peak_ratio'] = (
        df['horse_today_surface_beyer_peak'] /
        df['horse_overall_beyer_peak'].replace(0, np.nan)
    ).fillna(0.5)

    # Surface switch flag
    df['_prev_surface'] = df.groupby('horse')['surface_cat'].shift(1)
    df['us_surface_switch'] = (df['surface_cat'] != df['_prev_surface']).astype(int)

    # Surface return advantage: returning to a surface where peak > 90% of overall ceiling
    df['us_surface_return_advantage'] = (
        (df['us_surface_switch'] == 1) &
        (df['us_surface_peak_ratio'] >= 0.90)
    ).astype(int)

    # Synthetic → dirt transition flag (notoriously unreliable in US racing)
    df['us_synthetic_to_dirt'] = (
        (df['_prev_surface'] == 'synthetic') & (df['surface_cat'] == 'dirt')
    ).astype(int)

    # Surface suppression flag: LTO was on wrong surface, today's surface peak is 8+ Beyer points higher
    df['_lto_fig'] = df.groupby('horse')[fig_col].shift(1)
    df['us_surface_suppression_flag'] = (
        (df['us_surface_switch'] == 1) &
        (df['horse_today_surface_beyer_peak'] - df['_lto_fig'] >= 8)
    ).astype(int)

    df.drop(columns=['_prev_surface', '_lto_fig'], inplace=True, errors='ignore')
    return df
```

**Features added**: `surface_cat`, `horse_dirt_beyer_peak`, `horse_turf_beyer_peak`, `horse_synthetic_beyer_peak`, `horse_overall_beyer_peak`, `horse_today_surface_beyer_peak`, `us_surface_peak_ratio`, `us_surface_switch`, `us_surface_return_advantage`, `us_synthetic_to_dirt`, `us_surface_suppression_flag`

---

### 2. Beyer-Based Peak-12m Race Rank (S1)

The US model has `beyer_best` (career best Beyer) but no race-relative ranking. The S1 signal is triggered when a horse owns the highest peak figure in the race over the last 12 months at today's track.

Add to `scripts/train_us_model.py` post-feature engineering:

```python
def add_us_peak_figure_race_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each horse's Beyer peak (12-month rolling window) against the rest of today's field.

    S1 fires when:
      - us_beyer_peak_12m_rank == 1  (best Beyer in field over last 12m)
      - us_beyer_peak_12m_lead >= threshold (default 3 Beyer points)

    Temporal integrity: 12m peak uses shift(1) inside rolling. Race rank is
    computed within the same race group (date/track/race_number) — no cross-race leakage.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'beyer_last' if 'beyer_last' in df.columns else 'or_numeric'
    PEAK_LEAD_THRESHOLD = 3  # minimum Beyer points gap to fire S1

    # 12-month rolling peak Beyer (prior races only)
    df['us_beyer_peak_12m'] = (
        df.groupby('horse')[fig_col]
          .transform(lambda x: x.shift(1).rolling('365D', min_periods=1).max())
    )

    # Track-specific career peak (analogous to UK course peak)
    df['us_beyer_peak_at_track'] = (
        df.groupby(['horse', 'track'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Race-relative ranking
    race_key = ['date_dt', 'track', 'race_number']
    df['us_beyer_peak_12m_rank'] = (
        df.groupby(race_key)['us_beyer_peak_12m']
          .rank(ascending=False, method='min', na_option='bottom')
    )
    df['us_is_beyer_leader'] = (df['us_beyer_peak_12m_rank'] == 1).astype(int)

    # Gap over second-best (S1 minimum lead)
    race_second_best = (
        df.groupby(race_key)['us_beyer_peak_12m']
          .transform(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else x.max())
    )
    df['us_beyer_lead_over_2nd'] = (df['us_beyer_peak_12m'] - race_second_best).clip(lower=0)
    df['us_s1_beyer_leader_signal'] = (
        (df['us_is_beyer_leader'] == 1) &
        (df['us_beyer_lead_over_2nd'] >= PEAK_LEAD_THRESHOLD)
    ).astype(int)

    return df
```

**Features added**: `us_beyer_peak_12m`, `us_beyer_peak_at_track`, `us_beyer_peak_12m_rank`, `us_is_beyer_leader`, `us_beyer_lead_over_2nd`, `us_s1_beyer_leader_signal`

---

### 3. Workout-Adjusted Improver Signal (S4 Variant)

US racing publishes workout data that is unavailable in the UK. A horse posting a blistering workout in the 5–10 days before a target race is a standard "trainer is pointing" signal. Combined with the S4 upward figure trend, this is a meaningfully stronger composite signal.

Add to `scripts/train_us_model.py` or `scripts/add_us_surface_features.py`:

```python
def add_us_workout_improver_signal(df: pd.DataFrame, 
                                    workout_fast_threshold: float = 0.95,
                                    min_fig_gain: float = 3.0) -> pd.DataFrame:
    """
    S4 Improver signal augmented by workout recency and quality.

    In US racing, a horse showing:
      1. Upward Beyer trend over last 3 races (recent_beyer_gain >= min_fig_gain)
      2. A fast workout in the last 10 days (workout_fast_flag == 1)
    
    ...is a significantly stronger S4 signal than figure trend alone.

    workout_fast_threshold: percentile cutoff for a "fast" workout at the track/distance
                           (0.95 = top 5% of workouts at that track/furlong in the last 90d)
    min_fig_gain: minimum Beyer gain over the 3-race window to qualify as an improver.

    Temporal integrity: Beyer trend uses shift(1)/shift(3). Workout data is prior
    to race day by definition (workouts happen before races).
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'beyer_last' if 'beyer_last' in df.columns else 'or_numeric'

    # Beyer trend over last 3 races
    df['_beyer_3_ago'] = df.groupby('horse')[fig_col].shift(3)
    df['_beyer_lto']   = df.groupby('horse')[fig_col].shift(1)
    df['us_recent_beyer_gain'] = (df['_beyer_lto'] - df['_beyer_3_ago']).fillna(0)
    df['us_is_beyer_improver'] = (df['us_recent_beyer_gain'] >= min_fig_gain).astype(int)

    # Workout fast flag (requires workout_count_30d and days_since_last_workout from existing features)
    if 'days_since_last_workout' in df.columns:
        df['workout_recency_flag'] = (df['days_since_last_workout'] <= 10).astype(int)
    else:
        df['workout_recency_flag'] = 0

    # Combined S4 workout-improver signal
    df['us_s4_workout_improver_signal'] = (
        (df['us_is_beyer_improver'] == 1) &
        (df['workout_recency_flag'] == 1)
    ).astype(int)

    df.drop(columns=['_beyer_3_ago', '_beyer_lto'], inplace=True, errors='ignore')
    return df
```

**Features added**: `us_recent_beyer_gain`, `us_is_beyer_improver`, `workout_recency_flag`, `us_s4_workout_improver_signal`

---

### 4. US Multi-Condition Fit Score (S5 / Gold Pills — US Version)

The same Alfred Boucher logic applies to US racing, but the five conditions need adapting:

- **Distance**: furlongs band (sprint/route, same concept)
- **Going**: dirt fast/sloppy/muddy OR turf firm/good/yielding (surface-conditioned going)
- **Class**: Grade/Allowance/Claiming level (use `us_class_num`)
- **Track**: track-specific Beyer peak (replaces UK course form)
- **Weight**: less critical in US but still relevant for handicap races

Add to `scripts/train_us_model.py`:

```python
def add_us_condition_fit_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    US version of the multi-condition gold pill count.

    Computes career-best Beyer for today's conditions (distance/surface+going/class/track),
    ranks each horse against the rest of today's field on each dimension,
    and counts how many conditions the horse leads the field in.

    A horse with 3–4 gold pills in the US context is equivalent to Alfred Boucher's
    multi-condition profile.

    Temporal integrity: all career bests use shift(1). Race ranks are within-race-group.
    """
    df = df.sort_values(['horse', 'date_dt']).copy()
    fig_col = 'beyer_last' if 'beyer_last' in df.columns else 'or_numeric'
    race_key = ['date_dt', 'track', 'race_number']

    # Career-best Beyer at today's distance band (prior races only)
    df['us_peak_at_distance'] = (
        df.groupby(['horse', 'distance_band'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Career-best Beyer on today's surface+going key
    # e.g. "dirt_fast", "turf_firm" — combines surface and going condition
    if 'surface_cat' not in df.columns:
        df['surface_cat'] = 'dirt'
    if 'going_category' not in df.columns:
        df['going_category'] = 'fast'
    df['surface_going_key'] = df['surface_cat'] + '_' + df['going_category'].astype(str)
    df['us_peak_at_surface_going'] = (
        df.groupby(['horse', 'surface_going_key'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Career-best Beyer at today's class level (prior races only)
    df['us_peak_at_class'] = (
        df.groupby(['horse', 'us_class_num'] if 'us_class_num' in df.columns
                   else ['horse', 'class_num'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Career-best Beyer at today's track
    df['us_peak_at_track'] = (
        df.groupby(['horse', 'track'])[fig_col]
          .transform(lambda x: x.shift(1).expanding().max())
    )

    # Race-relative rankings (gold pills)
    condition_cols = {
        'us_peak_at_distance':     'us_dist_pill_rank',
        'us_peak_at_surface_going': 'us_going_pill_rank',
        'us_peak_at_class':        'us_class_pill_rank',
        'us_peak_at_track':        'us_track_pill_rank',
    }
    for peak_col, rank_col in condition_cols.items():
        if peak_col in df.columns:
            df[rank_col] = (
                df.groupby(race_key)[peak_col]
                  .rank(ascending=False, method='min', na_option='bottom')
            )

    # Gold pill count
    rank_cols = list(condition_cols.values())
    available = [c for c in rank_cols if c in df.columns]
    df['us_gold_pill_count'] = (df[available] == 1).sum(axis=1)
    df['us_top2_pill_count']  = (df[available] <= 2).sum(axis=1)

    # S5 multi-condition leader: 3+ gold pills in US context
    df['us_s5_multi_condition_leader'] = (df['us_gold_pill_count'] >= 3).astype(int)

    return df
```

**Features added**: `us_peak_at_distance`, `us_peak_at_surface_going`, `us_peak_at_class`, `us_peak_at_track`, `us_dist_pill_rank`, `us_going_pill_rank`, `us_class_pill_rank`, `us_track_pill_rank`, `us_gold_pill_count`, `us_top2_pill_count`, `us_s5_multi_condition_leader`

---

## Integration into the US Training Pipeline

Add the new feature functions to `scripts/train_us_model.py` in the feature-build block (after existing features are assembled), and include the new columns in `US_EXTRA_FEATURES`:

```python
# In scripts/train_us_model.py — append to feature build block

from scripts.add_us_surface_features import (
    add_us_surface_peak_features,
    add_us_peak_figure_race_rank,
    add_us_workout_improver_signal,
    add_us_condition_fit_score,
)

# Apply in order (dependencies: surface features first)
df = add_us_surface_peak_features(df)
df = add_us_peak_figure_race_rank(df)
df = add_us_workout_improver_signal(df)
df = add_us_condition_fit_score(df)

# Add new features to US_EXTRA_FEATURES (only if present, model degrades gracefully)
NEW_US_FEATURES = [
    # Surface peaks
    "horse_dirt_beyer_peak", "horse_turf_beyer_peak", "horse_synthetic_beyer_peak",
    "horse_overall_beyer_peak", "us_surface_peak_ratio",
    "us_surface_switch", "us_surface_return_advantage",
    "us_synthetic_to_dirt", "us_surface_suppression_flag",
    # Peak figure race rank
    "us_beyer_peak_12m", "us_beyer_peak_at_track",
    "us_is_beyer_leader", "us_beyer_lead_over_2nd", "us_s1_beyer_leader_signal",
    # Workout improver
    "us_recent_beyer_gain", "us_is_beyer_improver",
    "workout_recency_flag", "us_s4_workout_improver_signal",
    # Multi-condition fit
    "us_peak_at_distance", "us_peak_at_surface_going",
    "us_peak_at_class", "us_peak_at_track",
    "us_gold_pill_count", "us_top2_pill_count", "us_s5_multi_condition_leader",
]

US_EXTRA_FEATURES = US_EXTRA_FEATURES + [f for f in NEW_US_FEATURES if f in df.columns]
```

---

## US-Specific Leakage Considerations

Surface switching in US racing creates a subtle leakage risk:

- A horse switching from dirt to turf on a given race day: `surface_cat` for the **current** race must be determined from the racecard (today's declared surface), not from historical results.
- The `us_synthetic_to_dirt` flag compares `_prev_surface` (from prior results, safe) to `surface_cat` (from today's racecard, also safe — it is race-level, not outcome-level).
- **Do not** compute `surface_cat` from any performance metric of today's race. It should be derived from the race entry data only.

After implementing, run:

```bash
python scripts/verify_no_leakage.py
```

Key additional checks:
- `horse_dirt_beyer_peak` / `horse_turf_beyer_peak` / `horse_synthetic_beyer_peak` — must be NaN for horses with zero prior runs on that surface
- `us_beyer_peak_12m` — must not exceed `beyer_last` value for the same row
- `us_gold_pill_count` — must be computed within the same race date/track/race_number group only

---

## Comparison: UK vs US Signal Priority

| Signal | UK Model Priority | US Model Priority | Reason for Difference |
|--------|------------------|------------------|----------------------|
| Surface peak split (S3) | High | **Very High** | Three surfaces in US, dramatic performance differences |
| Multi-condition fit (S5) | High | High | Same logic, slightly different conditions |
| Peak-12m race rank (S1) | High | High | Beyer replaces RPR, same concept |
| Workout improver (S4 variant) | N/A | Medium-High | Public workout data unique to US |
| Trainer RTF | Medium | Medium | US trainer patterns use condition book angle instead |
| Second run after break | Medium | Medium | Same concept, US trainers also use prep races |
| Synthetic → dirt flag | N/A | Medium | US-specific transition risk |
