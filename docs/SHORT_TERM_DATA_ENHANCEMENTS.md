# Short-Term Data Enhancements (1-4 Weeks)

Quick wins that can be implemented with your existing data infrastructure and The Racing API.

---

## 1. Draw Position Advantage

**Impact**: High for sprint races; lower for longer distances  
**Difficulty**: Easy  
**Current Status**: Not in model

### Why It Matters
- Sprint races (5-7f): High draw often advantaged on many UK courses
- Draw bias varies by course, going conditions, and track configuration
- Can be a 5-10% edge on certain tracks

### Data Available
The Racing API returns `draw` field in racecard runners.

### Code Snippet: Extract & Engineer Draw Features

```python
# Add to scripts/phase3_build_horse_model.py

def engineer_draw_features(df):
    """
    Engineer draw-related features.
    Draw advantage is course and distance dependent.
    """
    print("\nEngineering draw features...")
    
    df = df.copy()
    
    # Extract draw position
    df['draw'] = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
    
    # Draw position relative to field size (0=rail, 1=widest)
    df['draw_pct'] = df['draw'] / df['field_size'].clip(lower=1)
    
    # Historical draw bias by course+distance+going
    # Calculate BEFORE current race (expanding window)
    df['date_dt'] = pd.to_datetime(df['date'])
    df = df.sort_values(['course_clean', 'dist_f', 'date_dt'])
    
    # Group draw positions into bins (low/middle/high)
    df['draw_group'] = pd.cut(
        df['draw_pct'], 
        bins=[0, 0.33, 0.66, 1.0], 
        labels=['low', 'middle', 'high'],
        include_lowest=True
    )
    
    # Calculate historical win rate by draw group per course/distance
    draw_stats = df.groupby(['course_clean', 'distance_band', 'draw_group']).apply(
        lambda x: (x['won'].shift(1).expanding().mean())
    ).reset_index(name='draw_group_win_rate')
    
    df = df.merge(draw_stats, on=['course_clean', 'distance_band', 'draw_group'], how='left')
    df['draw_group_win_rate'] = df['draw_group_win_rate'].fillna(0.1)
    
    print(f"  Draw features: draw, draw_pct, draw_group_win_rate")
    
    return df
```

### Add to predict_todays_races.py

```python
def build_horse_features_from_racecard(runner, race_info, historical_df):
    # ... existing code ...
    
    # Add draw features
    draw = runner.get('draw') or runner.get('stall')
    field_size = race_info.get('field_size', 10)
    
    features['draw'] = int(draw) if draw and str(draw).isdigit() else 0
    features['draw_pct'] = features['draw'] / max(field_size, 1)
    
    # Get historical draw bias for this course/distance
    course = race_info['course']
    dist_f = extract_distance_furlongs(race_info.get('distance_f', 8.0))
    
    draw_history = historical_df[
        (historical_df['course'].str.lower() == course.lower()) &
        (historical_df['dist_f'].between(dist_f - 1, dist_f + 1))
    ]
    
    if not draw_history.empty and features['draw'] > 0:
        # Calculate win rate for this draw range
        draw_min = max(1, features['draw'] - 2)
        draw_max = features['draw'] + 2
        similar_draws = draw_history[
            draw_history['draw'].between(draw_min, draw_max)
        ]
        features['draw_group_win_rate'] = (
            (similar_draws['pos'] == 1).sum() / len(similar_draws) 
            if len(similar_draws) > 10 else 0.1
        )
    else:
        features['draw_group_win_rate'] = 0.1
    
    return features
```

---

## 2. Weight Carried (Handicaps)

**Impact**: High for handicap races  
**Difficulty**: Easy  
**Current Status**: Not in model

### Why It Matters
- Weight is the great equalizer in handicaps
- 1lb = ~1 length over 1 mile (rule of thumb)
- Horses at top/bottom of weights behave differently

### Code Snippet: Weight Features

```python
def engineer_weight_features(df):
    """
    Engineer weight-related features for handicaps.
    """
    print("\nEngineering weight features...")
    
    df = df.copy()
    
    # Parse weight (lbs or st-lb format)
    def parse_weight_lbs(weight_str):
        if pd.isna(weight_str):
            return np.nan
        weight_str = str(weight_str)
        
        # Format: "9-7" (9 stone 7 lbs) or just "133" (lbs)
        if '-' in weight_str:
            parts = weight_str.split('-')
            stones = int(parts[0])
            lbs = int(parts[1]) if len(parts) > 1 else 0
            return stones * 14 + lbs
        else:
            try:
                return float(weight_str)
            except:
                return np.nan
    
    df['weight_lbs'] = df['wgt'].apply(parse_weight_lbs)
    
    # Weight relative to race (top weight = high, bottom = low)
    df['weight_rank'] = df.groupby(['date', 'course', 'off'])['weight_lbs'].rank(ascending=False)
    df['weight_vs_avg'] = df.groupby(['date', 'course', 'off'])['weight_lbs'].transform(
        lambda x: x - x.mean()
    )
    
    # Is this horse carrying top weight?
    df['is_top_weight'] = (df['weight_rank'] == 1).astype(int)
    
    # Weight change from last race (if handicap)
    df['prev_weight'] = df.groupby('horse')['weight_lbs'].shift(1)
    df['weight_change'] = df['weight_lbs'] - df['prev_weight']
    df['weight_change'] = df['weight_change'].fillna(0)
    
    print(f"  Weight features: weight_lbs, weight_vs_avg, is_top_weight, weight_change")
    
    return df
```

### Add to predictions feature builder

```python
# In build_horse_features_from_racecard()

# Weight features
weight_str = runner.get('wgt') or runner.get('weight') or runner.get('lbs')
features['weight_lbs'] = parse_weight_lbs(weight_str) if weight_str else 140  # Default

# Calculate weight vs race average (need all runners)
if 'all_weights' in race_info:
    avg_weight = sum(race_info['all_weights']) / len(race_info['all_weights'])
    features['weight_vs_avg'] = features['weight_lbs'] - avg_weight
    features['is_top_weight'] = 1 if features['weight_lbs'] == max(race_info['all_weights']) else 0
else:
    features['weight_vs_avg'] = 0
    features['is_top_weight'] = 0
```

---

## 3. Age Feature Enhancement

**Impact**: Medium  
**Difficulty**: Easy  
**Current Status**: Not in model

### Why It Matters
- 2-year-olds improve rapidly through the season
- Peak performance typically ages 4-5 for flat, 7-10 for jumps
- "Improving 3-year-old" is a common profitable angle

### Code Snippet

```python
def engineer_age_features(df):
    """
    Engineer age-related features.
    """
    print("\nEngineering age features...")
    
    df = df.copy()
    
    # Parse age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Age at peak performance (differs by race type)
    # Flat horses peak around 4, jump horses later
    df['is_peak_age'] = df['age'].between(4, 5).astype(int)
    
    # Is improving 3yo?
    df['is_3yo'] = (df['age'] == 3).astype(int)
    
    # Is veteran (older, declining)?
    df['is_veteran'] = (df['age'] >= 8).astype(int)
    
    # Age relative to race average
    df['age_vs_avg'] = df.groupby(['date', 'course', 'off'])['age'].transform(
        lambda x: x - x.mean()
    )
    
    print(f"  Age features: age, is_peak_age, is_3yo, is_veteran, age_vs_avg")
    
    return df
```

---

## 4. Trainer Form (Hot/Cold)

**Impact**: Medium-High  
**Difficulty**: Easy  
**Current Status**: Jockey-trainer combo only

### Why It Matters
- Trainers go through hot/cold streaks
- A trainer sending horses fit = winners cluster
- 14-day/30-day form is predictive

### Code Snippet

```python
def engineer_trainer_form(df):
    """
    Calculate trainer recent form (hot/cold streak).
    Uses rolling window to avoid lookahead.
    """
    print("\nEngineering trainer form...")
    
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # Sort chronologically
    df = df.sort_values('date_dt').reset_index(drop=True)
    
    # Track trainer stats over rolling 14-day and 30-day windows
    trainer_stats = {}
    
    features_14d = []
    features_30d = []
    
    for idx, row in df.iterrows():
        trainer = row.get('trainer', 'Unknown')
        race_date = row['date_dt']
        won = 1 if row.get('pos_clean') == 1 else 0
        
        # Get stats from last 14/30 days
        if trainer in trainer_stats:
            history = trainer_stats[trainer]
            
            # Filter to last 14 days
            recent_14d = [h for h in history if (race_date - h['date']).days <= 14]
            if recent_14d:
                runs_14d = len(recent_14d)
                wins_14d = sum(h['won'] for h in recent_14d)
                features_14d.append(wins_14d / runs_14d if runs_14d > 0 else 0.0)
            else:
                features_14d.append(0.0)
            
            # Filter to last 30 days
            recent_30d = [h for h in history if (race_date - h['date']).days <= 30]
            if recent_30d:
                runs_30d = len(recent_30d)
                wins_30d = sum(h['won'] for h in recent_30d)
                features_30d.append(wins_30d / runs_30d if runs_30d > 0 else 0.0)
            else:
                features_30d.append(0.0)
        else:
            features_14d.append(0.0)
            features_30d.append(0.0)
        
        # Update trainer history AFTER recording
        if trainer not in trainer_stats:
            trainer_stats[trainer] = []
        trainer_stats[trainer].append({'date': race_date, 'won': won})
        
        # Keep only last 60 days of history (memory optimization)
        trainer_stats[trainer] = [
            h for h in trainer_stats[trainer] 
            if (race_date - h['date']).days <= 60
        ]
    
    df['trainer_win_rate_14d'] = features_14d
    df['trainer_win_rate_30d'] = features_30d
    
    print(f"  Trainer form: trainer_win_rate_14d, trainer_win_rate_30d")
    
    return df
```

---

## 5. Beaten Lengths (Last Race Quality)

**Impact**: Medium  
**Difficulty**: Medium  
**Current Status**: Only position used

### Why It Matters
- 2nd beaten 0.5 lengths ≠ 2nd beaten 15 lengths
- Close finishes indicate better form than distant losses
- Useful for "unlucky loser" angle

### API Data Available
The Racing API returns `btn` (beaten lengths) in results.

### Code Snippet

```python
def engineer_beaten_lengths_features(df):
    """
    Calculate beaten lengths features for form analysis.
    """
    print("\nEngineering beaten lengths features...")
    
    df = df.copy()
    
    # Parse beaten lengths (can be decimal or fraction)
    def parse_btn(btn_str):
        if pd.isna(btn_str) or btn_str in ['', '-', 'W', 'won']:
            return 0.0  # Winner
        try:
            btn_str = str(btn_str).strip()
            if 'nk' in btn_str.lower():
                return 0.25
            if 'hd' in btn_str.lower() or 'head' in btn_str.lower():
                return 0.1
            if 'shd' in btn_str.lower() or 'short head' in btn_str.lower():
                return 0.05
            if 'nse' in btn_str.lower() or 'nose' in btn_str.lower():
                return 0.01
            if 'dist' in btn_str.lower():
                return 30.0
            return float(btn_str)
        except:
            return np.nan
    
    df['btn_lengths'] = df['btn'].apply(parse_btn)
    
    # Calculate features from recent races
    df = df.sort_values(['horse', 'date']).reset_index(drop=True)
    
    # Average beaten lengths last 3 races (lower = closer finishes)
    df['avg_btn_last_3'] = df.groupby('horse')['btn_lengths'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Was beaten narrowly last race? (unlucky loser)
    df['prev_btn'] = df.groupby('horse')['btn_lengths'].shift(1)
    df['unlucky_last'] = (
        (df['prev_btn'] <= 1.0) & (df['prev_btn'] > 0)
    ).astype(int)
    
    print(f"  BTN features: btn_lengths, avg_btn_last_3, unlucky_last")
    
    return df
```

---

## 6. Equipment Changes (First-Time Gear)

**Impact**: Medium (can be high for specific gear)  
**Difficulty**: Easy  
**Current Status**: Not in model

### Why It Matters
- First-time blinkers = often significant improvement
- Visor on/off = trainer trying something new
- Tongue tie, cheekpieces can all have impact

### API Data Available
The Racing API returns `gear` and `headgear` fields.

### Code Snippet

```python
def engineer_gear_features(df):
    """
    Engineer equipment/headgear features.
    """
    print("\nEngineering gear features...")
    
    df = df.copy()
    
    # Common headgear codes
    # b = blinkers, v = visor, h = hood, t = tongue strap
    # p = cheekpieces, e = eye shield
    
    df['headgear'] = df['headgear'].fillna('').str.lower()
    
    # Binary flags for common gear
    df['has_blinkers'] = df['headgear'].str.contains('b').astype(int)
    df['has_visor'] = df['headgear'].str.contains('v').astype(int)
    df['has_cheekpieces'] = df['headgear'].str.contains('p').astype(int)
    df['has_tongue_tie'] = df['headgear'].str.contains('t').astype(int)
    
    # First time with this gear (important!)
    df = df.sort_values(['horse', 'date']).reset_index(drop=True)
    
    df['prev_headgear'] = df.groupby('horse')['headgear'].shift(1).fillna('')
    
    # First-time blinkers (historically good angle)
    df['first_time_blinkers'] = (
        (df['has_blinkers'] == 1) & 
        (~df['prev_headgear'].str.contains('b', na=False))
    ).astype(int)
    
    # First-time visor
    df['first_time_visor'] = (
        (df['has_visor'] == 1) & 
        (~df['prev_headgear'].str.contains('v', na=False))
    ).astype(int)
    
    # Any gear change
    df['gear_changed'] = (df['headgear'] != df['prev_headgear']).astype(int)
    
    print(f"  Gear features: has_blinkers, has_visor, first_time_blinkers, gear_changed")
    
    return df
```

### For predictions

```python
# In build_horse_features_from_racecard()

headgear = runner.get('headgear', '') or ''
features['has_blinkers'] = 1 if 'b' in headgear.lower() else 0
features['has_visor'] = 1 if 'v' in headgear.lower() else 0

# First-time gear requires horse history lookup
if not horse_history.empty and len(horse_history) > 0:
    prev_gear = horse_history.head(1)['headgear'].values[0] if 'headgear' in horse_history.columns else ''
    prev_gear = prev_gear or ''
    features['first_time_blinkers'] = 1 if ('b' in headgear.lower() and 'b' not in prev_gear.lower()) else 0
else:
    features['first_time_blinkers'] = 1 if 'b' in headgear.lower() else 0  # Treat as first if no history
```

---

## 7. Race Conditions Refinement

**Impact**: Medium  
**Difficulty**: Easy  
**Current Status**: Basic (is_turf, going_numeric)

### Enhancements

```python
def engineer_race_condition_features(df):
    """
    Enhanced race condition features.
    """
    print("\nEngineering enhanced race condition features...")
    
    df = df.copy()
    
    # Is this a handicap race?
    df['is_handicap'] = df['type'].str.contains('Handicap|Hcap', case=False, na=False).astype(int)
    
    # Is this a maiden race? (first-time winners only)
    df['is_maiden'] = df['type'].str.contains('Maiden', case=False, na=False).astype(int)
    
    # Is this a stakes/pattern race?
    df['is_pattern'] = df['pattern'].notna().astype(int) if 'pattern' in df.columns else 0
    
    # Prize money tier (normalized)
    df['prize_numeric'] = pd.to_numeric(
        df['prize'].str.replace('[£,]', '', regex=True), 
        errors='coerce'
    ).fillna(0)
    df['prize_log'] = np.log1p(df['prize_numeric'])
    
    # Distance bands (more granular)
    df['dist_f'] = pd.to_numeric(df['dist_f'], errors='coerce')
    df['is_sprint'] = (df['dist_f'] <= 7).astype(int)
    df['is_mile'] = df['dist_f'].between(7.5, 9).astype(int)
    df['is_middle'] = df['dist_f'].between(9, 12).astype(int)
    df['is_staying'] = (df['dist_f'] > 12).astype(int)
    
    # Going preference (horse's historical going performance)
    # ... implement going suitability scoring
    
    print(f"  Race conditions: is_handicap, is_maiden, is_pattern, prize_log, is_sprint")
    
    return df
```

---

## Quick Implementation Checklist

- [ ] Add draw features to `phase3_build_horse_model.py`
- [ ] Add weight features for handicaps
- [ ] Add age features
- [ ] Add trainer 14d/30d form
- [ ] Add beaten lengths from last race
- [ ] Add first-time blinkers flag
- [ ] Add is_handicap/is_maiden flags
- [ ] Update `feature_columns.txt` with new features
- [ ] Retrain model with new features
- [ ] Validate improvement on holdout set

---

## Expected Impact

| Feature | Est. AUC Improvement | Implementation Time |
|---------|---------------------|---------------------|
| Draw position | +0.005-0.015 | 1 hour |
| Weight carried | +0.005-0.010 | 1 hour |
| Age features | +0.003-0.008 | 30 mins |
| Trainer form | +0.005-0.012 | 2 hours |
| Beaten lengths | +0.008-0.015 | 1 hour |
| First-time gear | +0.003-0.008 | 1 hour |
| Race conditions | +0.005-0.010 | 1 hour |

**Total estimated improvement**: +0.03-0.08 AUC (significant for betting edge)

---

## Testing New Features

```python
# scripts/test_new_features.py

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def test_feature_value(df, base_features, new_feature):
    """
    Test if adding a new feature improves model performance.
    """
    # Prepare data
    X_base = df[base_features].fillna(0)
    X_new = df[base_features + [new_feature]].fillna(0)
    y = df['won']
    
    # Split
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )
    X_new_train, X_new_test, _, _ = train_test_split(
        X_new, y, test_size=0.2, random_state=42
    )
    
    # Train baseline
    model_base = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_base.fit(X_base_train, y_train)
    auc_base = roc_auc_score(y_test, model_base.predict_proba(X_base_test)[:, 1])
    
    # Train with new feature
    model_new = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_new.fit(X_new_train, y_train)
    auc_new = roc_auc_score(y_test, model_new.predict_proba(X_new_test)[:, 1])
    
    improvement = auc_new - auc_base
    
    print(f"Feature: {new_feature}")
    print(f"  Baseline AUC: {auc_base:.4f}")
    print(f"  With feature: {auc_new:.4f}")
    print(f"  Improvement:  {improvement:+.4f} {'✓' if improvement > 0 else '✗'}")
    
    return improvement

# Run tests for each new feature
if __name__ == '__main__':
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    base_features = [
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
        'cd_runs', 'cd_win_rate', 'class_num', 'class_step',
        'or_numeric', 'or_change', 'or_trend_3', 'avg_last_3_pos', 'wins_last_3',
        'days_since_last', 'field_size', 'is_turf', 'going_numeric', 'race_score'
    ]
    
    # Test each new feature
    new_features = ['draw', 'draw_pct', 'weight_lbs', 'age', 'trainer_win_rate_14d']
    
    for feat in new_features:
        if feat in df.columns:
            test_feature_value(df, base_features, feat)
```
