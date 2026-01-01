# Medium-Term Data Enhancements (1-3 Months)

More sophisticated features requiring additional data collection, API integration, or model restructuring.

---

## 1. Pedigree & Breeding Analysis

**Impact**: High (especially for maidens and younger horses)  
**Difficulty**: Medium  
**Data Source**: The Racing API (sire, dam, damsire fields)

### Why It Matters
- Sire progeny perform consistently on certain surfaces/distances
- Dam's offspring history indicates potential
- Damsire (maternal grandfather) influences stamina
- Critical for 2yo/maiden races where horse has limited form

### Data Collection Script

```python
#!/usr/bin/env python3
"""scripts/build_pedigree_lookup.py
Build sire/dam performance lookup tables from historical data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def build_pedigree_stats(df, min_runners=20):
    """
    Build performance statistics for each sire, dam, and damsire.
    
    Args:
        df: Historical race data with pedigree columns
        min_runners: Minimum runners to include in lookup
    
    Returns:
        dict: {sire_id: stats, dam_id: stats, damsire_id: stats}
    """
    print("Building pedigree statistics...")
    
    # Ensure outcome columns exist
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    
    # === SIRE STATS ===
    sire_stats = df.groupby('sire_id').agg({
        'won': ['sum', 'count'],
        'placed': 'sum',
        'is_turf': 'mean',  # % of runs on turf
        'dist_f': 'mean',   # Average distance
        'class_num': 'mean' # Average class
    }).reset_index()
    
    sire_stats.columns = [
        'sire_id', 'sire_wins', 'sire_runs', 'sire_places',
        'sire_turf_pct', 'sire_avg_dist', 'sire_avg_class'
    ]
    
    sire_stats['sire_win_rate'] = sire_stats['sire_wins'] / sire_stats['sire_runs']
    sire_stats['sire_place_rate'] = sire_stats['sire_places'] / sire_stats['sire_runs']
    
    # Filter by minimum runners
    sire_stats = sire_stats[sire_stats['sire_runs'] >= min_runners]
    
    print(f"  Sires: {len(sire_stats):,} with {min_runners}+ runners")
    
    # === SIRE SURFACE SPECIALIZATION ===
    sire_turf = df[df['is_turf'] == 1].groupby('sire_id').agg({
        'won': ['sum', 'count']
    }).reset_index()
    sire_turf.columns = ['sire_id', 'sire_turf_wins', 'sire_turf_runs']
    sire_turf['sire_turf_win_rate'] = sire_turf['sire_turf_wins'] / sire_turf['sire_turf_runs']
    
    sire_aw = df[df['is_turf'] == 0].groupby('sire_id').agg({
        'won': ['sum', 'count']
    }).reset_index()
    sire_aw.columns = ['sire_id', 'sire_aw_wins', 'sire_aw_runs']
    sire_aw['sire_aw_win_rate'] = sire_aw['sire_aw_wins'] / sire_aw['sire_aw_runs']
    
    sire_stats = sire_stats.merge(sire_turf[['sire_id', 'sire_turf_win_rate']], on='sire_id', how='left')
    sire_stats = sire_stats.merge(sire_aw[['sire_id', 'sire_aw_win_rate']], on='sire_id', how='left')
    
    # === SIRE DISTANCE BANDS ===
    for dist_name, dist_range in [('sprint', (0, 7)), ('mile', (7, 9)), ('middle', (9, 12)), ('staying', (12, 99))]:
        dist_df = df[df['dist_f'].between(*dist_range)].groupby('sire_id').agg({
            'won': ['sum', 'count']
        }).reset_index()
        dist_df.columns = ['sire_id', f'sire_{dist_name}_wins', f'sire_{dist_name}_runs']
        dist_df[f'sire_{dist_name}_win_rate'] = (
            dist_df[f'sire_{dist_name}_wins'] / dist_df[f'sire_{dist_name}_runs']
        )
        sire_stats = sire_stats.merge(
            dist_df[['sire_id', f'sire_{dist_name}_win_rate']], 
            on='sire_id', how='left'
        )
    
    # === DAM STATS ===
    dam_stats = df.groupby('dam_id').agg({
        'won': ['sum', 'count'],
        'placed': 'sum'
    }).reset_index()
    dam_stats.columns = ['dam_id', 'dam_wins', 'dam_runs', 'dam_places']
    dam_stats['dam_win_rate'] = dam_stats['dam_wins'] / dam_stats['dam_runs']
    dam_stats['dam_place_rate'] = dam_stats['dam_places'] / dam_stats['dam_runs']
    dam_stats = dam_stats[dam_stats['dam_runs'] >= 5]  # Lower threshold for dams
    
    print(f"  Dams: {len(dam_stats):,} with 5+ runners")
    
    # === DAMSIRE STATS ===
    damsire_stats = df.groupby('damsire_id').agg({
        'won': ['sum', 'count'],
        'placed': 'sum'
    }).reset_index()
    damsire_stats.columns = ['damsire_id', 'damsire_wins', 'damsire_runs', 'damsire_places']
    damsire_stats['damsire_win_rate'] = damsire_stats['damsire_wins'] / damsire_stats['damsire_runs']
    damsire_stats['damsire_place_rate'] = damsire_stats['damsire_places'] / damsire_stats['damsire_runs']
    damsire_stats = damsire_stats[damsire_stats['damsire_runs'] >= min_runners]
    
    print(f"  Damsires: {len(damsire_stats):,} with {min_runners}+ runners")
    
    return {
        'sire': sire_stats,
        'dam': dam_stats,
        'damsire': damsire_stats
    }


def save_pedigree_lookups(pedigree_stats, output_dir='data/processed'):
    """Save pedigree lookup tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pedigree_stats['sire'].to_csv(output_dir / 'sire_stats.csv', index=False)
    pedigree_stats['dam'].to_csv(output_dir / 'dam_stats.csv', index=False)
    pedigree_stats['damsire'].to_csv(output_dir / 'damsire_stats.csv', index=False)
    
    print(f"\n✓ Saved pedigree lookups to {output_dir}")


if __name__ == '__main__':
    df = pd.read_parquet('data/processed/race_scores.parquet')
    stats = build_pedigree_stats(df)
    save_pedigree_lookups(stats)
```

### Feature Engineering for Pedigree

```python
def engineer_pedigree_features(df, pedigree_lookups):
    """
    Add pedigree-based features to race data.
    Uses lookup tables to avoid lookahead bias.
    """
    print("\nEngineering pedigree features...")
    
    sire_stats = pedigree_lookups['sire']
    dam_stats = pedigree_lookups['dam']
    damsire_stats = pedigree_lookups['damsire']
    
    # Merge sire stats
    df = df.merge(
        sire_stats[['sire_id', 'sire_win_rate', 'sire_place_rate', 
                    'sire_turf_win_rate', 'sire_aw_win_rate',
                    'sire_sprint_win_rate', 'sire_mile_win_rate',
                    'sire_middle_win_rate', 'sire_staying_win_rate']],
        on='sire_id', how='left'
    )
    
    # Merge dam stats
    df = df.merge(
        dam_stats[['dam_id', 'dam_win_rate', 'dam_place_rate']],
        on='dam_id', how='left'
    )
    
    # Merge damsire stats
    df = df.merge(
        damsire_stats[['damsire_id', 'damsire_win_rate', 'damsire_place_rate']],
        on='damsire_id', how='left'
    )
    
    # Fill missing with global averages
    df['sire_win_rate'] = df['sire_win_rate'].fillna(df['sire_win_rate'].mean())
    df['dam_win_rate'] = df['dam_win_rate'].fillna(df['dam_win_rate'].mean())
    df['damsire_win_rate'] = df['damsire_win_rate'].fillna(df['damsire_win_rate'].mean())
    
    # Calculate sire suitability for this race
    # Match sire's best distance to race distance
    def get_sire_distance_suitability(row):
        dist_f = row.get('dist_f', 8)
        if dist_f <= 7:
            return row.get('sire_sprint_win_rate', 0.1)
        elif dist_f <= 9:
            return row.get('sire_mile_win_rate', 0.1)
        elif dist_f <= 12:
            return row.get('sire_middle_win_rate', 0.1)
        else:
            return row.get('sire_staying_win_rate', 0.1)
    
    df['sire_dist_suitability'] = df.apply(get_sire_distance_suitability, axis=1)
    
    # Sire surface suitability
    df['sire_surface_win_rate'] = np.where(
        df['is_turf'] == 1,
        df['sire_turf_win_rate'],
        df['sire_aw_win_rate']
    ).astype(float)
    df['sire_surface_win_rate'] = df['sire_surface_win_rate'].fillna(0.1)
    
    print(f"  Pedigree features: sire_win_rate, dam_win_rate, damsire_win_rate")
    print(f"  Suitability features: sire_dist_suitability, sire_surface_win_rate")
    
    return df
```

### Prediction Integration

```python
# In predict_todays_races.py

def load_pedigree_lookups():
    """Load pre-computed pedigree lookup tables."""
    data_dir = Path('data/processed')
    
    sire_stats = pd.read_csv(data_dir / 'sire_stats.csv')
    dam_stats = pd.read_csv(data_dir / 'dam_stats.csv')
    damsire_stats = pd.read_csv(data_dir / 'damsire_stats.csv')
    
    return {
        'sire': sire_stats.set_index('sire_id').to_dict('index'),
        'dam': dam_stats.set_index('dam_id').to_dict('index'),
        'damsire': damsire_stats.set_index('damsire_id').to_dict('index')
    }


def add_pedigree_features_to_prediction(features, runner, pedigree_lookups, race_info):
    """Add pedigree features for a runner."""
    
    sire_id = runner.get('sire_id')
    dam_id = runner.get('dam_id')
    damsire_id = runner.get('damsire_id')
    
    # Sire features
    if sire_id and sire_id in pedigree_lookups['sire']:
        sire = pedigree_lookups['sire'][sire_id]
        features['sire_win_rate'] = sire.get('sire_win_rate', 0.1)
        features['sire_place_rate'] = sire.get('sire_place_rate', 0.25)
        
        # Distance suitability
        dist_f = race_info.get('distance_f', 8)
        if dist_f <= 7:
            features['sire_dist_suitability'] = sire.get('sire_sprint_win_rate', 0.1)
        elif dist_f <= 9:
            features['sire_dist_suitability'] = sire.get('sire_mile_win_rate', 0.1)
        elif dist_f <= 12:
            features['sire_dist_suitability'] = sire.get('sire_middle_win_rate', 0.1)
        else:
            features['sire_dist_suitability'] = sire.get('sire_staying_win_rate', 0.1)
        
        # Surface suitability
        is_turf = race_info.get('surface') == 'Turf'
        features['sire_surface_win_rate'] = (
            sire.get('sire_turf_win_rate', 0.1) if is_turf 
            else sire.get('sire_aw_win_rate', 0.1)
        )
    else:
        features['sire_win_rate'] = 0.1
        features['sire_place_rate'] = 0.25
        features['sire_dist_suitability'] = 0.1
        features['sire_surface_win_rate'] = 0.1
    
    # Dam features
    if dam_id and dam_id in pedigree_lookups['dam']:
        dam = pedigree_lookups['dam'][dam_id]
        features['dam_win_rate'] = dam.get('dam_win_rate', 0.1)
        features['dam_place_rate'] = dam.get('dam_place_rate', 0.25)
    else:
        features['dam_win_rate'] = 0.1
        features['dam_place_rate'] = 0.25
    
    # Damsire features
    if damsire_id and damsire_id in pedigree_lookups['damsire']:
        damsire = pedigree_lookups['damsire'][damsire_id]
        features['damsire_win_rate'] = damsire.get('damsire_win_rate', 0.1)
        features['damsire_place_rate'] = damsire.get('damsire_place_rate', 0.25)
    else:
        features['damsire_win_rate'] = 0.1
        features['damsire_place_rate'] = 0.25
    
    return features
```

---

## 2. Going Preference Profiling

**Impact**: High (especially for specialists)  
**Difficulty**: Medium  
**Data Source**: Historical performance by going condition

### Why It Matters
- Some horses only perform on firm ground
- Others are "mud larks" who excel in soft/heavy
- Going changes can make/break a horse's chance

### Code Implementation

```python
#!/usr/bin/env python3
"""scripts/build_going_profiles.py
Build going preference profiles for each horse.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Going condition numeric mapping (1=Firm to 7=Heavy)
GOING_MAP = {
    'Hard': 1, 'Firm': 1.5, 'Good To Firm': 2, 'Good': 3,
    'Good To Soft': 4, 'Soft': 5, 'Soft To Heavy': 6, 'Heavy': 7,
    'Standard': 3, 'Standard To Slow': 4.5, 'Slow': 5.5
}

# Going categories
GOING_CATEGORIES = {
    'fast': (0, 2.5),      # Firm/Good to Firm
    'good': (2.5, 4),      # Good
    'soft': (4, 6),        # Good to Soft / Soft
    'heavy': (6, 8)        # Heavy
}


def build_going_profiles(df, min_runs=3):
    """
    Build going preference profile for each horse.
    
    Returns win rate and run count for each going category.
    """
    print("Building going preference profiles...")
    
    df = df.copy()
    df['going_numeric'] = df['going'].map(GOING_MAP).fillna(3)
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    
    # Categorize going
    def categorize_going(going_num):
        for cat, (low, high) in GOING_CATEGORIES.items():
            if low < going_num <= high:
                return cat
        return 'good'  # Default
    
    df['going_category'] = df['going_numeric'].apply(categorize_going)
    
    # Build profiles per horse
    profiles = []
    
    for horse in df['horse'].unique():
        horse_data = df[df['horse'] == horse]
        
        profile = {'horse': horse}
        
        for cat in ['fast', 'good', 'soft', 'heavy']:
            cat_data = horse_data[horse_data['going_category'] == cat]
            runs = len(cat_data)
            wins = cat_data['won'].sum()
            places = cat_data['placed'].sum()
            
            profile[f'{cat}_runs'] = runs
            profile[f'{cat}_win_rate'] = wins / runs if runs >= min_runs else np.nan
            profile[f'{cat}_place_rate'] = places / runs if runs >= min_runs else np.nan
        
        # Determine best going
        going_rates = {
            'fast': profile.get('fast_win_rate'),
            'good': profile.get('good_win_rate'),
            'soft': profile.get('soft_win_rate'),
            'heavy': profile.get('heavy_win_rate')
        }
        valid_rates = {k: v for k, v in going_rates.items() if pd.notna(v)}
        
        if valid_rates:
            profile['best_going'] = max(valid_rates, key=valid_rates.get)
            profile['best_going_win_rate'] = max(valid_rates.values())
        else:
            profile['best_going'] = 'unknown'
            profile['best_going_win_rate'] = np.nan
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    print(f"  Built profiles for {len(profiles_df):,} horses")
    
    return profiles_df


def engineer_going_features(df, going_profiles):
    """
    Add going preference features to race data.
    """
    print("\nEngineering going preference features...")
    
    df = df.copy()
    df['going_numeric'] = df['going'].map(GOING_MAP).fillna(3)
    
    # Categorize going
    def categorize_going(going_num):
        for cat, (low, high) in GOING_CATEGORIES.items():
            if low < going_num <= high:
                return cat
        return 'good'
    
    df['going_category'] = df['going_numeric'].apply(categorize_going)
    
    # Merge profiles
    df = df.merge(going_profiles, on='horse', how='left')
    
    # Get win rate for this going condition
    def get_going_win_rate(row):
        cat = row['going_category']
        rate = row.get(f'{cat}_win_rate')
        if pd.isna(rate):
            # Fall back to overall win rate
            return row.get('career_win_rate', 0.1)
        return rate
    
    df['going_preference_win_rate'] = df.apply(get_going_win_rate, axis=1)
    
    # Is this the horse's preferred going?
    df['is_preferred_going'] = (df['going_category'] == df['best_going']).astype(int)
    
    # Going suitability score (0-1, higher = better match)
    def going_suitability(row):
        best = row.get('best_going')
        current = row['going_category']
        
        if pd.isna(best) or best == 'unknown':
            return 0.5  # Unknown preference
        
        # Adjacent categories = partial suitability
        order = ['fast', 'good', 'soft', 'heavy']
        if best == current:
            return 1.0
        elif abs(order.index(best) - order.index(current)) == 1:
            return 0.6
        else:
            return 0.2
    
    df['going_suitability'] = df.apply(going_suitability, axis=1)
    
    print(f"  Going features: going_preference_win_rate, is_preferred_going, going_suitability")
    
    return df


if __name__ == '__main__':
    df = pd.read_parquet('data/processed/race_scores.parquet')
    profiles = build_going_profiles(df)
    profiles.to_csv('data/processed/going_profiles.csv', index=False)
    print(f"\n✓ Saved going profiles")
```

---

## 3. Speed Figures & Performance Ratings

**Impact**: Very High  
**Difficulty**: High  
**Data Source**: Calculate from sectional times or finishing margins

### Why It Matters
- Normalize performance across different courses/conditions
- A "fast" 1m on firm ground ≠ a "slow" 1m in heavy
- Speed figures predict future performance better than raw times

### Simplified Speed Figure Calculation

```python
#!/usr/bin/env python3
"""scripts/calculate_speed_figures.py
Calculate simplified speed figures from finishing times and beaten lengths.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Standard times by distance (furlongs) and going - based on class 4 average
STANDARD_TIMES = {
    5: {'fast': 60, 'good': 61, 'soft': 63, 'heavy': 66},
    6: {'fast': 72, 'good': 73, 'soft': 76, 'heavy': 80},
    7: {'fast': 84, 'good': 86, 'soft': 89, 'heavy': 94},
    8: {'fast': 96, 'good': 98, 'soft': 102, 'heavy': 108},
    10: {'fast': 120, 'good': 123, 'soft': 128, 'heavy': 136},
    12: {'fast': 144, 'good': 148, 'soft': 155, 'heavy': 166},
    14: {'fast': 168, 'good': 173, 'soft': 181, 'heavy': 195},
    16: {'fast': 192, 'good': 198, 'soft': 208, 'heavy': 224}
}

# Lengths to seconds conversion (approximate)
LENGTHS_PER_SECOND = 5.0  # ~5 lengths per second at racing pace


def parse_time_seconds(time_str):
    """Parse race time to seconds (e.g., '1:23.45' -> 83.45)"""
    if pd.isna(time_str):
        return np.nan
    
    time_str = str(time_str).strip()
    
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return np.nan


def calculate_speed_figure(winner_time, dist_f, going_category, class_num):
    """
    Calculate speed figure relative to standard time.
    
    Higher = faster = better.
    100 = standard class 4 performance.
    """
    # Get closest standard distance
    dist_key = min(STANDARD_TIMES.keys(), key=lambda x: abs(x - dist_f))
    
    # Get standard time for this distance/going
    standard = STANDARD_TIMES[dist_key].get(going_category, STANDARD_TIMES[dist_key]['good'])
    
    # Adjust standard for distance if not exact
    if dist_f != dist_key:
        # ~12 seconds per furlong at good going
        standard += (dist_f - dist_key) * 12
    
    # Calculate raw speed figure
    # Positive = faster than standard, negative = slower
    if pd.isna(winner_time) or winner_time <= 0:
        return np.nan
    
    diff_seconds = standard - winner_time
    raw_figure = 100 + (diff_seconds * 5)  # 5 points per second
    
    # Class adjustment: higher class should produce higher figures
    class_adjustment = (4 - class_num) * 3  # +9 for class 1, -9 for class 7
    
    return raw_figure + class_adjustment


def calculate_horse_speed_figure(winner_time, btn_lengths, dist_f, going_category, class_num):
    """
    Calculate individual horse's speed figure based on beaten lengths.
    """
    winner_figure = calculate_speed_figure(winner_time, dist_f, going_category, class_num)
    
    if pd.isna(winner_figure):
        return np.nan
    
    # Subtract points for beaten lengths
    # Approximately 0.2 seconds per length = 1 speed point per length
    length_adjustment = btn_lengths * 1.0 if pd.notna(btn_lengths) else 0
    
    return winner_figure - length_adjustment


def engineer_speed_features(df):
    """
    Calculate speed figures for historical data.
    """
    print("\nEngineering speed figure features...")
    
    df = df.copy()
    
    # Parse winner time
    df['winner_time_secs'] = df['time'].apply(parse_time_seconds)
    
    # Parse beaten lengths
    def parse_btn(btn_str):
        if pd.isna(btn_str) or btn_str in ['', '-', 'W', 'won']:
            return 0.0
        try:
            btn_str = str(btn_str).strip().lower()
            if 'nk' in btn_str: return 0.25
            if 'hd' in btn_str: return 0.1
            if 'shd' in btn_str: return 0.05
            if 'nse' in btn_str: return 0.01
            if 'dist' in btn_str: return 30.0
            return float(btn_str)
        except:
            return np.nan
    
    df['btn_lengths'] = df['btn'].apply(parse_btn)
    
    # Going category
    GOING_MAP = {
        'Hard': 'fast', 'Firm': 'fast', 'Good To Firm': 'fast',
        'Good': 'good', 'Standard': 'good',
        'Good To Soft': 'soft', 'Soft': 'soft', 'Standard To Slow': 'soft',
        'Soft To Heavy': 'heavy', 'Heavy': 'heavy', 'Slow': 'heavy'
    }
    df['going_category'] = df['going'].map(GOING_MAP).fillna('good')
    
    # Calculate speed figures
    df['speed_figure'] = df.apply(
        lambda row: calculate_horse_speed_figure(
            row['winner_time_secs'],
            row['btn_lengths'],
            row['dist_f'],
            row['going_category'],
            row['class_num']
        ),
        axis=1
    )
    
    # Historical speed figure features
    df = df.sort_values(['horse', 'date']).reset_index(drop=True)
    
    # Best speed figure (career peak)
    df['best_speed_figure'] = df.groupby('horse')['speed_figure'].transform(
        lambda x: x.shift(1).expanding().max()
    )
    
    # Average last 3 speed figures
    df['avg_speed_figure_3'] = df.groupby('horse')['speed_figure'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Speed figure trend (improving or declining)
    df['speed_figure_trend'] = df.groupby('horse')['speed_figure'].transform(
        lambda x: x.shift(1).diff().rolling(3, min_periods=1).mean()
    )
    
    print(f"  Speed features: speed_figure, best_speed_figure, avg_speed_figure_3, speed_figure_trend")
    
    return df


if __name__ == '__main__':
    df = pd.read_parquet('data/processed/race_scores.parquet')
    df = engineer_speed_features(df)
    df.to_parquet('data/processed/race_scores_with_speed.parquet', index=False)
    print("\n✓ Saved data with speed figures")
```

---

## 4. Market Signal Integration (Odds Movement)

**Impact**: Very High  
**Difficulty**: Medium  
**Data Source**: Multiple odds snapshots or exchange data

### Why It Matters
- "Steam" = sharp money moving a horse in
- Drifters = often a warning sign
- Combining model prediction + market signal = powerful

### Code Implementation

```python
#!/usr/bin/env python3
"""scripts/track_odds_movement.py
Track odds movement from opening to SP.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

def calculate_odds_movement_features(opening_odds, final_odds):
    """
    Calculate features from odds movement.
    
    Args:
        opening_odds: Decimal odds at market open (e.g., 5.0)
        final_odds: Starting price (SP) at race off (e.g., 4.0)
    
    Returns:
        dict: Odds movement features
    """
    features = {}
    
    if pd.isna(opening_odds) or pd.isna(final_odds):
        return {
            'odds_movement': 0,
            'odds_pct_change': 0,
            'is_steamer': 0,
            'is_drifter': 0,
            'market_confidence': 0.5
        }
    
    # Raw movement (negative = shortened/supported)
    features['odds_movement'] = final_odds - opening_odds
    
    # Percentage change
    features['odds_pct_change'] = (final_odds - opening_odds) / opening_odds
    
    # Is this a "steamer" (heavily backed)? 20%+ reduction = steamer
    features['is_steamer'] = 1 if features['odds_pct_change'] < -0.20 else 0
    
    # Is this a "drifter" (money moving away)? 30%+ increase = drifter
    features['is_drifter'] = 1 if features['odds_pct_change'] > 0.30 else 0
    
    # Market confidence (0-1 scale)
    # Lower odds = higher confidence
    features['market_confidence'] = 1 / final_odds
    
    # Implied probability from SP
    features['sp_implied_prob'] = 1 / final_odds
    
    return features


def engineer_market_features(df, odds_history_df):
    """
    Add market signal features from odds history.
    
    Args:
        df: Race data with horse_id, race_id
        odds_history_df: Historical odds snapshots
    """
    print("\nEngineering market signal features...")
    
    df = df.copy()
    
    # Merge odds data
    odds_summary = odds_history_df.groupby(['race_id', 'horse_id']).agg({
        'opening_odds': 'first',
        'final_odds': 'last',  # SP
        'min_odds': 'min',     # Best price available
        'max_odds': 'max'      # Worst price
    }).reset_index()
    
    df = df.merge(odds_summary, on=['race_id', 'horse_id'], how='left')
    
    # Calculate movement features
    movement_features = df.apply(
        lambda row: calculate_odds_movement_features(
            row.get('opening_odds'),
            row.get('final_odds')
        ),
        axis=1
    )
    
    movement_df = pd.DataFrame(movement_features.tolist())
    
    for col in movement_df.columns:
        df[col] = movement_df[col]
    
    # Fill missing with neutral values
    df['odds_movement'] = df['odds_movement'].fillna(0)
    df['is_steamer'] = df['is_steamer'].fillna(0)
    df['is_drifter'] = df['is_drifter'].fillna(0)
    
    print(f"  Market features: odds_movement, is_steamer, is_drifter, market_confidence")
    
    return df


def calculate_value_signal(model_prob, market_prob):
    """
    Calculate value signal from model vs market.
    
    Returns:
        float: Value signal (positive = model sees edge)
    """
    if pd.isna(model_prob) or pd.isna(market_prob):
        return 0
    
    edge = model_prob - market_prob
    
    # Normalize to meaningful scale
    # 5% edge = 0.05, 10% edge = 0.10, etc.
    return edge


def combine_model_and_market(df, model_col='win_probability', market_col='sp_implied_prob'):
    """
    Create combined prediction using model + market signals.
    """
    print("\nCombining model and market signals...")
    
    df = df.copy()
    
    # Value signal
    df['value_edge'] = df[model_col] - df[market_col]
    
    # Combined probability (weighted average)
    # Model weight higher when we have more horse history
    df['model_weight'] = np.clip(df['career_runs'] / 10, 0.3, 0.7)
    
    df['combined_prob'] = (
        df[model_col] * df['model_weight'] + 
        df[market_col] * (1 - df['model_weight'])
    )
    
    # Adjust for steamers (increase confidence)
    df.loc[df['is_steamer'] == 1, 'combined_prob'] *= 1.1
    
    # Adjust for drifters (decrease confidence)
    df.loc[df['is_drifter'] == 1, 'combined_prob'] *= 0.85
    
    # Normalize within race
    df['combined_prob'] = df.groupby(['date', 'course', 'off'])['combined_prob'].transform(
        lambda x: x / x.sum()
    )
    
    print(f"  Combined features: value_edge, combined_prob")
    
    return df
```

---

## 5. Sectional Times & Pace Analysis

**Impact**: High (especially for sprints)  
**Difficulty**: High  
**Data Source**: The Racing API sectional data (where available)

### Why It Matters
- Pace makes the race
- Fast early pace = back-runners advantaged
- Slow early pace = front-runners hold on
- Sectional times reveal true ability

### Code Implementation

```python
#!/usr/bin/env python3
"""scripts/analyze_pace.py
Analyze race pace and sectional times.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def classify_running_style(position_in_running):
    """
    Classify horse's running style from in-running positions.
    
    Args:
        position_in_running: String like "1-1-2-1" or "8-7-5-2"
    
    Returns:
        str: Running style classification
    """
    if pd.isna(position_in_running):
        return 'unknown'
    
    try:
        positions = [int(p) for p in str(position_in_running).split('-') if p.isdigit()]
        if not positions:
            return 'unknown'
        
        early_pos = np.mean(positions[:2]) if len(positions) >= 2 else positions[0]
        final_pos = positions[-1]
        
        if early_pos <= 3:
            if final_pos <= 3:
                return 'front_runner'
            else:
                return 'front_fader'  # Led but weakened
        elif early_pos <= 6:
            return 'tracker'  # Mid-pack
        else:
            if final_pos < early_pos:
                return 'closer'  # Back to front
            else:
                return 'back_marker'  # Never involved
    except:
        return 'unknown'


def engineer_pace_features(df):
    """
    Engineer pace and running style features.
    """
    print("\nEngineering pace features...")
    
    df = df.copy()
    
    # Classify running style
    df['running_style'] = df['in_running'].apply(classify_running_style)
    
    # One-hot encode running style
    style_dummies = pd.get_dummies(df['running_style'], prefix='style')
    df = pd.concat([df, style_dummies], axis=1)
    
    # Historical running style success rate
    df = df.sort_values(['horse', 'date']).reset_index(drop=True)
    
    for style in ['front_runner', 'tracker', 'closer']:
        col = f'style_{style}'
        if col in df.columns:
            # Win rate when running this style
            df[f'{style}_win_rate'] = df.groupby('horse').apply(
                lambda x: (x['won'] * x[col]).shift(1).expanding().sum() / 
                          x[col].shift(1).expanding().sum()
            ).reset_index(drop=True)
            df[f'{style}_win_rate'] = df[f'{style}_win_rate'].fillna(0)
    
    # Preferred style (most common in last 5 runs)
    df['prev_styles'] = df.groupby('horse')['running_style'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).apply(
            lambda y: pd.Series(y).mode().iloc[0] if len(y) > 0 else 'unknown',
            raw=False
        )
    )
    
    print(f"  Pace features: running_style, style_* dummies, *_win_rate")
    
    return df


def calculate_pace_scenario(df):
    """
    Calculate expected pace scenario for a race based on runners' styles.
    """
    print("\nCalculating pace scenarios...")
    
    df = df.copy()
    
    # Count front runners in each race
    race_groups = df.groupby(['date', 'course', 'off'])
    
    front_runner_counts = race_groups['style_front_runner'].sum().reset_index()
    front_runner_counts.columns = ['date', 'course', 'off', 'num_front_runners']
    
    closer_counts = race_groups['style_closer'].sum().reset_index()
    closer_counts.columns = ['date', 'course', 'off', 'num_closers']
    
    df = df.merge(front_runner_counts, on=['date', 'course', 'off'], how='left')
    df = df.merge(closer_counts, on=['date', 'course', 'off'], how='left')
    
    # Pace scenario
    def determine_pace(row):
        front = row.get('num_front_runners', 0)
        closers = row.get('num_closers', 0)
        field = row.get('field_size', 10)
        
        front_pct = front / field
        
        if front_pct >= 0.3:
            return 'fast_pace'
        elif front_pct <= 0.1:
            return 'slow_pace'
        else:
            return 'average_pace'
    
    df['pace_scenario'] = df.apply(determine_pace, axis=1)
    
    # Does this horse's style suit the pace?
    def style_suits_pace(row):
        style = row.get('running_style', 'unknown')
        pace = row.get('pace_scenario', 'average_pace')
        
        # Fast pace favors closers
        if pace == 'fast_pace' and style == 'closer':
            return 1.2  # 20% boost
        elif pace == 'fast_pace' and style == 'front_runner':
            return 0.8  # 20% penalty
        
        # Slow pace favors front runners
        elif pace == 'slow_pace' and style == 'front_runner':
            return 1.2
        elif pace == 'slow_pace' and style == 'closer':
            return 0.8
        
        return 1.0  # Neutral
    
    df['pace_suitability'] = df.apply(style_suits_pace, axis=1)
    
    print(f"  Pace scenario features: num_front_runners, pace_scenario, pace_suitability")
    
    return df
```

---

## 6. Owner/Syndicate Patterns

**Impact**: Low-Medium  
**Difficulty**: Easy  
**Data Source**: The Racing API owner field

### Why It Matters
- Some owners consistently buy well-bred stock
- Certain syndicates aim for specific race types
- Owner+trainer combos can be profitable

### Code Implementation

```python
def engineer_owner_features(df, min_runners=50):
    """
    Calculate owner performance statistics.
    """
    print("\nEngineering owner features...")
    
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Build owner stats with temporal ordering
    owner_stats = {}
    
    owner_features = {
        'owner_runs': [],
        'owner_win_rate': [],
        'owner_place_rate': []
    }
    
    for idx, row in df.iterrows():
        owner = row.get('owner', 'Unknown')
        won = 1 if row.get('pos_clean') == 1 else 0
        placed = 1 if row.get('pos_clean', 99) <= 3 else 0
        
        # Get historical stats
        if owner in owner_stats:
            stats = owner_stats[owner]
            owner_features['owner_runs'].append(stats['runs'])
            owner_features['owner_win_rate'].append(
                stats['wins'] / stats['runs'] if stats['runs'] > 0 else 0
            )
            owner_features['owner_place_rate'].append(
                stats['places'] / stats['runs'] if stats['runs'] > 0 else 0
            )
        else:
            owner_features['owner_runs'].append(0)
            owner_features['owner_win_rate'].append(0)
            owner_features['owner_place_rate'].append(0)
        
        # Update stats
        if owner not in owner_stats:
            owner_stats[owner] = {'runs': 0, 'wins': 0, 'places': 0}
        owner_stats[owner]['runs'] += 1
        owner_stats[owner]['wins'] += won
        owner_stats[owner]['places'] += placed
    
    for col, values in owner_features.items():
        df[col] = values
    
    print(f"  Owner features: owner_runs, owner_win_rate, owner_place_rate")
    
    return df
```

---

## Implementation Roadmap

### Week 1-2: Pedigree System
- [ ] Run `build_pedigree_lookup.py` to create sire/dam tables
- [ ] Add pedigree features to training pipeline
- [ ] Test on maiden/2yo races specifically

### Week 3-4: Going Profiles
- [ ] Build going preference profiles for all horses
- [ ] Add going suitability scoring
- [ ] Backtest on races with going changes

### Week 5-6: Speed Figures
- [ ] Calculate speed figures for historical data
- [ ] Add speed figure features to model
- [ ] Validate speed figure methodology

### Week 7-8: Market Signals
- [ ] Set up odds tracking pipeline
- [ ] Calculate movement features
- [ ] Test combined model+market approach

### Week 9-10: Pace Analysis
- [ ] Classify running styles
- [ ] Calculate pace scenarios
- [ ] Add pace suitability features

### Week 11-12: Integration & Testing
- [ ] Combine all new features
- [ ] Retrain model with full feature set
- [ ] Comprehensive backtesting

---

## Expected Impact

| Feature Category | Est. AUC Improvement | ROI Impact |
|-----------------|---------------------|------------|
| Pedigree | +0.015-0.025 | High for maidens |
| Going Profiles | +0.010-0.020 | High for specialists |
| Speed Figures | +0.020-0.035 | High overall |
| Market Signals | +0.015-0.030 | Very high |
| Pace Analysis | +0.010-0.020 | High for sprints |
| Owner Patterns | +0.003-0.008 | Low |

**Total Medium-Term Expected**: +0.05-0.12 AUC improvement

---

## Data Collection Requirements

### From The Racing API
- `sire`, `sire_id`, `dam`, `dam_id`, `damsire`, `damsire_id`
- `time` (finishing time)
- `btn` (beaten lengths)
- `in_running` (in-running positions)
- `owner`, `owner_id`

### External Data Needed
- Odds snapshots (opening → SP) - Betfair Exchange API
- Sectional times (where available) - TurfTrax or similar

### Estimated API Usage
- One-time historical build: ~500 calls
- Ongoing daily: 5-10 calls per race day
- Stay within 500 calls/month limit by caching heavily
