# Feature Engineering V2

Advanced feature engineering to improve prediction accuracy beyond the current 47-feature model.

---

## Current Feature Analysis

### Features with HIGH Importance (Keep & Enhance)
| Feature | Importance | Enhancement Opportunity |
|---------|------------|------------------------|
| `field_size` | 11.5% | Add field strength metric |
| `is_top_weight` | 5.1% | Add weight context features |
| `avg_last_3_pos` | 5.0% | Weighted recency, position by class |
| `career_place_rate` | 4.0% | Add surface/distance splits |
| `draw` | 4.0% | Track-specific draw models |

### Features with ZERO Importance (Fix or Remove)
| Feature | Current Value | Problem |
|---------|---------------|---------|
| `has_visor` | 0.0 | Likely all same value |
| `has_blinkers` | 0.0 | Likely all same value |
| `is_maiden` | 0.0 | Feature leakage or encoding |
| `is_handicap` | 0.0 | Race-level, not predictive |
| `gear_changed` | 0.0 | Not extracted correctly |
| `first_time_blinkers` | 0.0 | Not extracted correctly |
| `is_sprint/mile/middle/staying` | 0.0 | Redundant with distance |

---

## New Feature Categories

### 1. Pedigree Features (Cold Start Solution)

```python
def engineer_pedigree_features(df, sire_lookup, dam_lookup):
    """
    Add pedigree-based features using pre-computed lookup tables.
    Critical for horses with limited form.
    """
    # Sire statistics
    df['sire_win_rate'] = df['sire_id'].map(sire_lookup['win_rate'])
    df['sire_place_rate'] = df['sire_id'].map(sire_lookup['place_rate'])
    df['sire_avg_or'] = df['sire_id'].map(sire_lookup['avg_or'])
    
    # Surface preference (sire progeny turf vs AW)
    df['sire_turf_edge'] = (
        df['sire_id'].map(sire_lookup['turf_win_rate']) -
        df['sire_id'].map(sire_lookup['aw_win_rate'])
    )
    df['sire_surface_match'] = np.where(
        df['is_turf'] == 1,
        df['sire_turf_edge'],
        -df['sire_turf_edge']
    )
    
    # Distance affinity
    for dist_band in ['sprint', 'mile', 'middle', 'staying']:
        df[f'sire_{dist_band}_rate'] = df['sire_id'].map(
            sire_lookup[f'{dist_band}_win_rate']
        )
    
    # Match sire distance strength to current race
    df['sire_distance_match'] = df.apply(
        lambda r: r[f'sire_{r["distance_band"]}_rate'], axis=1
    )
    
    # Going preference from sire
    df['sire_soft_rate'] = df['sire_id'].map(sire_lookup['soft_win_rate'])
    df['sire_firm_rate'] = df['sire_id'].map(sire_lookup['firm_win_rate'])
    
    return df
```

### 2. Pace Analysis Features

```python
def engineer_pace_features(df):
    """
    Classify running style and predict race pace scenario.
    """
    # Classify horse's running style from historical positions
    # Using early position comments or in-running positions
    
    def classify_pace_style(form_string, in_running_positions):
        """
        Classify based on historical early positions.
        Returns: 'LEADER', 'PRESSER', 'MIDPACK', 'CLOSER'
        """
        if in_running_positions:
            avg_early = np.mean([p[0] for p in in_running_positions])  # First call
            if avg_early <= 2:
                return 'LEADER'
            elif avg_early <= 4:
                return 'PRESSER'
            elif avg_early <= 7:
                return 'MIDPACK'
            else:
                return 'CLOSER'
        return 'UNKNOWN'
    
    df['pace_style'] = df.apply(
        lambda r: classify_pace_style(r['form'], r.get('in_running')), 
        axis=1
    )
    
    # Count pace types in each race
    pace_counts = df.groupby('race_id')['pace_style'].apply(
        lambda x: x.value_counts().to_dict()
    )
    
    # Pace pressure: many front-runners = fast pace
    df['race_leader_count'] = df['race_id'].map(
        lambda r: pace_counts.get(r, {}).get('LEADER', 0)
    )
    df['pace_pressure'] = df['race_leader_count'] / df['field_size']
    
    # Style advantage based on pace scenario
    # Fast pace (many leaders) favors closers
    # Slow pace (few leaders) favors front-runners
    df['style_advantage'] = np.where(
        (df['pace_pressure'] > 0.3) & (df['pace_style'] == 'CLOSER'),
        1.0,
        np.where(
            (df['pace_pressure'] < 0.15) & (df['pace_style'] == 'LEADER'),
            1.0,
            0.0
        )
    )
    
    return df
```

### 3. Market-Derived Features

```python
def engineer_market_features(df, odds_data):
    """
    Features derived from betting market data.
    Betfair SP is highly predictive.
    """
    # Merge BSP data
    df = df.merge(
        odds_data[['race_id', 'horse', 'bsp', 'place_bsp']], 
        on=['race_id', 'horse'], 
        how='left'
    )
    
    # Implied probability from BSP
    df['market_win_prob'] = 1 / df['bsp']
    df['market_place_prob'] = 1 / df['place_bsp']
    
    # Market rank within race
    df['bsp_rank'] = df.groupby('race_id')['bsp'].rank()
    df['is_favourite'] = (df['bsp_rank'] == 1).astype(int)
    df['is_top3_market'] = (df['bsp_rank'] <= 3).astype(int)
    
    # Price movement (if intraday data available)
    if 'opening_price' in odds_data.columns:
        df['price_drift'] = df['bsp'] / df['opening_price']
        df['price_shortened'] = (df['price_drift'] < 0.9).astype(int)
        df['price_drifted'] = (df['price_drift'] > 1.1).astype(int)
    
    # Over-round analysis
    race_over_round = df.groupby('race_id')['market_win_prob'].sum()
    df['race_over_round'] = df['race_id'].map(race_over_round)
    df['adjusted_market_prob'] = df['market_win_prob'] / df['race_over_round']
    
    return df
```

### 4. Recent Form Features (Enhanced)

```python
def engineer_enhanced_form_features(df):
    """
    More sophisticated form analysis.
    """
    # Weighted position average (recent races count more)
    def weighted_pos_avg(positions, weights=[0.5, 0.3, 0.2]):
        """Weight: most recent = 0.5, second = 0.3, third = 0.2"""
        if len(positions) < len(weights):
            weights = weights[:len(positions)]
        return np.average(positions, weights=weights)
    
    # Position relative to field size (1st of 5 vs 1st of 15)
    df['pos_pct_last_3'] = (
        df['avg_last_3_pos'] / 
        df.groupby('horse')['field_size'].transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )
    
    # Consistency (std dev of last 5 positions)
    df['form_consistency'] = df.groupby('horse')['pos_clean'].transform(
        lambda x: x.shift(1).rolling(5).std()
    )
    
    # Improvement trend (are positions getting better?)
    df['form_trend'] = df.groupby('horse')['pos_clean'].transform(
        lambda x: -x.shift(1).rolling(3).apply(
            lambda p: np.polyfit(range(len(p)), p, 1)[0] if len(p) == 3 else 0
        )
    )
    
    # Lengths beaten analysis (more granular than position)
    df['avg_btn_weighted'] = df.groupby('horse')['btn_lengths'].transform(
        lambda x: (x.shift(1) * np.array([0.5, 0.3, 0.2][:len(x.shift(1))])).sum()
    )
    
    # Class-adjusted form
    df['form_at_class'] = df.groupby(['horse', 'class_num'])['won'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    return df
```

### 5. Trainer/Jockey Recent Form

```python
def engineer_connections_form(df):
    """
    Recent form for trainer and jockey (14-day and 30-day).
    """
    for role in ['jockey', 'trainer']:
        for days in [14, 30]:
            # Calculate rolling form
            df = df.sort_values(['date', role])
            
            # Create date-based windows
            df[f'{role}_runs_{days}d'] = df.groupby(role).apply(
                lambda g: g.rolling(f'{days}D', on='date')['race_id'].count()
            ).values
            
            df[f'{role}_wins_{days}d'] = df.groupby(role).apply(
                lambda g: g.rolling(f'{days}D', on='date')['won'].sum()
            ).values
            
            df[f'{role}_form_{days}d'] = (
                df[f'{role}_wins_{days}d'] / 
                df[f'{role}_runs_{days}d'].clip(lower=1)
            )
    
    # Trainer-jockey combination recent success
    df['combo_key'] = df['trainer'] + '_' + df['jockey']
    df['combo_wins_30d'] = df.groupby('combo_key').apply(
        lambda g: g.rolling('30D', on='date')['won'].sum()
    ).values
    
    return df
```

### 6. Going Preference Features

```python
def engineer_going_features(df):
    """
    Horse's going (ground) preferences based on historical performance.
    """
    # Create going categories
    going_map = {
        'firm': ['firm', 'good to firm', 'fast'],
        'good': ['good', 'standard', 'standard to fast'],
        'soft': ['soft', 'good to soft', 'yielding'],
        'heavy': ['heavy', 'soft to heavy', 'slow']
    }
    
    def categorize_going(going_str):
        going_lower = going_str.lower() if going_str else ''
        for category, keywords in going_map.items():
            if any(kw in going_lower for kw in keywords):
                return category
        return 'good'  # default
    
    df['going_category'] = df['going'].apply(categorize_going)
    
    # Performance by going type
    for going_cat in ['firm', 'good', 'soft', 'heavy']:
        going_mask = df['going_category'] == going_cat
        
        df[f'wins_on_{going_cat}'] = df.groupby('horse').apply(
            lambda g: (g[going_mask]['won'].shift(1).cumsum())
        ).values
        
        df[f'runs_on_{going_cat}'] = df.groupby('horse').apply(
            lambda g: going_mask.shift(1).cumsum()
        ).values
        
        df[f'rate_on_{going_cat}'] = (
            df[f'wins_on_{going_cat}'] / 
            df[f'runs_on_{going_cat}'].clip(lower=1)
        )
    
    # Best going for this horse
    going_cols = [f'rate_on_{g}' for g in ['firm', 'good', 'soft', 'heavy']]
    df['best_going'] = df[going_cols].idxmax(axis=1).str.replace('rate_on_', '')
    
    # Going preference match score
    df['going_match'] = df.apply(
        lambda r: r[f'rate_on_{r["going_category"]}'], axis=1
    )
    
    return df
```

### 7. Field Strength Features

```python
def engineer_field_strength(df):
    """
    Measure the quality/competitiveness of the field.
    """
    # Average OR of field
    df['field_avg_or'] = df.groupby('race_id')['or_numeric'].transform('mean')
    df['or_vs_field'] = df['or_numeric'] - df['field_avg_or']
    
    # OR standard deviation (competitive vs one-sided)
    df['field_or_spread'] = df.groupby('race_id')['or_numeric'].transform('std')
    
    # Number of horses with good form (top 3 last time)
    df['top_form_horses'] = df.groupby('race_id')['avg_last_3_pos'].transform(
        lambda x: (x <= 3).sum()
    )
    
    # Favourites in field (by OR or career win rate)
    df['field_avg_win_rate'] = df.groupby('race_id')['career_win_rate'].transform('mean')
    df['win_rate_vs_field'] = df['career_win_rate'] - df['field_avg_win_rate']
    
    return df
```

---

## Feature Selection Strategy

### Remove Low-Value Features
```python
LOW_VALUE_FEATURES = [
    'has_visor', 'has_blinkers', 'gear_changed', 'first_time_blinkers',
    'is_sprint', 'is_mile', 'is_middle', 'is_staying',  # redundant
    'is_maiden', 'is_handicap'  # race-level, not predictive
]
```

### Feature Importance Threshold
- Remove features with importance < 1%
- Or use recursive feature elimination

### Correlation Analysis
```python
# Remove highly correlated features
corr_matrix = df[feature_cols].corr()
high_corr_pairs = np.where(np.abs(corr_matrix) > 0.9)
# Keep the one with higher target correlation
```

---

## Expected Impact

| Feature Category | Est. AUC Improvement | Effort |
|-----------------|---------------------|--------|
| Pedigree | +0.02-0.03 | Medium |
| Pace Analysis | +0.02-0.03 | Medium |
| Market Data | +0.03-0.05 | Low-Medium |
| Enhanced Form | +0.01-0.02 | Low |
| Connections Form | +0.01-0.02 | Low |
| Going Preferences | +0.01-0.02 | Low |
| Field Strength | +0.01 | Low |

**Total Potential Improvement**: +0.08-0.15 (to AUC ~0.75-0.82)

Note: Market data (BSP) alone could provide +0.05 AUC improvement — it's the single most impactful addition.
