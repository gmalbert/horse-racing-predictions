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

---

## IMPLEMENTATION RESULTS (February 2026)

### Features Implemented

✅ **Enhanced Form Features** (6 features)
- `weighted_pos_avg` - Recent positions weighted more heavily (0.5, 0.3, 0.2)
- `pos_pct_last_3` - Position as percentage of field size
- `form_consistency` - Standard deviation of last 5 positions
- `form_trend` - Linear trend of recent positions (improving/declining)
- `form_at_class` - Win rate at this specific class level
- `runs_at_class` - Experience at this class level

✅ **Connections Form V2 Features** (13 features)
- `jockey_form_14d_v2` / `jockey_form_30d_v2` - Recent jockey win rates
- `trainer_form_14d_v2` / `trainer_form_30d_v2` - Recent trainer win rates
- `jockey_hot_v2` / `trainer_hot_v2` - >25% win rate in 30d (hot flags)
- `combo_form_30d_v2` - Trainer-jockey combination win rate
- `combo_hot_v2` - Hot combination flag
- `jockey_runs_14d_v2` / `jockey_runs_30d_v2` - Recent ride counts
- `trainer_runs_14d_v2` / `trainer_runs_30d_v2` - Recent runner counts
- `combo_runs_30d_v2` - Recent combination rides

### Actual Performance Impact

**Model Progression:**

| Model | Features | Train AUC | Test AUC | Δ vs Baseline |
|-------|----------|-----------|----------|---------------|
| Baseline (v2.0) | 72 | 0.7817 | 0.6841 | - |
| + Enhanced Form | 78 | 0.7952 | 0.6930 | **+0.0089** |
| + Connections V2 | 91 | 0.7999 | 0.6984 | **+0.0144** |

**Final Improvement: +0.0144 AUC (+2.10%)**

### Feature Importance (Top 20 - Full Model v2.1)

| Rank | Feature | Importance | Type | New? |
|------|---------|------------|------|------|
| 1 | field_size | 0.0808 | Race Context | - |
| 2 | sprint_specialist | 0.0445 | Pace | - |
| 3 | **pos_pct_last_3** | **0.0339** | **Enhanced Form** | **🆕** |
| 4 | staying_specialist | 0.0328 | Pace | - |
| 5 | class_num | 0.0202 | Class | - |
| 6 | draw | 0.0164 | Draw | - |
| 7 | age_vs_avg | 0.0151 | Age | - |
| 8 | is_pattern | 0.0149 | Race Type | - |
| 9 | prize_log | 0.0144 | Prize | - |
| 10 | career_place_rate | 0.0142 | Career | - |
| 11 | **weighted_pos_avg** | **0.0220** | **Enhanced Form** | **🆕** |
| 12 | **jockey_hot_v2** | **0.0205** | **Connections V2** | **🆕** |
| 13 | trainer_form_30d | 0.0203 | Connections | - |
| 14 | pace_style_presser | 0.0175 | Pace | - |
| 15 | **trainer_form_30d_v2** | **0.0166** | **Connections V2** | **🆕** |
| 16 | **form_at_class** | **0.0164** | **Enhanced Form** | **🆕** |
| 17 | **jockey_runs_14d_v2** | **0.0161** | **Connections V2** | **🆕** |
| 18 | **jockey_runs_30d_v2** | **0.0160** | **Connections V2** | **🆕** |
| 19 | **trainer_form_14d_v2** | **0.0159** | **Connections V2** | **🆕** |
| 20 | avg_last_3_pos | 0.0153 | Recent Form | - |

**Key Insights:**
- `pos_pct_last_3` (position % of field) ranks #3 overall (0.0339 importance)
- `weighted_pos_avg` is highly predictive (0.0220 importance)
- `jockey_hot_v2` ranks in top 15 (0.0205 importance)
- Enhanced form features contribute 3 of top 20 features
- Connections V2 features contribute 5 of top 20 features
- 8 of top 20 features are NEW (40%)

### Coverage Statistics

**Enhanced Form Features:**
- weighted_pos_avg: 62.0% coverage (requires 1+ prior races)
- pos_pct_last_3: 100% coverage (defaults to 0.5 mid-pack)
- form_consistency: 100% coverage
- form_trend: 100% coverage
- form_at_class: 100% coverage (defaults to 0)
- runs_at_class: 100% coverage

**Connections V2 Features:**
- Jockey 14d coverage: 94.8% (232,548 horses)
- Jockey 30d coverage: 97.1% (238,274 horses)
- Trainer 14d coverage: 91.8% (225,249 horses)
- Trainer 30d coverage: 96.2% (236,088 horses)
- Combo 30d coverage: 66.8% (163,804 horses)
- Hot jockeys: 4.5% of horses
- Hot trainers: 5.7% of horses
- Hot combos: 7.3% of horses

### Implementation Notes

**Data Leakage Prevention:**
- All features use `.shift(1)` to exclude current race
- Expanding windows for cumulative stats
- Time-based rolling windows for recent form (14d/30d)
- Manual date filtering for connections form (race_date - timedelta)

**Computation Time:**
- Enhanced form features: ~30 seconds
- Connections V2 features: ~15 minutes (iterative date-based calculations)

**Files Created:**
- `scripts/add_enhanced_form_features.py` - Enhanced form feature engineering
- `scripts/add_connections_form_v2.py` - Connections form V2 engineering
- `scripts/compare_feature_impact.py` - Model comparison and impact analysis
- `data/processed/race_scores_enhanced_form.parquet` - Intermediate dataset (121 cols)
- `data/processed/race_scores_connections_v2.parquet` - Final dataset (140 cols)
- `models/feature_impact_analysis.json` - Performance comparison results
- `models/feature_importance_v2.1.csv` - Full feature importance rankings

### Conclusion

The implementation of **Enhanced Form** and **Connections V2** features successfully improved model performance by **+2.10%** (0.0144 AUC points). The most impactful additions were:

1. **pos_pct_last_3** - Position relative to field size (#3 feature overall)
2. **weighted_pos_avg** - Recency-weighted positions (#11 overall)
3. **jockey_hot_v2** - Hot jockey indicator (#12 overall)
4. **form_at_class** - Class-specific performance (#16 overall)

These features provide the model with more sophisticated form analysis and better understanding of current connections (jockey/trainer) momentum, which are critical factors in horse racing predictions.

**Next Steps:**
- ❌ Market-Derived Features (requires betting exchange data - BSP not available)
- ✅ Enhanced Form Features (IMPLEMENTED - +0.89% AUC)
- ✅ Connections Form V2 (IMPLEMENTED - +0.55% AUC)
- ⏳ Going Preference Features (future enhancement)
- ⏳ Field Strength Features (future enhancement)
