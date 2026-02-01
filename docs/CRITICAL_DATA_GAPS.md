# Critical Data Gaps

Analysis of current predictions reveals several critical data gaps that significantly impact prediction accuracy.

---

## 1. Pedigree/Breeding Data (HIGHEST PRIORITY)

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

### Features to Engineer
| Feature | Description | Expected Impact |
|---------|-------------|-----------------|
| `sire_win_rate` | Sire's progeny overall win rate | High |
| `sire_surface_pref` | Sire win rate on turf vs AW | Medium |
| `sire_distance_pref` | Sire progeny avg winning distance | High |
| `sire_class_avg` | Average class of sire's winners | Medium |
| `dam_offspring_win_rate` | Dam's previous offspring performance | Medium |
| `damsire_stamina_index` | Damsire influence on stamina | Medium |

### Implementation Priority: 🔴 URGENT (Week 1)

---

## 2. Pace/Running Style Analysis

### Why This Matters
- Racing is about race dynamics, not just individual ability
- A race with 5 front-runners creates different conditions vs 5 closers
- "Pace makes the race" — fundamental racing principle
- Current model has NO pace features

### Missing Data
- Horse's preferred running style (leader, presser, mid-pack, closer)
- Sectional times (if available)
- Race pace predictions
- Number of likely pace-setters in field

### Derivable from Current Data
Using historical form, we can classify:
```python
# Example classification based on historical in-running comments
PACE_STYLES = {
    'LEADER': ['led', 'made all', 'front', 'set pace'],
    'PRESSER': ['tracked leader', 'chased', 'prominent'],
    'MIDPACK': ['midfield', 'mid-division', 'waited with'],
    'CLOSER': ['rear', 'held up', 'patient', 'late run']
}
```

### Features to Engineer
| Feature | Description | Expected Impact |
|---------|-------------|-----------------|
| `horse_pace_style` | Categorical: L/P/M/C | High |
| `pace_pressure_score` | Number of front-runners in race | High |
| `style_advantage` | Is this style suited to track/distance? | Medium |
| `likely_pace_scenario` | Fast/moderate/slow pace prediction | High |

### Implementation Priority: 🔴 URGENT (Week 1-2)

---

## 3. Jockey/Trainer Form (Current vs Career)

### Problem with Current Features
- Current model uses `jockey_career_runs` and `jockey_career_win_rate`
- Career stats are **static** — don't capture current form
- A jockey in poor form riding a good horse is a red flag

### Missing Features
| Feature | Description | Why Important |
|---------|-------------|---------------|
| `jockey_form_14d` | Win rate last 14 days | Recent confidence/fitness |
| `jockey_form_30d` | Win rate last 30 days | Sustained form |
| `trainer_form_14d` | Trainer's yard in form? | Yard illness/confidence |
| `trainer_form_30d` | Trainer month form | Pattern detection |
| `jockey_course_recent` | Recent course performance | Current track feel |

### Implementation Approach
```python
def calculate_recent_jockey_form(df, jockey, date, days=14):
    """Calculate jockey's recent win rate before given date."""
    cutoff = date - timedelta(days=days)
    recent = df[(df['jockey'] == jockey) & 
                (df['date'] >= cutoff) & 
                (df['date'] < date)]
    if len(recent) < 3:
        return None  # Insufficient data
    return recent['won'].mean()
```

### Implementation Priority: 🟠 HIGH (Week 2)

---

## 4. Going/Ground Preference Analysis

### Current State
- `going_numeric` is a single numeric encoding
- No horse-specific going preference

### Missing Analysis
- Horse's historical performance by going type
- Going preference score (does horse prefer soft/good/firm?)
- Going vs breeding expectation (heavy ground sire on firm?)

### Features to Engineer
| Feature | Description | Impact |
|---------|-------------|--------|
| `horse_going_pref` | Best going for this horse | High |
| `going_match_score` | How close is today's going to preference? | High |
| `sire_going_stats` | Sire progeny going preferences | Medium |

### Implementation Priority: 🟠 HIGH (Week 2)

---

## 5. Official Rating (OR) Context

### Current Features
- `or_numeric` — raw OR value
- `or_change` — change from last run
- `or_trend_3` — 3-race OR trend

### Missing Context
| Feature | Description | Why Important |
|---------|-------------|---------------|
| `or_vs_race_max` | Horse OR vs highest rated in race | Competitiveness |
| `or_vs_race_avg` | Horse OR vs race average | Above/below average? |
| `or_vs_class_typical` | Horse OR vs typical for this class | Well-handicapped? |
| `or_percentile` | Where horse sits in OR distribution | Elite vs ordinary |
| `or_career_high` | Has horse reached OR peak? | Improvement potential |

### Implementation Priority: 🟡 MEDIUM (Week 3)

---

## 6. Equipment Changes

### Current State
- `has_blinkers`, `has_visor` features show 0 importance
- `gear_changed`, `first_time_blinkers` also 0 importance

### Problem
- These features are likely encoded incorrectly (all 0 or all 1)
- Need to check data extraction from racecards

### Investigation Needed
```python
# Check current values
df['has_blinkers'].value_counts()
df['first_time_blinkers'].value_counts()
```

### Expected Fixes
- Correctly extract headgear from racecard data
- "First-time blinkers" should be a strong signal
- "Blinkers off" (removed) is also meaningful

### Implementation Priority: 🟡 MEDIUM (Week 2)

---

## 7. Weight Analysis Improvements

### Current Features (Working)
- `weight_lbs`, `weight_vs_avg`, `is_top_weight`, `weight_change`

### Missing Features
| Feature | Description | Impact |
|---------|-------------|--------|
| `weight_for_age` | WFA-adjusted weight | High |
| `weight_trend` | Weight carried trend over last 3 runs | Medium |
| `lb_per_length` | Historical lengths beaten per lb | Medium |
| `handicap_efficiency` | Wins per lb carried above minimum | Medium |

### Implementation Priority: 🟡 MEDIUM (Week 3)

---

## Summary: Priority Order

| Priority | Gap | Est. Impact on AUC | Effort |
|----------|-----|-------------------|--------|
| 1 | Pedigree data | +0.02-0.03 | Medium |
| 2 | Pace analysis | +0.02-0.03 | Medium |
| 3 | Jockey/trainer form | +0.01-0.02 | Low |
| 4 | Going preferences | +0.01-0.02 | Low |
| 5 | OR context | +0.01 | Low |
| 6 | Equipment fixes | +0.005-0.01 | Low |
| 7 | Weight improvements | +0.005-0.01 | Low |

**Total Estimated AUC Improvement**: +0.05 to +0.08 (to ~0.72-0.75)
