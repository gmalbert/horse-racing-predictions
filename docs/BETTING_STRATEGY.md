# Horse Racing Betting Strategy & Data Modeling Roadmap

## Executive Summary

**The Challenge**: Racecards are released 24-48 hours before races, making it impossible to pre-identify specific horses. However, you can pre-identify **race types** with favorable characteristics and build models that predict outcomes based on historical patterns.

**The Solution**: A two-phase approach:
1. **Pre-filter races** based on structural characteristics (class, distance, surface, course, prize money) that are known weeks/months in advance
2. **Apply predictive models** to specific race participants once the racecard is published

---

## Part 1: Understanding the Problem

### What We Know Early (Weeks/Months Ahead)
- Course (e.g., Ascot, Newmarket)
- Date and time
- Race type (handicap, maiden, conditions)
- Distance
- Surface (Turf/All-Weather)
- Class (1-7)
- Age restrictions
- Prize money
- Pattern status (Group 1/2/3, Listed)

### What We Learn Late (24-48 Hours Before)
- Specific horses entered
- Jockeys assigned
- Trainers
- Draw positions
- Weight allocations (handicaps)
- Recent form
- Odds

### The Modeling Dilemma
You **cannot** model specific horses weeks in advance. But you **can** model:
- Which race types produce predictable outcomes
- Which combinations of features (class + distance + surface) favor certain betting strategies
- Historical profitability of specific race characteristics

---

## Part 2: Strategic Approach

### Phase 1: Race Pre-Filtering (Implement First)
Build a **race quality score** to identify high-probability races based on historical data.

**Goal**: Filter 630k+ historical races → identify patterns → predict which upcoming races are worth analyzing when racecards drop.

**Key Insight**: Some race types are more predictable than others. For example:
- **High predictability**: Class 1-2 races at premium courses (form holds up)
- **Medium predictability**: Competitive handicaps with large fields (favorites underperform)
- **Low predictability**: Low-class all-weather maidens (random outcomes)

### Phase 2: Participant Modeling (Implement Second)
Once racecard is published, apply horse-level models.

**Goal**: Predict win probability for each horse based on:
- Historical performance at this distance/surface/course
- Jockey/trainer stats
- Recent form
- Weight carried
- Draw advantage
- Market odds (wisdom of crowd)

---

## Part 3: Race Selection Criteria

### Tier 1: Focus Races (Highest ROI Potential)

**Characteristics to Pre-Filter:**
```python
focus_criteria = {
    'class': ['Class 1', 'Class 2'],  # Quality holds up
    'pattern': ['Group 1', 'Group 2', 'Group 3', 'Listed'],  # Elite horses
    'prize_min': 20000,  # Significant purses attract serious runners
    'field_size': (6, 12),  # Not too small (no value), not too large (chaos)
    'courses': ['Ascot', 'Newmarket', 'York', 'Doncaster'],  # Premium tracks
    'surface': 'Turf',  # More predictable than AW
    'distance_bands': ['Mile', 'Middle']  # Specialists perform consistently
}
```

**Why These Work:**
- Elite races = best horses with proven form
- Medium fields = favorites win ~33-40% (vs. 25% in large handicaps)
- Premium courses = better data quality, consistent track conditions
- Turf mile/middle distance = horses specialize, form is reliable

### Tier 2: Value Handicaps (Medium Risk/Reward)

**Characteristics:**
```python
value_handicap_criteria = {
    'class': ['Class 3', 'Class 4'],
    'type': 'Handicap',
    'field_size': (10, 16),  # Large fields = favorites overbet
    'prize_min': 10000,
    'courses': ['Goodwood', 'Haydock', 'Sandown'],
    'distance_bands': ['Sprint', 'Mile']
}
```

**Strategy:** Back 2nd/3rd favorites or well-handicapped horses (OR trending up).

### Tier 3: Avoid (Low Predictability)

```python
avoid_criteria = {
    'class': ['Class 5', 'Class 6', 'Class 7'],  # Weak form
    'surface': 'All-Weather',  # Unless winter specialist angle
    'race_name_contains': ['Apprentice', 'Amateur'],  # Jockey inexperience
    'field_size': (1, 4),  # Walk-overs or no value
    'maiden_3yo+': True  # Poor horses past their prime
}
```

---

## Part 4: Data Modeling Approach

### Model 1: Race Profitability Predictor (Pre-Race Card)

**Purpose**: Score upcoming races (0-100) based on historical betting ROI.

**Features** (all known in advance):
- Course ID
- Class
- Distance (furlongs)
- Surface
- Prize money
- Field size (estimated from historical average)
- Month/season
- Day of week

**Target Variable**: Historical favorite win rate OR place rate OR exotic payout variance

**Model Type**: Random Forest or XGBoost

**Training Data**:
```python
# Aggregate historical races by characteristics
race_profitability = df.groupby([
    'course', 'class', 'dist_f', 'surface', 'month'
]).agg({
    'pos': lambda x: (x == 1).sum() / len(x),  # Win rate (proxy for favorite performance)
    'prize': 'mean',
    'ran': 'mean'
}).reset_index()
```

**Output**: Probability that a race type will produce a profitable betting opportunity.

### Model 2: Horse Win Probability (Post-Race Card)

**Purpose**: Predict each horse's win probability once participants are known.

**Features** (available 24-48 hours before):
- Horse's historical win % at this distance
- Horse's win % at this course
- Horse's win % on this surface
- Days since last race
- Jockey win % (overall and on this horse)
- Trainer win % at this course
- Official Rating (OR)
- Weight carried
- Draw position (if relevant)
- Recent form (last 3-5 races)
- Market odds (if available early)

**Target**: Binary win (1/0) or finish position (1-20)

**Model Type**: Gradient Boosting (XGBoost/LightGBM) or Neural Network

**Training Data**: Your 630k historical races with horse-level features

### Model 3: Value Bet Identifier

**Purpose**: Find horses whose model probability > implied probability from odds.

**Formula**:
```python
model_prob = 0.25  # Your model says 25% chance
market_odds = 5.0  # Bookmaker offers 5/1
implied_prob = 1 / market_odds  # 0.20 = 20%

if model_prob > implied_prob * 1.1:  # 10% edge threshold
    # This is a value bet
    expected_value = (model_prob * (market_odds - 1)) - (1 - model_prob)
    if expected_value > 0:
        bet_size = kelly_criterion(model_prob, market_odds, bankroll)
```

---

## Part 5: Practical Workflow

### Weekly Planning (Sunday)
1. **Download fixtures** for the next 7 days (from BHA or Racing API)
2. **Score races** using Model 1 (Race Profitability Predictor)
3. **Create watchlist** of top 20-30 races for the week
4. **Schedule analysis** for 48h before each watchlist race

### Daily Operations (Race Day - 2 Days)
1. **Fetch racecards** for tomorrow's watchlist races (via Racing API)
2. **Generate horse features** from historical database
3. **Run Model 2** (Horse Win Probability) for each race
4. **Identify value bets** using Model 3
5. **Review and validate** selections (manual sanity check)
6. **Place bets** 12-24 hours before race (best odds window)

### Post-Race Analysis (Daily)
1. **Record results** (wins, losses, ROI)
2. **Update models** with new data (retrain weekly)
3. **Track model performance** by race type
4. **Adjust strategy** based on profitability metrics

---

## Part 6: Code Samples

### Sample 1: Pre-Filter High-Value Races

```python
import pandas as pd
from datetime import datetime, timedelta

def score_race_profitability(race_row, historical_stats):
    """
    Score a race (0-100) based on historical betting ROI.
    
    Args:
        race_row: dict with course, class, distance, surface
        historical_stats: DataFrame of aggregated historical performance
    
    Returns:
        float: Profitability score (0-100)
    """
    score = 50  # Baseline
    
    # Premium course bonus
    if race_row['course'] in ['Ascot', 'Newmarket', 'York', 'Doncaster']:
        score += 15
    
    # Class bonus (higher class = more predictable)
    class_scores = {'Class 1': 20, 'Class 2': 15, 'Class 3': 10, 'Class 4': 5}
    score += class_scores.get(race_row.get('class'), 0)
    
    # Distance band (specialists = predictable)
    if 7 <= race_row.get('dist_f', 0) <= 12:  # Mile to middle distance
        score += 10
    
    # Field size (medium fields = better odds value)
    field_size = race_row.get('estimated_runners', 10)
    if 8 <= field_size <= 14:
        score += 10
    elif field_size > 16:
        score -= 5  # Large fields = chaos
    
    # Pattern race bonus
    if race_row.get('pattern') in ['Group 1', 'Group 2', 'Group 3', 'Listed']:
        score += 15
    
    # Historical lookup
    hist_match = historical_stats[
        (historical_stats['course'] == race_row['course']) &
        (historical_stats['class'] == race_row.get('class'))
    ]
    if not hist_match.empty:
        # Add bonus based on historical favorite win rate
        fav_win_rate = hist_match.iloc[0].get('favorite_win_rate', 0.33)
        if fav_win_rate > 0.40:
            score += 10  # Favorites perform well = predictable
        elif fav_win_rate < 0.25:
            score -= 10  # Favorites underperform = chaotic
    
    return min(100, max(0, score))


def filter_upcoming_races(fixtures_df, historical_df, min_score=70):
    """
    Filter upcoming races to a watchlist based on profitability score.
    
    Args:
        fixtures_df: DataFrame of upcoming races from BHA/API
        historical_df: Your 630k race database
        min_score: Minimum score threshold (default 70/100)
    
    Returns:
        DataFrame of high-value races to monitor
    """
    # Build historical stats by race type
    historical_stats = historical_df.groupby(['course', 'class']).agg({
        'pos': lambda x: (x == 1).mean(),  # Favorite win rate proxy
        'prize': 'mean',
        'ran': 'mean'
    }).reset_index()
    historical_stats.columns = ['course', 'class', 'favorite_win_rate', 'avg_prize', 'avg_runners']
    
    # Score each upcoming race
    fixtures_df['score'] = fixtures_df.apply(
        lambda row: score_race_profitability(row, historical_stats), axis=1
    )
    
    # Filter and sort
    watchlist = fixtures_df[fixtures_df['score'] >= min_score].copy()
    watchlist = watchlist.sort_values('score', ascending=False)
    
    return watchlist[['date', 'course', 'race_name', 'class', 'dist', 'prize', 'score']]


# Example usage
# fixtures = pd.read_csv('data/processed/bha_2026_ascot_newmarket_doncaster_york.csv')
# historical = pd.read_parquet('data/processed/all_gb_races.parquet')
# watchlist = filter_upcoming_races(fixtures, historical, min_score=75)
# print(watchlist.head(20))
```

### Sample 2: Build Horse Features for Prediction

```python
def build_horse_features(horse_id, race_conditions, historical_df):
    """
    Generate predictive features for a specific horse in a specific race.
    
    Args:
        horse_id: Unique horse identifier
        race_conditions: dict with course, distance, surface, class
        historical_df: Historical race database
    
    Returns:
        dict of features for ML model
    """
    # Filter to this horse's history
    horse_history = historical_df[historical_df['horse_id'] == horse_id].copy()
    
    if horse_history.empty:
        return None  # New horse, no data
    
    # Sort by date
    horse_history = horse_history.sort_values('date', ascending=False)
    
    features = {}
    
    # Overall stats
    features['career_runs'] = len(horse_history)
    features['career_wins'] = (horse_history['pos'] == 1).sum()
    features['career_win_pct'] = features['career_wins'] / features['career_runs']
    features['career_top3_pct'] = (horse_history['pos'] <= 3).sum() / features['career_runs']
    
    # Course-specific
    at_course = horse_history[horse_history['course'] == race_conditions['course']]
    features['course_runs'] = len(at_course)
    features['course_win_pct'] = (at_course['pos'] == 1).mean() if len(at_course) > 0 else 0
    
    # Distance-specific
    target_dist = race_conditions['dist_f']
    at_distance = horse_history[
        (horse_history['dist_f'] >= target_dist - 1) & 
        (horse_history['dist_f'] <= target_dist + 1)
    ]
    features['dist_runs'] = len(at_distance)
    features['dist_win_pct'] = (at_distance['pos'] == 1).mean() if len(at_distance) > 0 else 0
    
    # Surface-specific
    at_surface = horse_history[horse_history['surface'] == race_conditions['surface']]
    features['surface_runs'] = len(at_surface)
    features['surface_win_pct'] = (at_surface['pos'] == 1).mean() if len(at_surface) > 0 else 0
    
    # Recent form (last 3 races)
    recent = horse_history.head(3)
    features['recent_avg_pos'] = recent['pos'].mean()
    features['recent_wins'] = (recent['pos'] == 1).sum()
    features['days_since_last_race'] = (pd.Timestamp.now() - recent.iloc[0]['date']).days if len(recent) > 0 else 999
    
    # Class comparison
    features['avg_class'] = pd.to_numeric(horse_history['class'].str.extract(r'(\d+)')[0], errors='coerce').mean()
    target_class = int(race_conditions.get('class', 'Class 4').replace('Class ', ''))
    features['class_step'] = target_class - features['avg_class']  # Negative = stepping up in class
    
    # Official Rating trend
    if 'or' in horse_history.columns:
        recent_or = horse_history.head(5)['or'].dropna()
        if len(recent_or) >= 2:
            features['or_trend'] = recent_or.iloc[0] - recent_or.iloc[-1]  # Positive = improving
        else:
            features['or_trend'] = 0
    
    return features


def prepare_race_prediction_dataset(racecard, historical_df):
    """
    Prepare full dataset for a specific race once racecard is published.
    
    Args:
        racecard: dict with race details and list of horses
        historical_df: Historical database
    
    Returns:
        DataFrame ready for model prediction
    """
    race_features = []
    
    for horse in racecard['horses']:
        features = build_horse_features(
            horse['horse_id'], 
            racecard['race_conditions'], 
            historical_df
        )
        
        if features:
            # Add race-specific features
            features['horse_id'] = horse['horse_id']
            features['horse_name'] = horse['horse']
            features['draw'] = horse.get('draw')
            features['weight_lbs'] = horse.get('lbs')
            features['jockey_id'] = horse.get('jockey_id')
            features['trainer_id'] = horse.get('trainer_id')
            
            race_features.append(features)
    
    return pd.DataFrame(race_features)


# Example usage
# racecard = fetch_racecard_from_api(racecard_id=12345)
# race_df = prepare_race_prediction_dataset(racecard, historical_df)
# predictions = trained_model.predict_proba(race_df[feature_columns])
# race_df['win_probability'] = predictions[:, 1]
# print(race_df[['horse_name', 'win_probability']].sort_values('win_probability', ascending=False))
```

### Sample 3: Kelly Criterion Bet Sizing

```python
def kelly_criterion(win_prob, decimal_odds, bankroll, fraction=0.25):
    """
    Calculate optimal bet size using fractional Kelly Criterion.
    
    Args:
        win_prob: Your model's win probability (0-1)
        decimal_odds: Bookmaker's decimal odds (e.g., 5.0 for 4/1)
        bankroll: Total betting bankroll
        fraction: Kelly fraction (0.25 = quarter Kelly, safer)
    
    Returns:
        float: Recommended bet size in currency units
    """
    # Kelly formula: f = (bp - q) / b
    # where f = fraction of bankroll, b = decimal odds - 1, p = win prob, q = lose prob
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    
    kelly_pct = (b * p - q) / b
    
    # Apply fractional Kelly for risk management
    kelly_pct = kelly_pct * fraction
    
    # Never bet more than 5% of bankroll (safety cap)
    kelly_pct = min(kelly_pct, 0.05)
    
    # Don't bet if Kelly is negative (no edge)
    if kelly_pct <= 0:
        return 0
    
    return bankroll * kelly_pct


def evaluate_value_bets(predictions_df, odds_df, bankroll=1000):
    """
    Identify value bets by comparing model probabilities to market odds.
    
    Args:
        predictions_df: DataFrame with horse_id, win_probability
        odds_df: DataFrame with horse_id, decimal_odds
        bankroll: Total betting bankroll
    
    Returns:
        DataFrame of recommended bets with sizing
    """
    # Merge predictions with odds
    bets = predictions_df.merge(odds_df, on='horse_id')
    
    # Calculate implied probability from odds
    bets['implied_prob'] = 1 / bets['decimal_odds']
    
    # Calculate edge
    bets['edge'] = bets['win_probability'] - bets['implied_prob']
    
    # Filter to value bets (model prob > market prob + threshold)
    value_threshold = 0.05  # 5% edge minimum
    bets = bets[bets['edge'] > value_threshold].copy()
    
    # Calculate bet sizes
    bets['bet_size'] = bets.apply(
        lambda row: kelly_criterion(
            row['win_probability'], 
            row['decimal_odds'], 
            bankroll, 
            fraction=0.25
        ), 
        axis=1
    )
    
    # Filter out zero bets
    bets = bets[bets['bet_size'] > 0].copy()
    
    # Sort by edge (best value first)
    bets = bets.sort_values('edge', ascending=False)
    
    return bets[['horse_name', 'win_probability', 'decimal_odds', 'implied_prob', 'edge', 'bet_size']]


# Example usage
# predictions = pd.DataFrame({
#     'horse_id': [1, 2, 3],
#     'horse_name': ['Red Rum', 'Seabiscuit', 'Secretariat'],
#     'win_probability': [0.35, 0.25, 0.15]
# })
# odds = pd.DataFrame({
#     'horse_id': [1, 2, 3],
#     'decimal_odds': [3.5, 5.0, 8.0]
# })
# value_bets = evaluate_value_bets(predictions, odds, bankroll=1000)
# print(value_bets)
```

---

## Part 7: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Set up data infrastructure and basic analysis

- [ ] **Task 1.1**: Clean and validate historical database (630k races)
  - Remove duplicates
  - Handle missing values
  - Standardize course names, class labels
  - Create distance/class/surface lookup tables

- [ ] **Task 1.2**: Build aggregation scripts
  - Course-level stats (win rates, average field size)
  - Distance-level stats (favorite performance)
  - Class-level stats (ROI by class)

- [ ] **Task 1.3**: Create exploratory analysis notebook
  - Analyze which race types are most profitable
  - Identify courses with consistent form
  - Determine optimal field sizes for betting

**Deliverable**: Jupyter notebook with key insights + clean database

### Phase 2: Race Pre-Filtering (Weeks 3-4)
**Goal**: Build system to score upcoming races

- [ ] **Task 2.1**: Implement race scoring function
  - Code: `score_race_profitability()` (see Sample 1)
  - Test on historical data
  - Validate scores match actual profitability

- [ ] **Task 2.2**: Automate fixture ingestion
  - Pull weekly fixtures from BHA or Racing API
  - Parse into structured format
  - Store in `data/processed/upcoming_races.csv`

- [ ] **Task 2.3**: Build watchlist generator
  - Score all upcoming races
  - Filter to top 20-30 per week
  - Export to `data/processed/watchlist.csv`

**Deliverable**: Automated script that runs weekly and generates watchlist

### Phase 3: Feature Engineering (Weeks 5-6)
**Goal**: Build horse-level features for prediction

- [ ] **Task 3.1**: Create horse feature builder
  - Code: `build_horse_features()` (see Sample 2)
  - Test on sample horses
  - Validate features are accurate

- [ ] **Task 3.2**: Add jockey/trainer features
  - Win % at course
  - Win % with specific horse
  - Recent form (last 30 days)

- [ ] **Task 3.3**: Fetch and integrate odds data
  - Connect to odds API (Betfair, Oddschecker, or Racing API)
  - Store opening odds and SP (starting price)
  - Calculate implied probabilities

**Deliverable**: Feature pipeline that generates ML-ready dataset from racecard

### Phase 4: Model Development (Weeks 7-9)
**Goal**: Train and validate predictive models

- [ ] **Task 4.1**: Train Model 1 (Race Profitability)
  - Use historical races grouped by characteristics
  - Target: Favorite win rate or ROI
  - Validate with holdout set (2024-2025 data)

- [ ] **Task 4.2**: Train Model 2 (Horse Win Probability)
  - Use horse-level features
  - Target: Binary win (1/0)
  - Test multiple algorithms (XGBoost, LightGBM, Neural Net)
  - Cross-validate by year to prevent overfitting

- [ ] **Task 4.3**: Backtest on historical racecards
  - Simulate betting on 2023-2024 races
  - Calculate ROI, win rate, profitability by race type
  - Refine models based on results

**Deliverable**: Trained models with >50% ROI on validation set

### Phase 5: Betting System (Weeks 10-12)
**Goal**: Deploy live betting pipeline

- [ ] **Task 5.1**: Build value bet identifier
  - Code: `evaluate_value_bets()` (see Sample 3)
  - Implement Kelly Criterion sizing
  - Add safety caps (max 5% bankroll per bet)

- [ ] **Task 5.2**: Create daily automation
  - Cron job to fetch racecards 48h before race
  - Run predictions on watchlist races
  - Generate bet recommendations
  - Send alerts (email/Slack/Discord)

- [ ] **Task 5.3**: Build tracking dashboard
  - Streamlit page for bet history
  - Track: bets placed, results, ROI, bankroll
  - Visualize performance by race type

**Deliverable**: Automated system that runs daily and generates bet slips

### Phase 6: Optimization & Scaling (Weeks 13+)
**Goal**: Improve performance and expand coverage

- [ ] **Task 6.1**: A/B test strategies
  - Compare Model 2 variants (different features, algorithms)
  - Test different edge thresholds (5% vs. 10%)
  - Experiment with exotic bets (exactas, trifectas)

- [ ] **Task 6.2**: Expand to more courses
  - Add mid-tier courses (Chester, Newbury, etc.)
  - Test if models generalize beyond premium tracks

- [ ] **Task 6.3**: Incorporate live odds movement
  - Track odds drift (opening → SP)
  - Detect smart money (sharp bettors)
  - Adjust predictions based on market signals

**Deliverable**: Optimized system with proven ROI over 3+ months

---

## Part 8: Risk Management & Reality Check

### Bankroll Management
- **Never bet more than 5% of bankroll on a single race**
- Use fractional Kelly (25% Kelly = safer than full Kelly)
- Track ROI weekly; pause if down >20% from starting bankroll
- Start with small stakes ($10-50) until system is proven

### Expected Outcomes
- **Realistic win rate**: 15-25% of bets will win (betting underdogs)
- **Target ROI**: 5-15% over long term (hundreds of bets)
- **Variance**: Expect losing streaks of 10-15 bets
- **Breakeven**: May take 6-12 months to validate system

### Common Pitfalls to Avoid
1. **Overfitting**: Training on all data → no holdout → model fails live
2. **Selection bias**: Only betting on races you "feel good about"
3. **Chasing losses**: Doubling bets after losses (Kelly prevents this)
4. **Ignoring market**: Odds contain information; don't bet blindly
5. **No tracking**: If you don't record bets, you can't improve

### When to Stop
- If ROI is negative after 200+ bets, reassess models
- If bankroll drops below 50% of starting capital, pause and analyze
- If you're not enjoying the process, it's not worth it

---

## Part 9: Key Takeaways

### What Makes This Strategy Work

1. **Pre-filtering reduces noise**: Focus on 20-30 races/week instead of 200+
2. **Historical data guides selection**: 630k races = robust patterns
3. **Two-phase approach**: Race scoring (weeks ahead) + horse modeling (24-48h)
4. **Value-based betting**: Only bet when edge exists (model prob > market prob)
5. **Risk management**: Kelly Criterion prevents ruin

### Why Horse Racing Is Beatable (Sometimes)

- **Information asymmetry**: You have 630k races; casual bettors don't
- **Market inefficiency**: Small races are under-analyzed
- **Form cycles**: Horses improve/decline predictably
- **Course specialists**: Some horses excel at specific tracks
- **Jockey/trainer combos**: Patterns exist in partnerships

### Reality Check

- **Hard work**: Building this system = 100+ hours
- **Patience required**: Need 6-12 months to validate
- **No guarantees**: Even good models can lose money
- **Margins are thin**: 10% ROI is excellent; don't expect 50%
- **Enjoy the process**: If you're not learning/enjoying, don't do it

---

## Part 10: Next Steps

### This Week
1. Review historical data quality (check for missing values, duplicates)
2. Run exploratory analysis on race types (which are profitable?)
3. Choose 4 target courses to focus on initially

### This Month
1. Implement race scoring function
2. Create watchlist for upcoming races
3. Start collecting racecard data manually (to test workflow)

### This Quarter
1. Build and train Model 2 (horse win probability)
2. Backtest on 2023-2024 data
3. Paper trade (track hypothetical bets without real money)

### Long-Term
1. Deploy automated system
2. Track live results for 3-6 months
3. Refine models based on performance
4. Scale bankroll if profitable

---

## Appendix: Useful Resources

### Data Sources
- **BHA Fixtures**: https://www.britishhorseracing.com/fixtures-results/
- **Racing API**: https://theracingapi.com (racecards, results, form)
- **Odds APIs**: Betfair API, Oddschecker, The Odds API

### Tools & Libraries
- **pandas**: Data manipulation
- **scikit-learn**: ML models (Random Forest, Logistic Regression)
- **xgboost/lightgbm**: Gradient boosting (best for tabular data)
- **streamlit**: Dashboard for tracking bets
- **prophet**: Time series forecasting (optional for form trends)

### Further Reading
- "Betfair Trading Techniques" by Wayne Bailey
- "Monte Carlo Methods in Financial Engineering" (for Kelly Criterion)
- "Applied Predictive Modeling" by Kuhn & Johnson
- PunterSquare blog (UK horse racing analytics)

---

**Document Version**: 1.0  
**Last Updated**: December 20, 2025  
**Status**: Initial strategic framework — ready for Phase 1 implementation

