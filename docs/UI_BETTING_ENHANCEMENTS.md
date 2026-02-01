# UI and Betting Enhancements

Improvements to the Streamlit interface and betting strategy implementation.

---

## Current UI Issues

1. **No result tracking** — Can't see if past predictions were correct
2. **No confidence display** — All predictions shown equally
3. **No market comparison** — No bookmaker odds integration
4. **Limited filtering** — Hard to find value bets
5. **No bankroll management** — No stake sizing guidance

---

## Proposed UI Enhancements

### 1. Prediction Result Tracking

Show historical accuracy of predictions:

```python
def render_prediction_history():
    """Show how past predictions performed."""
    st.header("📊 Prediction Track Record")
    
    # Load past predictions with results
    history = load_prediction_history()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(history))
    with col2:
        st.metric("Winners", history['won'].sum())
    with col3:
        st.metric("Strike Rate", f"{history['won'].mean()*100:.1f}%")
    with col4:
        roi = calculate_roi(history)
        st.metric("ROI", f"{roi:.1f}%", delta_color="normal")
    
    # Rolling performance chart
    fig = px.line(
        history.groupby('date')['won'].mean().rolling(7).mean(),
        title="7-Day Rolling Strike Rate"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent results table
    st.subheader("Recent Predictions")
    recent = history.tail(20)[['date', 'course', 'horse', 'pred_prob', 'bsp', 'won', 'profit']]
    st.dataframe(
        recent.style.applymap(
            lambda x: 'background-color: #90EE90' if x == True else 'background-color: #FFB6C1',
            subset=['won']
        )
    )
```

### 2. Confidence Tiers

Display predictions with clear confidence levels:

```python
def render_confidence_tiers(predictions_df):
    """Show predictions grouped by confidence level."""
    st.header("🎯 Predictions by Confidence")
    
    # Define tiers
    predictions_df['tier'] = pd.cut(
        predictions_df['win_probability'],
        bins=[0, 0.10, 0.20, 0.35, 1.0],
        labels=['⚪ Low', '🟡 Medium', '🟠 High', '🔴 Very High']
    )
    
    # Tab for each tier
    tabs = st.tabs(['🔴 Very High', '🟠 High', '🟡 Medium', '⚪ Low'])
    
    for i, tier in enumerate(['🔴 Very High', '🟠 High', '🟡 Medium', '⚪ Low']):
        with tabs[i]:
            tier_preds = predictions_df[predictions_df['tier'] == tier]
            if len(tier_preds) == 0:
                st.info(f"No {tier} confidence predictions today")
            else:
                # Show with styling
                for _, row in tier_preds.iterrows():
                    with st.expander(f"{row['course']} {row['race_time']} - {row['horse']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Win Prob", f"{row['win_probability']*100:.1f}%")
                        col2.metric("Model Odds", row['win_odds_fractional'])
                        col3.metric("Class", row['race_class'])
                        
                        st.write(f"**Jockey:** {row['jockey']} | **Trainer:** {row['trainer']}")
                        st.write(f"**Form:** {row.get('form', 'N/A')}")
```

### 3. Value Betting Finder

Compare model odds to bookmaker odds:

```python
def render_value_finder(predictions_df, bookmaker_odds):
    """Find value bets where model disagrees with market."""
    st.header("💎 Value Bet Finder")
    
    # Merge with bookmaker odds
    merged = predictions_df.merge(
        bookmaker_odds,
        on=['date', 'course', 'race_time', 'horse'],
        how='left',
        suffixes=('_model', '_bookie')
    )
    
    # Calculate edge
    merged['model_implied'] = 1 / merged['win_odds_decimal_model']
    merged['bookie_implied'] = 1 / merged['bookie_odds']
    merged['edge'] = merged['model_implied'] - merged['bookie_implied']
    merged['edge_pct'] = merged['edge'] * 100
    
    # Filter to value bets
    value_bets = merged[merged['edge_pct'] > 5].sort_values('edge_pct', ascending=False)
    
    if len(value_bets) == 0:
        st.warning("No value bets found today (edge > 5%)")
        return
    
    st.success(f"Found {len(value_bets)} potential value bets!")
    
    for _, bet in value_bets.head(10).iterrows():
        with st.container():
            st.markdown(f"### {bet['horse']} - {bet['course']} {bet['race_time']}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Odds", bet['win_odds_fractional_model'])
            col2.metric("Bookie Odds", f"{bet['bookie_odds']:.2f}")
            col3.metric("Edge", f"+{bet['edge_pct']:.1f}%", delta=f"{bet['edge_pct']:.1f}%")
            col4.metric("Suggested Stake", f"£{calculate_kelly_stake(bet):.2f}")
            
            st.divider()
```

### 4. Race Card View Enhancement

Rich race card with all horses and insights:

```python
def render_race_card(race_predictions, race_info):
    """Display detailed race card with all horses."""
    st.subheader(f"🏇 {race_info['race_time']} - {race_info['race_name']}")
    
    # Race info bar
    cols = st.columns(6)
    cols[0].metric("Class", race_info['race_class'])
    cols[1].metric("Distance", race_info['distance_f'])
    cols[2].metric("Surface", race_info['surface'])
    cols[3].metric("Going", race_info['going'])
    cols[4].metric("Runners", race_info['field_size'])
    cols[5].metric("Prize", f"£{race_info.get('prize', 'N/A')}")
    
    # Sort by probability
    race_predictions = race_predictions.sort_values('win_probability', ascending=False)
    
    # Create visual table
    for rank, (_, horse) in enumerate(race_predictions.iterrows(), 1):
        prob = horse['win_probability']
        
        # Color code by rank
        if rank == 1:
            bg_color = "#FFD700"  # Gold
            icon = "🥇"
        elif rank == 2:
            bg_color = "#C0C0C0"  # Silver
            icon = "🥈"
        elif rank == 3:
            bg_color = "#CD7F32"  # Bronze
            icon = "🥉"
        else:
            bg_color = "#FFFFFF"
            icon = f"#{rank}"
        
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 2, 2, 2, 2])
            
            col1.markdown(f"**{icon}**")
            col2.markdown(f"**{horse['horse']}**")
            col3.markdown(f"Win: **{prob*100:.1f}%**")
            col4.markdown(f"Odds: **{horse['win_odds_fractional']}**")
            col5.markdown(f"Jockey: {horse['jockey'][:15]}")
            col6.markdown(f"Form: {horse.get('form', '-')}")
            
            # Probability bar
            st.progress(min(prob * 3, 1.0))  # Scale for visibility
    
    # Model insights
    with st.expander("📈 Model Insights"):
        top3 = race_predictions.head(3)
        
        # Exacta/Trifecta probabilities
        exacta_prob = top3.iloc[0]['win_probability'] * (top3.iloc[1]['win_probability'] / (1 - top3.iloc[0]['win_probability']))
        trifecta_prob = exacta_prob * (top3.iloc[2]['win_probability'] / (1 - top3.iloc[0]['win_probability'] - top3.iloc[1]['win_probability']))
        
        st.write(f"**Exacta (1-2):** {exacta_prob*100:.2f}% (Fair odds: {1/exacta_prob:.1f}/1)")
        st.write(f"**Trifecta (1-2-3):** {trifecta_prob*100:.3f}% (Fair odds: {1/trifecta_prob:.0f}/1)")
        
        # Field strength assessment
        avg_prob = race_predictions['win_probability'].mean()
        spread = race_predictions['win_probability'].std()
        
        if spread < 0.05:
            st.info("⚖️ **Competitive field** - Probabilities are close")
        elif race_predictions.iloc[0]['win_probability'] > 0.40:
            st.info("🎯 **Strong favourite** - Top pick stands out")
        else:
            st.info("📊 **Open race** - Multiple contenders")
```

### 5. Bankroll Management

```python
def render_bankroll_tracker():
    """Track bankroll and suggest stakes."""
    st.header("💰 Bankroll Management")
    
    # Load or initialize bankroll
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000.0
    
    # Display current bankroll
    st.metric(
        "Current Bankroll", 
        f"£{st.session_state.bankroll:.2f}",
        delta=f"£{st.session_state.get('pnl_today', 0):.2f} today"
    )
    
    # Stake calculator
    st.subheader("Stake Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        model_prob = st.slider("Model Win Probability", 0.05, 0.50, 0.20, 0.01)
    with col2:
        bookie_odds = st.number_input("Bookmaker Odds (decimal)", 2.0, 50.0, 5.0, 0.5)
    
    # Kelly calculation
    market_prob = 1 / bookie_odds
    edge = model_prob - market_prob
    
    if edge <= 0:
        st.warning("❌ No value - don't bet!")
    else:
        full_kelly = (model_prob * bookie_odds - 1) / (bookie_odds - 1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Edge", f"+{edge*100:.1f}%")
        col2.metric("Full Kelly", f"{full_kelly*100:.1f}%")
        col3.metric("Quarter Kelly", f"{full_kelly*25:.1f}%")
        
        suggested_stake = st.session_state.bankroll * full_kelly * 0.25
        st.success(f"💡 Suggested stake: **£{suggested_stake:.2f}** (quarter Kelly)")
    
    # Record bet
    if st.button("Record Bet"):
        st.session_state.bets = st.session_state.get('bets', [])
        st.session_state.bets.append({
            'date': datetime.now(),
            'stake': suggested_stake,
            'odds': bookie_odds,
            'result': None  # To be filled in later
        })
        st.success("Bet recorded!")
```

---

## Betting Strategy Improvements

### 1. Multi-Factor Value Identification

```python
def calculate_bet_score(prediction, market_odds, conditions):
    """
    Multi-factor score for bet quality.
    Higher = more attractive bet.
    """
    score = 0
    
    # Edge factor (30%)
    edge = prediction['win_probability'] - (1 / market_odds)
    edge_score = min(edge * 100, 30)
    score += edge_score
    
    # Confidence factor (25%)
    if prediction['win_probability'] > 0.25:
        score += 25
    elif prediction['win_probability'] > 0.18:
        score += 15
    elif prediction['win_probability'] > 0.12:
        score += 5
    
    # Form factor (20%)
    if prediction.get('avg_last_3_pos', 10) <= 3:
        score += 20
    elif prediction.get('avg_last_3_pos', 10) <= 5:
        score += 10
    
    # Jockey/trainer factor (15%)
    if prediction.get('jockey_form_14d', 0) > 0.20:
        score += 10
    if prediction.get('trainer_form_14d', 0) > 0.15:
        score += 5
    
    # Going factor (10%)
    if prediction.get('going_match', 0) > 0.15:
        score += 10
    
    return score
```

### 2. Staking Strategies

```python
class StakingStrategy:
    """Various staking approaches."""
    
    @staticmethod
    def level_stakes(bankroll, unit=0.02):
        """Fixed percentage of bankroll."""
        return bankroll * unit
    
    @staticmethod
    def kelly(bankroll, prob, odds, fraction=0.25):
        """Kelly criterion with fraction."""
        q = 1 - prob
        f = (prob * odds - q) / odds
        f = max(0, f) * fraction
        return bankroll * f
    
    @staticmethod
    def confidence_based(bankroll, prob, base=0.01):
        """Scale stake by confidence."""
        if prob > 0.30:
            multiplier = 3.0
        elif prob > 0.22:
            multiplier = 2.0
        elif prob > 0.15:
            multiplier = 1.5
        else:
            multiplier = 1.0
        return bankroll * base * multiplier
    
    @staticmethod
    def target_profit(bankroll, prob, odds, target_pct=0.05):
        """Stake to achieve target profit."""
        potential_profit = bankroll * target_pct
        stake = potential_profit / (odds - 1)
        # Cap at 5% of bankroll
        return min(stake, bankroll * 0.05)
```

### 3. When to Bet / When to Pass

```python
def should_bet(prediction, market, conditions):
    """Decision framework for placing bets."""
    
    # Hard stops
    if prediction['win_probability'] < 0.08:
        return False, "Probability too low"
    
    if prediction.get('career_runs', 0) == 0 and prediction.get('sire_win_rate', 0) < 0.10:
        return False, "Insufficient form data"
    
    # Value requirement
    edge = prediction['win_probability'] - (1 / market['bsp'])
    if edge < 0.03:
        return False, "Insufficient edge"
    
    # Recent form requirement
    if prediction.get('avg_last_3_pos', 10) > 7:
        return False, "Poor recent form"
    
    # Going mismatch
    if prediction.get('going_match', 0) < 0.05:
        return False, "Going doesn't suit"
    
    # All checks passed
    return True, f"Value bet: {edge*100:.1f}% edge"
```

---

## Implementation Priority

| Enhancement | Effort | User Value | Priority |
|-------------|--------|------------|----------|
| Result tracking | Medium | Very High | 🔴 Week 1-2 |
| Confidence tiers | Low | High | 🔴 Week 1 |
| Value bet finder | Medium | Very High | 🟠 Week 2-3 |
| Race card view | Medium | High | 🟠 Week 2 |
| Bankroll tracker | Low | Medium | 🟡 Week 3 |
| Staking calculator | Low | High | 🟡 Week 3 |
| Multi-factor scoring | Medium | High | 🟢 Week 4 |

---

## Technical Considerations

### Session State Management
```python
# Use Streamlit session state for persistent data
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = load_history_from_disk()
```

### Data Persistence
```python
# Save results after each race day
def save_results(results):
    results_file = Path('data/processed/betting_results.csv')
    if results_file.exists():
        existing = pd.read_csv(results_file)
        combined = pd.concat([existing, results])
    else:
        combined = results
    combined.to_csv(results_file, index=False)
```

### Real-time Updates
```python
# Auto-refresh for live racing
if st.checkbox("Auto-refresh (live racing)"):
    st_autorefresh(interval=60000)  # 60 seconds
```
