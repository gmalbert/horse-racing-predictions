# Validation and Backtesting Framework

Proper validation is critical — a model that looks good on paper but fails in production is worse than useless.

---

## Current Validation Issues

### Problem 1: Random Train/Test Split
```python
# WRONG: Current approach
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Why This Is Wrong:**
- Future data leaks into training (races from 2025 train model, test on 2024)
- Overstates performance by 5-15%
- Horses may appear in both train and test sets

### Problem 2: Single Point Evaluation
- Testing on one time period is not robust
- Performance varies by season, track conditions, etc.

### Problem 3: No Betting Simulation
- ROC AUC doesn't translate directly to betting profit
- Need to simulate actual betting strategies

---

## Proper Validation Framework

### 1. Strict Temporal Split

```python
def strict_temporal_split(df, train_end_date, test_end_date):
    """
    Strictly split data by time, ensuring no lookahead.
    """
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    train = df[df['date_dt'] < train_end_date]
    test = df[(df['date_dt'] >= train_end_date) & (df['date_dt'] < test_end_date)]
    
    # CRITICAL: Features must be computed BEFORE the split
    # to ensure no lookahead bias in feature engineering
    
    print(f"Training: {len(train):,} samples ({train['date'].min()} - {train['date'].max()})")
    print(f"Testing:  {len(test):,} samples ({test['date'].min()} - {test['date'].max()})")
    
    return train, test

# Example usage
train, test = strict_temporal_split(
    df,
    train_end_date='2025-10-01',
    test_end_date='2025-12-31'
)
```

### 2. Walk-Forward Cross-Validation

```python
def walk_forward_cv(df, feature_cols, target_col, 
                   train_months=24, test_months=1, n_splits=12):
    """
    Simulate production by training on past, testing on future.
    Roll forward monthly.
    """
    df = df.sort_values('date').copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    results = []
    
    max_date = df['date_dt'].max()
    
    for split_idx in range(n_splits):
        # Define test period
        test_end = max_date - pd.DateOffset(months=split_idx)
        test_start = test_end - pd.DateOffset(months=test_months)
        
        # Define train period
        train_end = test_start
        train_start = train_end - pd.DateOffset(months=train_months)
        
        # Split data
        train = df[(df['date_dt'] >= train_start) & (df['date_dt'] < train_end)]
        test = df[(df['date_dt'] >= test_start) & (df['date_dt'] < test_end)]
        
        if len(train) < 1000 or len(test) < 100:
            continue
        
        # Train model
        model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05)
        model.fit(train[feature_cols], train[target_col])
        
        # Evaluate
        test_preds = model.predict_proba(test[feature_cols])[:, 1]
        
        split_result = {
            'split': split_idx,
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(test_start.date()),
            'test_end': str(test_end.date()),
            'train_size': len(train),
            'test_size': len(test),
            'auc': roc_auc_score(test[target_col], test_preds),
            'top1_accuracy': None,  # Calculated below
            'top3_accuracy': None,
        }
        
        # Calculate top-N accuracy per race
        test_with_preds = test.copy()
        test_with_preds['pred_prob'] = test_preds
        
        race_accuracy = test_with_preds.groupby('race_id').apply(
            lambda g: {
                'winner_in_top1': g.nlargest(1, 'pred_prob')['won'].any(),
                'winner_in_top3': g.nlargest(3, 'pred_prob')['won'].any(),
            }
        )
        
        split_result['top1_accuracy'] = np.mean([r['winner_in_top1'] for r in race_accuracy])
        split_result['top3_accuracy'] = np.mean([r['winner_in_top3'] for r in race_accuracy])
        
        results.append(split_result)
    
    return pd.DataFrame(results)
```

### 3. Leave-One-Race-Out Cross-Validation

```python
def race_level_cv(df, feature_cols, target_col, n_races=1000):
    """
    Leave entire races out for validation.
    Ensures no information from test races leaks into training.
    """
    # Sample races for testing
    all_races = df['race_id'].unique()
    test_races = np.random.choice(all_races, size=n_races, replace=False)
    
    results = []
    
    for race_id in test_races:
        # Get test race
        test = df[df['race_id'] == race_id]
        
        # Train on all other races BEFORE this date
        race_date = test['date'].iloc[0]
        train = df[(df['race_id'] != race_id) & (df['date'] < race_date)]
        
        if len(train) < 1000:
            continue
        
        # Train and predict
        model = XGBClassifier(n_estimators=100, max_depth=5)
        model.fit(train[feature_cols], train[target_col])
        
        preds = model.predict_proba(test[feature_cols])[:, 1]
        test_with_preds = test.copy()
        test_with_preds['pred_prob'] = preds
        
        # Check if our top pick won
        top_pick = test_with_preds.nlargest(1, 'pred_prob')
        results.append({
            'race_id': race_id,
            'top_pick_won': top_pick['won'].values[0],
            'field_size': len(test),
            'actual_winner_prob': test_with_preds[test_with_preds['won'] == 1]['pred_prob'].values[0]
        })
    
    return pd.DataFrame(results)
```

---

## Betting Backtesting

### 1. Level Stakes Simulation

```python
def simulate_level_stakes(predictions_df, min_probability=0.15, stake=1.0):
    """
    Simulate level stakes betting on model selections.
    """
    results = []
    
    for race_id in predictions_df['race_id'].unique():
        race = predictions_df[predictions_df['race_id'] == race_id]
        
        # Get top pick
        top_pick = race.nlargest(1, 'pred_prob').iloc[0]
        
        if top_pick['pred_prob'] < min_probability:
            continue  # Skip low confidence
        
        # Calculate P&L
        if top_pick['won']:
            profit = (top_pick['bsp'] - 1) * stake  # Use BSP as odds
        else:
            profit = -stake
        
        results.append({
            'race_id': race_id,
            'date': top_pick['date'],
            'horse': top_pick['horse'],
            'pred_prob': top_pick['pred_prob'],
            'bsp': top_pick['bsp'],
            'won': top_pick['won'],
            'profit': profit
        })
    
    df = pd.DataFrame(results)
    
    # Summary
    summary = {
        'total_bets': len(df),
        'winners': df['won'].sum(),
        'strike_rate': df['won'].mean(),
        'total_staked': len(df) * stake,
        'total_return': (df['won'] * df['bsp'] * stake).sum(),
        'profit': (df['won'] * df['bsp'] * stake).sum() - len(df) * stake,
        'roi': ((df['won'] * df['bsp'] * stake).sum() - len(df) * stake) / (len(df) * stake)
    }
    
    return df, summary
```

### 2. Value Betting Simulation

```python
def simulate_value_betting(predictions_df, min_edge=0.05, stake=1.0, kelly_fraction=0.25):
    """
    Bet only when model probability > market probability (value bets).
    """
    results = []
    
    for _, row in predictions_df.iterrows():
        market_prob = 1 / row['bsp']
        model_prob = row['pred_prob']
        
        edge = model_prob - market_prob
        
        if edge < min_edge:
            continue  # No value
        
        # Kelly criterion stake sizing
        kelly_stake = (edge * row['bsp'] - (1 - edge)) / row['bsp']
        kelly_stake = max(0, kelly_stake) * kelly_fraction  # Fractional Kelly
        actual_stake = kelly_stake * stake
        
        if actual_stake <= 0:
            continue
        
        if row['won']:
            profit = (row['bsp'] - 1) * actual_stake
        else:
            profit = -actual_stake
        
        results.append({
            'date': row['date'],
            'horse': row['horse'],
            'model_prob': model_prob,
            'market_prob': market_prob,
            'edge': edge,
            'bsp': row['bsp'],
            'stake': actual_stake,
            'won': row['won'],
            'profit': profit
        })
    
    return pd.DataFrame(results)
```

### 3. Drawdown Analysis

```python
def analyze_drawdown(betting_results):
    """
    Calculate maximum drawdown and recovery periods.
    """
    cumulative = betting_results['profit'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    
    return {
        'max_drawdown': drawdown.min(),
        'max_drawdown_pct': drawdown.min() / running_max.max() if running_max.max() > 0 else 0,
        'current_drawdown': drawdown.iloc[-1],
        'peak_profit': running_max.max(),
        'final_profit': cumulative.iloc[-1],
        'sharpe_ratio': betting_results['profit'].mean() / betting_results['profit'].std() * np.sqrt(252)
    }
```

---

## Performance Metrics

### Classification Metrics
```python
def comprehensive_metrics(y_true, y_pred_proba, race_ids):
    """
    Calculate all relevant metrics for racing predictions.
    """
    metrics = {}
    
    # Standard ML metrics
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
    
    # Per-race metrics
    df = pd.DataFrame({
        'race_id': race_ids,
        'y_true': y_true,
        'y_pred': y_pred_proba
    })
    
    race_metrics = df.groupby('race_id').apply(
        lambda g: pd.Series({
            'winner_in_top1': g.nlargest(1, 'y_pred')['y_true'].any(),
            'winner_in_top3': g.nlargest(3, 'y_pred')['y_true'].any(),
            'winner_rank': g['y_pred'].rank(ascending=False)[g['y_true'] == 1].values[0] if g['y_true'].any() else None
        })
    )
    
    metrics['top1_accuracy'] = race_metrics['winner_in_top1'].mean()
    metrics['top3_accuracy'] = race_metrics['winner_in_top3'].mean()
    metrics['avg_winner_rank'] = race_metrics['winner_rank'].mean()
    
    return metrics
```

### What Good Looks Like

| Metric | Poor | Baseline | Good | Excellent |
|--------|------|----------|------|-----------|
| ROC AUC | < 0.60 | 0.65-0.70 | 0.70-0.75 | > 0.75 |
| Top-1 Accuracy | < 15% | 18-22% | 22-28% | > 28% |
| Top-3 Accuracy | < 45% | 50-55% | 55-65% | > 65% |
| Calibration Error | > 0.10 | 0.05-0.10 | 0.03-0.05 | < 0.03 |
| Level Stakes ROI | < -10% | -5% to 0% | 0% to 5% | > 5% |
| Value Bet ROI | < 0% | 0-5% | 5-10% | > 10% |

---

## Reporting Dashboard

```python
def generate_validation_report(model, test_data, feature_cols, target_col):
    """
    Generate comprehensive validation report.
    """
    report = {}
    
    # Predictions
    preds = model.predict_proba(test_data[feature_cols])[:, 1]
    
    # Metrics
    report['metrics'] = comprehensive_metrics(
        test_data[target_col], preds, test_data['race_id']
    )
    
    # Betting simulation (if BSP available)
    if 'bsp' in test_data.columns:
        test_data['pred_prob'] = preds
        _, betting_summary = simulate_level_stakes(test_data)
        report['betting'] = betting_summary
        report['drawdown'] = analyze_drawdown(test_data)
    
    # Feature importance
    report['feature_importance'] = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calibration plot data
    prob_true, prob_pred = calibration_curve(test_data[target_col], preds, n_bins=10)
    report['calibration'] = {
        'predicted': prob_pred.tolist(),
        'actual': prob_true.tolist()
    }
    
    return report
```

---

## Implementation Priority

| Component | Effort | Priority | Notes |
|-----------|--------|----------|-------|
| Temporal split | Low | 🔴 Immediate | Current validation is misleading |
| Walk-forward CV | Medium | 🔴 Week 1 | Essential for honest assessment |
| Level stakes backtest | Low | 🟠 Week 1-2 | Understand actual profit/loss |
| Value betting backtest | Medium | 🟠 Week 2 | Key for betting strategy |
| Calibration analysis | Low | 🟡 Week 2 | Improve probability outputs |
| Drawdown analysis | Low | 🟡 Week 3 | Risk management |
| Full dashboard | Medium | 🟢 Week 4 | Ongoing monitoring |
