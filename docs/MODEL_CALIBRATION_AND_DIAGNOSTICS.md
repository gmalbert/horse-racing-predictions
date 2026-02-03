# Model Calibration and Diagnostics

## Overview

The prediction system now includes automatic model diagnostics and optional probability calibration to improve prediction accuracy and reliability.

## Features

### 1. **Automatic Diagnostics** 
Every time predictions are generated, the system automatically creates diagnostic metrics:
- Probability distribution analysis
- Cold start horse detection
- Feature coverage statistics
- Race class distribution
- Top pick probability ranges

**Location**: `data/processed/model_diagnostics.json`

### 2. **Probability Calibration**
Calibrated models produce more accurate probability estimates by adjusting raw model outputs to match empirical frequencies.

**Benefits:**
- Better probability estimates (Brier score improvement ~2-5%)
- More reliable betting decisions
- Improved confidence calibration

**Location**: `models/horse_win_predictor_calibrated.pkl`

## Usage

### Generate Predictions with Diagnostics

Predictions automatically include diagnostics:

```bash
python scripts/predict_todays_races.py --date 2026-02-02
```

This creates:
- `data/processed/predictions_2026-02-02.csv` - Predictions
- `data/processed/diagnostics_2026-02-02.json` - Date-specific diagnostics
- `data/processed/model_diagnostics.json` - Latest diagnostics (for UI)

### Calibrate the Model

Run once after training or retraining:

```bash
python scripts/calibrate_model.py
```

This creates:
- `models/horse_win_predictor_calibrated.pkl` - Calibrated model
- `models/calibration_metrics.json` - Calibration performance metrics
- `models/calibration_plot.png` - Visual calibration curve

**Note**: Future predictions will automatically use the calibrated model if it exists.

### Run Diagnostics

Analyze existing predictions:

```bash
python scripts/diagnose_model.py
```

This analyzes all prediction files and shows:
- Win probability distribution of top picks
- Cold start horse percentages
- Feature availability and coverage
- Class distribution
- Feature value checks

## Viewing Results in the UI

### Model Insights Tab

The Streamlit UI now shows comprehensive model information:

1. **Model Status**
   - Model type and features
   - Training date

2. **Calibration Section**
   - Calibration status (✅ calibrated or ℹ️ not calibrated)
   - Brier score metrics
   - Calibration improvement percentage
   - Visual calibration curve
   - One-click calibration button

3. **Latest Diagnostics**
   - Prediction date
   - Total races and horses analyzed
   - Average field size
   - Probability distribution (mean, max)
   - Top pick probabilities
   - Cold start analysis
   - Full diagnostics JSON viewer

4. **Feature Importance**
   - Top 15 features ranked
   - Interactive bar chart

### Accessing the UI

```bash
streamlit run predictions.py
```

Navigate to the **"📊 Model Insights"** tab to see all metrics.

## Diagnostic Metrics Explained

### Probability Distribution
- **Mean Win Prob**: Average across all horses (~10% expected for balanced races)
- **Highest Prob**: The most confident prediction
- **Avg Top Pick**: Average probability of the top-ranked horse per race (should be 20-40%)

### Cold Start Analysis
- **Cold Start Horses**: Horses with 0 career runs (no historical data)
- **Percentage**: Should be <20% after pedigree features were added

### Feature Coverage
- **Null %**: Percentage of missing values
- **Zero %**: Percentage of zero values (may indicate missing data or true zeros)

## Calibration Metrics Explained

### Brier Score
- Measures accuracy of probabilistic predictions
- Range: 0 (perfect) to 1 (worst)
- Target: <0.10 for good calibration
- Improvement of 2-5% is typical from calibration

### Log Loss
- Alternative probability accuracy metric
- Lower is better
- More sensitive to extreme predictions

### Calibration Curve
- **Perfect calibration**: Points lie on diagonal line
- **Above diagonal**: Model is underconfident
- **Below diagonal**: Model is overconfident

## Best Practices

1. **Calibrate After Retraining**
   - Run `calibrate_model.py` whenever you retrain the model
   - Uses last 3 months of data for calibration

2. **Monitor Diagnostics**
   - Check diagnostics regularly to spot data quality issues
   - Watch for increasing cold start percentages

3. **Check Feature Coverage**
   - High null/zero percentages may indicate data pipeline issues
   - Investigate features with >30% missing values

4. **Validate Calibration**
   - Review calibration plot visually
   - Ensure Brier score improvement is positive
   - Re-calibrate if calibration data becomes stale (>3 months old)

## Workflow Integration

### Daily Workflow
```bash
# 1. Fetch racecards
python scripts/fetch_racecards.py --date 2026-02-02

# 2. Generate predictions (includes diagnostics automatically)
python scripts/predict_todays_races.py --date 2026-02-02

# 3. View in UI
streamlit run predictions.py
```

### Weekly Workflow (After Model Retraining)
```bash
# 1. Retrain model
python scripts/phase3_build_horse_model.py

# 2. Calibrate model
python scripts/calibrate_model.py

# 3. Run diagnostics on recent predictions
python scripts/diagnose_model.py

# 4. Generate fresh predictions
python scripts/predict_todays_races.py
```

## Files and Locations

### Scripts
- `scripts/calibrate_model.py` - Model calibration
- `scripts/diagnose_model.py` - Prediction diagnostics
- `scripts/predict_todays_races.py` - Prediction generation (auto-diagnostics)

### Model Files
- `models/horse_win_predictor.json` - Base XGBoost model
- `models/horse_win_predictor_calibrated.pkl` - Calibrated model (used if exists)
- `models/calibration_metrics.json` - Calibration performance
- `models/calibration_plot.png` - Calibration visualization

### Data Files
- `data/processed/predictions_YYYY-MM-DD.csv` - Daily predictions
- `data/processed/diagnostics_YYYY-MM-DD.json` - Date-specific diagnostics
- `data/processed/model_diagnostics.json` - Latest diagnostics (for UI)

## Troubleshooting

### "Calibrated model not found"
- Normal if you haven't calibrated yet
- Run `python scripts/calibrate_model.py`

### "No diagnostics available"
- Generate at least one prediction file first
- Run `python scripts/predict_todays_races.py`

### High cold start percentage (>30%)
- May indicate missing pedigree data
- Check if sire features are being generated properly
- Review historical data completeness

### Poor calibration (Brier score >0.15)
- May need more calibration data
- Check if model is overtrained
- Consider retraining with more balanced data

## Next Steps

After implementing calibration and monitoring diagnostics:
1. Review calibration metrics to ensure improvement
2. Monitor diagnostics over multiple prediction runs
3. Consider implementing Betfair BSP integration (see IMMEDIATE_ACTION_PLAN.md)
4. Use diagnostics to identify areas for further feature engineering
