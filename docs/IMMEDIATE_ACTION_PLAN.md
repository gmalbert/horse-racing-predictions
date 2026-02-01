# Immediate Action Plan (Week 1-2)

Quick wins to implement immediately for meaningful improvement.

---

## Day 1-2: Diagnosis

### Step 1: Verify Current Model Performance

Run this analysis to understand where predictions fail:

```python
#!/usr/bin/env python3
"""scripts/diagnose_model.py - Understand where model fails."""

import pandas as pd
import numpy as np
from pathlib import Path

def load_predictions_with_results():
    """Load predictions and match with actual results."""
    predictions_dir = Path('data/processed')
    
    # Load all prediction files
    pred_files = list(predictions_dir.glob('predictions_*.csv'))
    all_preds = []
    
    for f in pred_files:
        df = pd.read_csv(f)
        df['pred_date'] = f.stem.replace('predictions_', '')
        all_preds.append(df)
    
    predictions = pd.concat(all_preds, ignore_index=True)
    
    # Load historical results
    results = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Match predictions to results
    # This requires fetching results for prediction dates
    return predictions, results

def analyze_failures(predictions, results):
    """Analyze where model predictions fail."""
    
    # Group by race
    race_summary = predictions.groupby(['pred_date', 'course', 'race_time']).apply(
        lambda g: pd.Series({
            'field_size': len(g),
            'top_pick': g.loc[g['win_probability'].idxmax(), 'horse'],
            'top_prob': g['win_probability'].max(),
            'prob_spread': g['win_probability'].std(),
        })
    ).reset_index()
    
    # Analyze characteristics of failures
    print("\n=== MODEL DIAGNOSIS ===\n")
    
    # 1. Probability distribution
    print("1. Win Probability Distribution of Top Picks:")
    print(predictions.groupby(['pred_date', 'course', 'race_time'])['win_probability']
          .max().describe())
    
    # 2. Cold start horses (no career data)
    cold_start = predictions[predictions['career_runs'] == 0]
    print(f"\n2. Cold Start Horses: {len(cold_start)} / {len(predictions)} ({len(cold_start)/len(predictions)*100:.1f}%)")
    
    # 3. Feature availability
    print("\n3. Feature Availability:")
    for col in ['career_win_rate', 'cd_win_rate', 'avg_last_3_pos', 'or_numeric']:
        null_pct = predictions[col].isna().mean() * 100
        zero_pct = (predictions[col] == 0).mean() * 100
        print(f"   {col}: {null_pct:.1f}% null, {zero_pct:.1f}% zero")
    
    # 4. Class distribution
    print("\n4. Predictions by Race Class:")
    print(predictions['race_class'].value_counts())
    
    return race_summary

if __name__ == '__main__':
    preds, results = load_predictions_with_results()
    summary = analyze_failures(preds, results)
```

### Step 2: Check Feature Values

```python
# Quick check of problematic features
df = pd.read_parquet('data/processed/race_scores.parquet')

# Features showing 0 importance
problem_features = ['has_blinkers', 'has_visor', 'first_time_blinkers', 
                   'gear_changed', 'is_maiden', 'is_handicap']

for feat in problem_features:
    if feat in df.columns:
        print(f"{feat}: {df[feat].value_counts().to_dict()}")
    else:
        print(f"{feat}: NOT IN DATAFRAME")
```

---

## Day 3-4: Fix Cold Start Problem

The most impactful quick fix: use pedigree data for horses with limited form.

### Build Sire Lookup Table

```python
#!/usr/bin/env python3
"""scripts/build_sire_lookup.py - Create sire performance lookup."""

import pandas as pd
from pathlib import Path

def build_sire_lookup():
    """Build sire statistics from historical data."""
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Check if sire column exists
    if 'sire' not in df.columns and 'sire_id' not in df.columns:
        print("ERROR: No sire data in race_scores.parquet")
        print("Need to add sire extraction from racecards")
        return None
    
    sire_col = 'sire' if 'sire' in df.columns else 'sire_id'
    
    # Calculate sire statistics
    df['won'] = (df['pos_clean'] == 1).astype(int)
    df['placed'] = (df['pos_clean'] <= 3).astype(int)
    
    sire_stats = df.groupby(sire_col).agg({
        'won': ['sum', 'count'],
        'placed': 'sum',
        'dist_f': 'mean',
        'is_turf': 'mean',
    }).reset_index()
    
    sire_stats.columns = ['sire', 'wins', 'runs', 'places', 'avg_dist', 'turf_pct']
    sire_stats['win_rate'] = sire_stats['wins'] / sire_stats['runs']
    sire_stats['place_rate'] = sire_stats['places'] / sire_stats['runs']
    
    # Filter to sires with 20+ runners
    sire_stats = sire_stats[sire_stats['runs'] >= 20]
    
    print(f"Built lookup for {len(sire_stats)} sires")
    sire_stats.to_csv('data/processed/lookups/sire_stats.csv', index=False)
    
    return sire_stats

if __name__ == '__main__':
    build_sire_lookup()
```

### Update Feature Engineering to Use Sire

```python
def add_pedigree_features(df, sire_lookup):
    """Add sire-based features for cold start horses."""
    
    # Merge sire stats
    df = df.merge(
        sire_lookup[['sire', 'win_rate', 'place_rate', 'avg_dist', 'turf_pct']],
        on='sire',
        how='left',
        suffixes=('', '_sire')
    )
    
    # For horses with no form, use sire stats
    df['career_win_rate_adj'] = np.where(
        df['career_runs'] < 3,
        df['win_rate_sire'].fillna(0.08),  # Fall back to sire, then to 8%
        df['career_win_rate']
    )
    
    df['career_place_rate_adj'] = np.where(
        df['career_runs'] < 3,
        df['place_rate_sire'].fillna(0.25),
        df['career_place_rate']
    )
    
    return df
```

---

## Day 5-6: Fix Probability Calibration

Current probabilities may be poorly calibrated. Add calibration layer.

### Calibration Script

```python
#!/usr/bin/env python3
"""scripts/calibrate_model.py - Add probability calibration."""

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import joblib

def calibrate_and_save():
    """Calibrate the existing model."""
    
    # Load model
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model('models/horse_win_predictor.json')
    
    # Load training data
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Use last 3 months for calibration (held out)
    df['date_dt'] = pd.to_datetime(df['date'])
    calib_start = df['date_dt'].max() - pd.DateOffset(months=3)
    
    calib_data = df[df['date_dt'] >= calib_start]
    
    # Get features and target
    feature_cols = [...]  # Load from models/feature_columns.txt
    X_calib = calib_data[feature_cols]
    y_calib = (calib_data['pos_clean'] == 1).astype(int)
    
    # Calibrate using isotonic regression
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_calib, y_calib)
    
    # Verify calibration
    y_pred_uncalib = model.predict_proba(X_calib)[:, 1]
    y_pred_calib = calibrated.predict_proba(X_calib)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before calibration
    prob_true, prob_pred = calibration_curve(y_calib, y_pred_uncalib, n_bins=10)
    axes[0].plot(prob_pred, prob_true, 's-', label='Uncalibrated')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set_title('Before Calibration')
    
    # After calibration  
    prob_true, prob_pred = calibration_curve(y_calib, y_pred_calib, n_bins=10)
    axes[1].plot(prob_pred, prob_true, 's-', label='Calibrated')
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set_title('After Calibration')
    
    plt.savefig('models/calibration_plot.png')
    
    # Save calibrated model
    joblib.dump(calibrated, 'models/horse_win_predictor_calibrated.pkl')
    print("Saved calibrated model to models/horse_win_predictor_calibrated.pkl")

if __name__ == '__main__':
    calibrate_and_save()
```

---

## Day 7-8: Add Trainer/Jockey Recent Form

Quick win: 14-day form for connections.

```python
def add_recent_form_features(df):
    """Add 14-day trainer and jockey form."""
    
    df = df.sort_values('date').copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    df['won'] = (df['pos_clean'] == 1).astype(int)
    
    # Create date-indexed dataframe for rolling
    df = df.set_index('date_dt')
    
    # 14-day trainer form
    trainer_form = df.groupby('trainer')['won'].rolling('14D').mean()
    trainer_form = trainer_form.groupby('trainer').shift(1)  # Avoid lookahead
    df['trainer_form_14d'] = trainer_form.values
    
    # 14-day jockey form
    jockey_form = df.groupby('jockey')['won'].rolling('14D').mean()
    jockey_form = jockey_form.groupby('jockey').shift(1)
    df['jockey_form_14d'] = jockey_form.values
    
    # Fill NA with career rates
    df['trainer_form_14d'] = df['trainer_form_14d'].fillna(df['trainer_win_rate'])
    df['jockey_form_14d'] = df['jockey_form_14d'].fillna(df['jockey_career_win_rate'])
    
    return df.reset_index()
```

---

## Day 9-10: Add Betfair SP Integration

Get free historical BSP data and integrate.

### Download BSP Data

1. Register at https://historicdata.betfair.com/
2. Download UK Horse Racing data (free tier)
3. Extract and save to `data/raw/betfair/`

### Integration Script

```python
#!/usr/bin/env python3
"""scripts/integrate_betfair_bsp.py - Add Betfair SP to predictions."""

import pandas as pd
from pathlib import Path
from fuzzywuzzy import fuzz, process

def load_betfair_data(date_str):
    """Load Betfair BSP data for a given date."""
    betfair_dir = Path('data/raw/betfair')
    file_path = betfair_dir / f'bsp_{date_str}.csv'
    
    if not file_path.exists():
        return None
    
    return pd.read_csv(file_path)

def match_horse_names(prediction_name, betfair_names, threshold=85):
    """Fuzzy match horse names between sources."""
    match, score = process.extractOne(prediction_name, betfair_names, scorer=fuzz.ratio)
    if score >= threshold:
        return match
    return None

def add_bsp_to_predictions(predictions_df, date_str):
    """Add BSP data to predictions."""
    bsp_data = load_betfair_data(date_str)
    
    if bsp_data is None:
        print(f"No BSP data for {date_str}")
        predictions_df['bsp'] = None
        predictions_df['market_prob'] = None
        return predictions_df
    
    # Match and merge
    predictions_df['betfair_horse'] = predictions_df['horse'].apply(
        lambda x: match_horse_names(x, bsp_data['horse'].tolist())
    )
    
    predictions_df = predictions_df.merge(
        bsp_data[['horse', 'bsp']],
        left_on='betfair_horse',
        right_on='horse',
        how='left',
        suffixes=('', '_bf')
    )
    
    # Calculate market probability
    predictions_df['market_prob'] = 1 / predictions_df['bsp']
    predictions_df['model_vs_market'] = predictions_df['win_probability'] - predictions_df['market_prob']
    
    return predictions_df
```

---

## Week 2: Retrain Model

After implementing the above, retrain with new features.

```bash
# Rebuild features
python scripts/phase3_build_horse_model.py --rebuild-features

# Train with new features
python scripts/phase3_build_horse_model.py --train

# Calibrate
python scripts/calibrate_model.py

# Validate
python scripts/validate_model.py --walk-forward --months 12
```

---

## Expected Improvements

| Change | Est. AUC Gain | Time to Implement |
|--------|---------------|-------------------|
| Fix cold start (pedigree) | +0.01-0.02 | 2 days |
| Probability calibration | +0.01 | 1 day |
| 14-day form | +0.01 | 1 day |
| Betfair BSP integration | +0.02-0.03 | 2 days |
| Model retrain | +0.01 | 1 day |
| **Total** | **+0.05-0.08** | **~10 days** |

This would move the model from ROC AUC 0.671 to approximately **0.72-0.75**.

---

## Success Criteria

After Week 2, validate:

1. **ROC AUC > 0.70** on held-out test data
2. **Top-1 Accuracy > 20%** (winner was top pick)
3. **Calibration Error < 0.05** (predicted prob matches actual)
4. **Cold Start Coverage < 20%** (fewer horses with no features)
5. **Positive or Breakeven ROI** on simulated level stakes

If these aren't met, proceed to deeper architectural changes in Weeks 3-4.
