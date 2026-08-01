# Horse Racing Predictions — Model Deep Dive & Enhancement Recommendations

> Generated: 2026-07-31

---

## Current Model: XGBoost Classifier (AUC ~0.69)

The current Phase 3 model achieves AUC ~0.6892 with 75 engineered features.
This document outlines the path to AUC 0.75+.

---

## Critical Feature Gaps

### 1. Speed Figure Integration (Highest Priority)

The single biggest gap: the model lacks meaningful pace figures.
Beyer Speed Figures or TimeformUS figures would immediately improve accuracy:

```python
SPEED_FIGURE_FEATURES = [
    "avg_beyer_l3",       # Average Beyer figure last 3 races
    "best_beyer_l5",      # Best Beyer figure last 5 races  
    "beyer_trend",        # Improving/declining trajectory
    "beyer_vs_field_avg", # Horse's Beyer vs today's field average
    "speed_fig_consistency", # Standard deviation (low = consistent)
    "class_adjusted_beyer",  # Beyer adjusted for class level
]
```

### 2. Workout Pattern Features

Morning workout times are strong leading indicators of race readiness:

```python
WORKOUT_FEATURES = [
    "days_since_last_workout",
    "workout_times_l5_avg",     # Average workout time (faster = better)
    "workout_rank_among_peers", # How works rank vs stablemates
    "bullet_workout_flag",      # Flag for exceptional workout (fastest at distance)
    "workout_pattern_flag",     # Increasing/decreasing workout intensity
]
```

### 3. Pace Bias Features (Critical for Exactas/Trifectas)

```python
PACE_BIAS_FEATURES = [
    "front_runner_flag",    # Horse likes to lead
    "closers_flag",         # Horse closes late
    "pace_scenario_today",  # Predicted pace of today's race (fast/slow/contested)
    "pace_advantage",       # Does horse's running style suit today's pace scenario
    "speed_point_rating",   # DRF Speed Points (0-8 scale)
]
```

---

## Model Architecture Improvements

### 1. Calibrated Win Probability Output

Current probabilities are not well-calibrated. Essential for Kelly criterion:

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import numpy as np

def calibrate_win_model(model, X_val: pd.DataFrame, y_val: pd.Series):
    """Apply Platt scaling calibration to win probability outputs."""
    calibrated = CalibratedClassifierCV(
        model, method="isotonic", cv="prefit"
    )
    calibrated.fit(X_val, y_val)
    
    # Verify calibration
    frac_pos, mean_pred = calibration_curve(y_val, 
                                             calibrated.predict_proba(X_val)[:, 1],
                                             n_bins=10)
    ece = np.mean(np.abs(frac_pos - mean_pred))
    print(f"Calibrated ECE: {ece:.4f} (target < 0.05)")
    return calibrated
```

### 2. Race-Level Multi-Output Model

Instead of individual horse win probability, model the entire field simultaneously.
Softmax ensures probabilities sum to 1:

```python
import numpy as np

def softmax_field_probabilities(raw_probs: np.ndarray) -> np.ndarray:
    """Convert raw win probabilities to proper race-level distribution."""
    # Apply temperature scaling (T=0.8 increases confidence separation)
    T = 0.8
    scaled = np.exp(raw_probs / T)
    return scaled / scaled.sum()

def predict_full_field(X_field: pd.DataFrame, model) -> pd.DataFrame:
    """Predict win probability for entire field, normalized to sum to 1."""
    raw_probs = model.predict_proba(X_field)[:, 1]
    calibrated = softmax_field_probabilities(raw_probs)
    return pd.DataFrame({
        "horse": X_field["horse"],
        "win_prob_raw": raw_probs,
        "win_prob_calibrated": calibrated,
        "implied_odds": 1 / calibrated,
    })
```

### 3. Ensemble with Gradient Boosting + Neural Network

```python
import torch, torch.nn as nn
import numpy as np

class HorseRacingNN(nn.Module):
    """Lightweight neural network for horse racing feature combination."""
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def blend_predictions(xgb_prob: np.ndarray, nn_prob: np.ndarray,
                        w_xgb: float = 0.65, w_nn: float = 0.35) -> np.ndarray:
    """Blend XGBoost and NN predictions."""
    return w_xgb * xgb_prob + w_nn * nn_prob
```

---

## Temporal Validation Fix

**CRITICAL**: The model must use walk-forward validation, not random splits:

```python
from sklearn.model_selection import TimeSeriesSplit

def evaluate_with_proper_temporal_cv(
    df: pd.DataFrame, features: list, target: str = "won"
) -> dict:
    """
    Walk-forward validation: train on past races, validate on future races.
    Never mix past and future data in train/val splits.
    """
    df = df.sort_values("race_date")
    X = df[features].fillna(df[features].median())
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=6, gap=30)  # 30-day gap prevents leakage
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        train_start = df.iloc[train_idx]["race_date"].min()
        val_start = df.iloc[val_idx]["race_date"].min()
        print(f"Train: {train_start.date()} | Val: {val_start.date()}")
        
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X.iloc[val_idx])[:, 1]
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y.iloc[val_idx], preds)
        aucs.append(auc)
        print(f"  AUC: {auc:.4f}")

    return {"mean_auc": np.mean(aucs), "std_auc": np.std(aucs), "all_aucs": aucs}
```

---

## Recommended Feature Removal (Data Leakage Candidates)

```python
# These features may contain race outcome information
SUSPECT_FEATURES = [
    "finishing_position",          # OBVIOUS LEAKAGE - target itself
    "prize_money_earned",          # Depends on finishing position
    "points_earned",               # Depends on result
    "official_rating_after",       # Updated based on result
]

# Verify these use shift(1) before using
VERIFY_SHIFT_FEATURES = [
    "career_wins",          # Should NOT include today's race
    "win_rate",             # Should NOT include today's race
    "form_last_5",          # Must be computed before today
]
```

---

## Expected Performance Improvement Roadmap

| Enhancement | AUC Improvement | Time |
|-------------|-----------------|------|
| Speed figures (Beyer) | +0.03-0.04 | 1 week |
| Temporal CV fix | Correctness | 2 days |
| Probability calibration | Better EV | 3 days |
| Workout data | +0.01-0.02 | 1 week |
| Pace scenario model | +0.02-0.03 | 1 week |
| NN blend | +0.01-0.02 | 2 weeks |
| **Total Projected AUC** | **~0.75-0.77** | ~6 weeks |
