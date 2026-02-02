# Model Architecture Improvements

The current XGBoost classifier has fundamental limitations. This document outlines architectural improvements.

---

## Current Model Issues

### 1. Binary Classification Problem
```python
# Current approach
target = (df['pos_clean'] == 1).astype(int)  # Win = 1, Lose = 0
model = XGBClassifier().fit(X, target)
```

**Problems:**
- Treats all non-wins equally (2nd = 10th)
- Loses margin information (won by 10 lengths vs won by nose)
- Creates severe class imbalance (~10% wins in typical field)

### 2. Race Independence Assumption
- Current model treats each horse independently
- Ignores that horses compete AGAINST each other
- A horse's chance depends on who else is running

### 3. Single Model for All Races
- Same model for maidens, handicaps, Group races
- These race types have fundamentally different dynamics

---

## Evaluation Metrics

### Current Model: Binary Classification
```python
# Primary metric: AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)
```

### Learning-to-Rank Models
```python
# Primary metric: NDCG (Normalized Discounted Cumulative Gain)
from sklearn.metrics import ndcg_score

# For each race, compute NDCG
def compute_race_ndcg(y_true_race, y_pred_race):
    """Compute NDCG for a single race"""
    # y_true_race: actual finishing positions (1=1st, 2=2nd, etc.)
    # y_pred_race: predicted scores (higher = better predicted rank)
    
    # Convert positions to relevance (1st place = highest relevance)
    relevance = 1.0 / y_true_race  # 1st=1.0, 2nd=0.5, 3rd=0.33, etc.
    
    return ndcg_score([relevance], [y_pred_race], k=len(y_true_race))

# Secondary metric: Top-N Accuracy
def top_n_accuracy(y_true, y_pred_scores, n=3):
    """Fraction of races where top N predictions include actual winner"""
    correct = 0
    total = 0
    
    for race_id in y_true['race_id'].unique():
        race_mask = y_true['race_id'] == race_id
        race_true = y_true[race_mask]
        race_pred = y_pred_scores[race_mask]
        
        # Get predicted top N
        top_n_indices = np.argsort(race_pred)[-n:]
        actual_winner = (race_true == 1).idxmax()
        
        if actual_winner in top_n_indices:
            correct += 1
        total += 1
    
    return correct / total
```

### Position Prediction (MAE)
```python
# Only if predicting exact finishing positions
from sklearn.metrics import mean_absolute_error

# Convert predictions to position estimates
def scores_to_positions(scores):
    """Convert prediction scores to estimated positions"""
    return pd.Series(scores).rank(ascending=False).astype(int)

mae = mean_absolute_error(actual_positions, predicted_positions)
```

### Recommended Metrics by Model Type

| Model Type | Primary Metric | Secondary Metrics | MAE Appropriate? |
|------------|----------------|-------------------|------------------|
| **Binary Classifier** | AUC | Precision@TopK, Top-N Acc | ❌ Not applicable |
| **Learning-to-Rank** | NDCG@5 | Top-3 Accuracy, Mean Reciprocal Rank | ❌ Predicts ranking, not exact positions |
| **Position Predictor** | MAE | Spearman's ρ, Kendall's τ | ✅ If predicting exact positions 1,2,3... |
| **Ensemble** | AUC + NDCG | Calibration Error, Brier Score | ❌ Mixed objectives |

**Bottom Line:** MAE would only be appropriate if we switch to predicting exact finishing positions (1st, 2nd, 3rd, etc.) rather than win probabilities or rankings.

---

## Proposed Architecture Changes

### 1. Learning-to-Rank (LambdaMART)

Instead of predicting win probability independently, predict relative ranking.

```python
from lightgbm import LGBMRanker

# Group by race, rank horses within each race
train_df['rank_target'] = train_df.groupby('race_id')['pos_clean'].rank()

# LambdaMART ranking model
ranker = LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20
)

ranker.fit(
    X_train, 
    y_train_rank,
    group=train_groups,  # Number of horses per race
    eval_set=[(X_val, y_val_rank)],
    eval_group=[val_groups]
)
```

**Benefits:**
- Optimizes for ranking horses, not just win/lose
- Natural handling of race context
- Better calibrated probabilities

### 2. Conditional Logit Model (Specialized for Races)

Classic econometric model for choice problems:

```python
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

# Conditional logit: probability of choosing winner from field
# P(horse i wins | race r) = exp(X_i * β) / Σ exp(X_j * β) for j in race r

model = ConditionalLogit(
    endog=df['won'],
    exog=df[feature_cols],
    groups=df['race_id']
)
result = model.fit()
```

**Benefits:**
- Theoretically grounded for horse racing
- Probabilities sum to 1 within race (guaranteed)
- Handles varying field sizes naturally

### 3. Ensemble Approach

Combine multiple models for robustness:

```python
class RaceEnsemble:
    def __init__(self):
        self.models = {
            'xgb': XGBClassifier(n_estimators=500, max_depth=6),
            'lgbm': LGBMClassifier(n_estimators=500, num_leaves=31),
            'catboost': CatBoostClassifier(iterations=500, depth=6),
            'ranker': LGBMRanker(objective='lambdarank'),
            'nn': create_neural_net()  # See below
        }
        self.weights = None  # Learned via validation
    
    def fit(self, X, y, groups=None):
        for name, model in self.models.items():
            if name == 'ranker':
                model.fit(X, y.rank(), group=groups)
            else:
                model.fit(X, y)
        
        # Learn optimal weights via held-out validation
        self.weights = self._optimize_weights(X_val, y_val)
    
    def predict_proba(self, X, race_ids=None):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        # Weighted average
        combined = sum(
            self.weights[name] * predictions[name] 
            for name in predictions
        )
        
        # Normalize within race
        if race_ids is not None:
            combined = self._normalize_by_race(combined, race_ids)
        
        return combined
```

### 4. Neural Network Architecture

For capturing complex interactions:

```python
import torch
import torch.nn as nn

class RacePredictor(nn.Module):
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        
        # Horse feature encoder
        self.horse_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Race context (attention over horses in race)
        self.race_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Final prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, horse_features, race_mask):
        # Encode each horse
        horse_embed = self.horse_encoder(horse_features)
        
        # Attention: each horse attends to others in race
        # This captures relative quality
        race_context, _ = self.race_attention(
            horse_embed, horse_embed, horse_embed,
            key_padding_mask=race_mask
        )
        
        # Combine horse features with race context
        combined = torch.cat([horse_embed, race_context], dim=-1)
        
        return self.predictor(combined)
```

### 5. Specialized Models by Race Type

Different models for different contexts:

```python
class SpecializedEnsemble:
    def __init__(self):
        self.models = {
            'maiden_2yo': self._create_maiden_model(),      # More pedigree weight
            'maiden_3yo': self._create_maiden_model(),
            'handicap': self._create_handicap_model(),      # More weight/OR focus
            'conditions': self._create_conditions_model(),
            'group_listed': self._create_class_model(),     # Class form emphasis
        }
    
    def _create_maiden_model(self):
        """Maiden model emphasizes pedigree, price movement."""
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            feature_weights={
                'sire_win_rate': 2.0,
                'sire_distance_match': 2.0,
                'market_win_prob': 1.5,
                'career_runs': 0.5  # Less weight on limited form
            }
        )
    
    def _create_handicap_model(self):
        """Handicap model emphasizes weight, OR, trends."""
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            feature_weights={
                'weight_vs_avg': 2.0,
                'or_vs_field': 2.0,
                'or_trend_3': 1.5,
                'class_step': 1.5
            }
        )
    
    def predict(self, X, race_type):
        return self.models[race_type].predict_proba(X)[:, 1]
```

---

## Training Improvements

### 1. Proper Temporal Validation

**Never use random train/test split for time series!**

```python
def temporal_train_test_split(df, test_months=3):
    """
    Split data temporally: train on past, test on future.
    """
    df = df.sort_values('date')
    
    cutoff_date = df['date'].max() - pd.DateOffset(months=test_months)
    
    train = df[df['date'] < cutoff_date]
    test = df[df['date'] >= cutoff_date]
    
    print(f"Train: {train['date'].min()} to {train['date'].max()}")
    print(f"Test: {test['date'].min()} to {test['date'].max()}")
    
    return train, test
```

### 2. Walk-Forward Validation

```python
def walk_forward_validation(df, n_splits=12, train_months=24, test_months=1):
    """
    Rolling window validation that mimics real-world usage.
    """
    df = df.sort_values('date')
    results = []
    
    for split in range(n_splits):
        test_end = df['date'].max() - pd.DateOffset(months=split)
        test_start = test_end - pd.DateOffset(months=test_months)
        train_end = test_start
        train_start = train_end - pd.DateOffset(months=train_months)
        
        train = df[(df['date'] >= train_start) & (df['date'] < train_end)]
        test = df[(df['date'] >= test_start) & (df['date'] < test_end)]
        
        # Train and evaluate
        model.fit(train[features], train[target])
        preds = model.predict_proba(test[features])[:, 1]
        auc = roc_auc_score(test[target], preds)
        
        results.append({
            'split': split,
            'train_period': f"{train_start} to {train_end}",
            'test_period': f"{test_start} to {test_end}",
            'auc': auc
        })
    
    return pd.DataFrame(results)
```

### 3. Class Imbalance Handling

```python
# Option 1: Class weights
model = XGBClassifier(
    scale_pos_weight=len(y[y==0]) / len(y[y==1])  # ~9:1 ratio
)

# Option 2: Focal loss (for gradient boosting)
def focal_loss(y_true, y_pred, gamma=2.0):
    """Focus learning on hard examples."""
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    return -((1 - pt) ** gamma) * np.log(pt + 1e-8)

# Option 3: SMOTE oversampling (with caution)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 4. Probability Calibration

Raw model probabilities are often poorly calibrated:

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probabilities using isotonic regression
calibrated_model = CalibratedClassifierCV(
    model, 
    method='isotonic',  # or 'sigmoid'
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Verify calibration
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)
```

---

## Hyperparameter Tuning

### Optuna for Bayesian Optimization

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = XGBClassifier(**params)
    
    # Use temporal CV, not random
    scores = []
    for train_idx, val_idx in temporal_cv.split(X, groups=dates):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], preds))
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## Implementation Priority

| Improvement | Est. AUC Gain | Effort | Priority |
|-------------|---------------|--------|----------|
| Temporal validation | +0.02 (realistic) | Low | 🔴 Week 1 |
| Probability calibration | +0.01 | Low | 🔴 Week 1 |
| LambdaMART ranker | +0.02-0.03 | Medium | 🟠 Week 2 |
| Ensemble (3 models) | +0.01-0.02 | Medium | 🟠 Week 2-3 |
| Specialized models | +0.01-0.02 | Medium | 🟡 Week 4 |
| Neural attention | +0.02-0.03 | High | 🟢 Month 2 |
| Hyperparameter tuning | +0.01 | Medium | 🟡 Week 3 |

**Note:** Temporal validation may actually DECREASE measured AUC initially, but gives a more honest assessment of real-world performance.
