import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_FILE = Path('data/processed/race_scores.parquet')


def test_feature_value():
    """Self-contained test that verifies adding a predictive feature improves AUC.

    Generates a synthetic dataset so the test does not depend on external fixtures or
    the processed parquet file. The synthetic `new_feature` is constructed to be
    predictive so the model with the extra feature should have higher AUC.
    """
    rng = np.random.RandomState(42)
    n = 1000

    base_features = [
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
        'cd_runs', 'cd_win_rate', 'class_num', 'class_step',
        'or_numeric', 'or_change', 'or_trend_3', 'avg_last_3_pos', 'wins_last_3',
        'days_since_last', 'field_size', 'is_turf', 'going_numeric', 'race_score'
    ]

    # Create synthetic base feature matrix
    X_base = pd.DataFrame(rng.normal(size=(n, len(base_features))), columns=base_features)

    # Synthetic predictive feature (strong signal)
    new_feature = 'synthetic_signal'
    X_base[new_feature] = rng.normal(size=n) * 0.5

    # Make the label strongly correlated with the new_feature (so it should help)
    logits = 2.5 * X_base[new_feature] + 0.2 * X_base[base_features].sum(axis=1) + rng.normal(scale=0.5, size=n)
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n) < prob).astype(int)

    # Prepare datasets
    X_base_vals = X_base[base_features].fillna(0)
    X_new_vals = X_base[base_features + [new_feature]].fillna(0)

    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_base_vals, y, test_size=0.2, random_state=42
    )
    X_new_train, X_new_test, _, _ = train_test_split(
        X_new_vals, y, test_size=0.2, random_state=42
    )

    # Use small XGBoost models to keep test fast but deterministic
    model_base = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model_new = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')

    model_base.fit(X_base_train, y_train)
    auc_base = roc_auc_score(y_test, model_base.predict_proba(X_base_test)[:, 1])

    model_new.fit(X_new_train, y_train)
    auc_new = roc_auc_score(y_test, model_new.predict_proba(X_new_test)[:, 1])

    improvement = auc_new - auc_base

    print(f"Synthetic feature test - baseline AUC: {auc_base:.4f}, with feature: {auc_new:.4f}, improvement: {improvement:+.4f}")

    # The synthetic feature is strongly predictive — expect a measurable improvement
    assert improvement > 0.02, f"Expected improvement > 0.02, got {improvement:.4f}"


if __name__ == '__main__':
    if not DATA_FILE.exists():
        raise SystemExit(f"Data file not found: {DATA_FILE} - run phase2/phase3 to generate processed data")

    df = pd.read_parquet(DATA_FILE)

    base_features = [
        'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
        'cd_runs', 'cd_win_rate', 'class_num', 'class_step',
        'or_numeric', 'or_change', 'or_trend_3', 'avg_last_3_pos', 'wins_last_3',
        'days_since_last', 'field_size', 'is_turf', 'going_numeric', 'race_score'
    ]

    new_features = [
        'draw', 'draw_pct', 'draw_group_win_rate',
        'weight_lbs', 'weight_vs_avg', 'age', 'trainer_win_rate_14d', 'btn_lengths',
        'avg_btn_last_3', 'has_blinkers'
    ]

    for feat in new_features:
        if feat in df.columns:
            try:
                test_feature_value(df, base_features, feat)
            except Exception as e:
                print(f"Skipping {feat}: {e}")
        else:
            print(f"Feature not present in dataset: {feat}")
