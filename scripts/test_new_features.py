import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_FILE = Path('data/processed/race_scores.parquet')


def test_feature_value(df, base_features, new_feature):
    X_base = df[base_features].fillna(0)
    X_new = df[base_features + [new_feature]].fillna(0)
    y = df['won']

    # Split
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )
    X_new_train, X_new_test, _, _ = train_test_split(
        X_new, y, test_size=0.2, random_state=42
    )

    # Train baseline
    model_base = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model_base.fit(X_base_train, y_train)
    auc_base = roc_auc_score(y_test, model_base.predict_proba(X_base_test)[:, 1])

    # Train with new feature
    model_new = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model_new.fit(X_new_train, y_train)
    auc_new = roc_auc_score(y_test, model_new.predict_proba(X_new_test)[:, 1])

    improvement = auc_new - auc_base

    print(f"Feature: {new_feature}")
    print(f"  Baseline AUC: {auc_base:.4f}")
    print(f"  With feature: {auc_new:.4f}")
    print(f"  Improvement:  {improvement:+.4f} {'✓' if improvement > 0 else '✗'}")

    return improvement


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
