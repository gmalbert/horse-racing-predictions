#!/usr/bin/env python3
"""
Ensemble Model: XGBoost + LightGBM + RandomForest stacked with Logistic Regression.

Architecture
------------
Level-0 (base models — trained on temporal train fold):
  - XGBoost classifier (gradient boosting)
  - LightGBM classifier (gradient boosting, faster)
  - ExtraTreesClassifier (random forest variant, fast + diverse)

Level-1 (meta-learner — trained on held-out stacking fold):
  - LogisticRegression with calibrated probability outputs
  - Combines base-model out-of-fold predictions

Probability calibration:
  - Platt scaling (sigmoid) applied to final ensemble output via
    sklearn CalibratedClassifierCV

Output
------
  models/ensemble_model.pkl            — serialised stacked ensemble
  models/ensemble_feature_columns.txt  — feature list used at fit time
  models/ensemble_calibration_metrics.json — Brier, ECE, log-loss vs XGBoost alone

Usage
-----
  python scripts/ensemble_model.py                    # train + save
  python scripts/ensemble_model.py --evaluate-only    # load saved model and print metrics
  python scripts/ensemble_model.py --feature-importance  # print top-20 features
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Base learners ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[!] XGBoost not found — base learner set will be limited")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[!] LightGBM not found — install via `pip install lightgbm`")

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


# ─────────────────────────────────────────────────────────────
# Data helpers (same as backtest_walk_forward.py)
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    candidates = [
        "race_scores_engineered.parquet",   # saved by phase3 — full 75-feature set
        "race_scores_or_context.parquet",
        "race_scores_going_pref.parquet",
        "race_scores_connections_v2.parquet",
        "race_scores_with_all_features_no_leakage.parquet",
        "race_scores.parquet",
    ]
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            print(f"[data] Loading {name}")
            df = pd.read_parquet(p)
            df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date_dt"])
            print(f"[data] {len(df):,} rows")
            return df
    raise FileNotFoundError("No historical parquet found")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    feat_file = MODEL_DIR / "feature_columns.txt"
    if feat_file.exists():
        preferred = [l.strip() for l in feat_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        available = [f for f in preferred if f in df.columns]
        print(f"[features] Using {len(available)}/{len(preferred)} features from feature_columns.txt")
        return available

    exclude = {
        "won", "placed", "pos_clean", "pos", "btn", "btn_lengths",
        "race_id", "horse_id", "jockey_id", "trainer_id",
        "date", "date_dt", "off", "course", "horse", "jockey", "trainer",
    }
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    print(f"[features] Fallback: {len(cols)} numeric features")
    return cols


def temporal_split(df: pd.DataFrame, test_size: float = 0.20):
    """80/20 strict temporal split (no shuffle)."""
    df = df.sort_values("date_dt")
    split_idx = int(len(df) * (1.0 - test_size))
    split_date = df["date_dt"].iloc[split_idx]
    train = df[df["date_dt"] < split_date].copy()
    test  = df[df["date_dt"] >= split_date].copy()
    print(f"[split] Train: {len(train):,}  ({train['date_dt'].min().date()} – {train['date_dt'].max().date()})")
    print(f"[split] Test:  {len(test):,}  ({test['date_dt'].min().date()} – {test['date_dt'].max().date()})")
    return train, test


# ─────────────────────────────────────────────────────────────
# Stacking CV for out-of-fold meta-features
# ─────────────────────────────────────────────────────────────

def make_oof_predictions(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    n_temporal_folds: int = 5,
) -> np.ndarray:
    """Generate out-of-fold predictions using temporal (non-shuffled) folds.

    Each fold trains on the earlier portion and predicts on the later portion,
    avoiding any lookahead contamination.

    Returns array of shape (n_samples,) with P(win) for each sample.
    """
    df = df.copy().reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date_dt"], errors="coerce")
    df = df.sort_values("date_dt").reset_index(drop=True)

    oof_preds = np.zeros(len(df))
    fold_size = len(df) // n_temporal_folds

    for fold_i in range(n_temporal_folds):
        val_start = fold_i * fold_size
        val_end   = val_start + fold_size if fold_i < n_temporal_folds - 1 else len(df)

        # Training uses all data BEFORE the validation window
        train_idx = list(range(0, val_start))
        val_idx   = list(range(val_start, val_end))

        if len(train_idx) < 100:
            # First fold: not enough data to train; use zero predictions
            continue

        est_clone = _clone_estimator(estimator)
        est_clone.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = est_clone.predict_proba(X[val_idx])[:, 1]

    return oof_preds


def _clone_estimator(est):
    """Deep-copy an estimator safely."""
    import copy
    return copy.deepcopy(est)


# ─────────────────────────────────────────────────────────────
# Build base estimators
# ─────────────────────────────────────────────────────────────

def build_base_estimators() -> list[tuple[str, object]]:
    """Return a list of (name, fitted-able estimator) tuples."""
    estimators = []

    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=10,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )
        estimators.append(("xgb", xgb))

    if HAS_LGBM:
        lgbm = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            n_jobs=-1,
            verbose=-1,
        )
        estimators.append(("lgbm", lgbm))

    # Extra Trees — always available, fast, uncorrelated with GB
    et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
    )
    estimators.append(("et", et))

    # Fallback if neither XGBoost nor LightGBM available
    if not HAS_XGBOOST and not HAS_LGBM:
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
        )
        estimators.append(("gb", gb))

    print(f"[ensemble] Base estimators: {[n for n,_ in estimators]}")
    return estimators


# ─────────────────────────────────────────────────────────────
# Build stacked ensemble
# ─────────────────────────────────────────────────────────────

def build_stacking_ensemble():
    """Return an sklearn StackingClassifier with calibrated meta-learner."""
    base = build_base_estimators()

    # Meta-learner: logistic regression with standard scaling
    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])

    stack = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=False,  # only pass OOF preds to meta, not raw features
        n_jobs=1,           # serial to preserve memory
        cv=3,               # 3-fold is sufficient for stacking meta-features
    )

    return stack


# ─────────────────────────────────────────────────────────────
# Calibration + metrics
# ─────────────────────────────────────────────────────────────

def calibrate(model, X_cal, y_cal):
    """Wrap a fitted model in Platt sigmoid calibration."""
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv="prefit",  # model already fitted
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate(model_name: str, model, X_test, y_test, df_test: pd.DataFrame) -> dict:
    """Evaluate a model and return a metrics dict."""
    preds = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, preds)
    ll   = log_loss(y_test, preds)
    bs   = brier_score_loss(y_test, preds)

    # Race-level top-1 / top-3
    df_eval = df_test.copy()
    if "race_id" not in df_eval.columns:
        df_eval["race_id"] = df_eval["date"].astype(str) + "_" + df_eval.get("course_clean", "").astype(str) + "_" + df_eval.get("off", "").astype(str)
    df_eval["pred_prob"] = preds

    grps = df_eval.groupby("race_id")
    top1 = [int(g.nlargest(1, "pred_prob")["won"].any()) for _, g in grps if g["won"].sum() > 0]
    top3 = [int(g.nlargest(3, "pred_prob")["won"].any()) for _, g in grps if g["won"].sum() > 0]

    return {
        "model": model_name,
        "auc": round(auc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(bs, 4),
        "top1_accuracy": round(np.mean(top1), 4) if top1 else 0.0,
        "top3_accuracy": round(np.mean(top3), 4) if top3 else 0.0,
        "n_races_evaluated": len(top1),
    }


# ─────────────────────────────────────────────────────────────
# Main train / evaluate
# ─────────────────────────────────────────────────────────────

def train_ensemble(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train ensemble, calibrate, save, and return calibration metrics."""
    # Target
    if "won" not in df.columns:
        df["won"] = (df["pos_clean"] == 1).astype(int)

    # Temporal split: 80% train, 20% test
    # Use first 70% for base-model training, next 10% for calibration, last 20% test
    df = df.sort_values("date_dt").reset_index(drop=True)
    n = len(df)
    train_end_idx = int(n * 0.70)
    cal_end_idx   = int(n * 0.80)

    train_df = df.iloc[:train_end_idx].copy()
    cal_df   = df.iloc[train_end_idx:cal_end_idx].copy()
    test_df  = df.iloc[cal_end_idx:].copy()

    # Filter to finishers
    for frame in [train_df, cal_df, test_df]:
        frame.dropna(subset=["pos_clean"], inplace=True)

    avail = [f for f in feature_cols if f in df.columns]
    X_train = train_df[avail].fillna(0).values
    y_train = train_df["won"].values
    X_cal   = cal_df[avail].fillna(0).values
    y_cal   = cal_df["won"].values
    X_test  = test_df[avail].fillna(0).values
    y_test  = test_df["won"].values

    print(f"\n[ensemble] Train:{len(X_train):,}  Cal:{len(X_cal):,}  Test:{len(X_test):,}")
    print(f"[ensemble] Features: {len(avail)}")

    # ── Build and fit stacking ensemble ──
    print("\n[ensemble] Fitting stacking ensemble…")
    stack = build_stacking_ensemble()
    stack.fit(X_train, y_train)
    print("[ensemble] Stacking fit complete.")

    # ── Calibrate on held-out calibration set ──
    print("[ensemble] Calibrating probabilities (Platt scaling)…")
    calibrated_ensemble = calibrate(stack, X_cal, y_cal)
    print("[ensemble] Calibration complete.")

    # ── Also train a plain XGBoost for comparison ──
    metrics_all = []
    if HAS_XGBOOST:
        xgb_solo = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=10, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, n_jobs=-1,
        )
        xgb_solo.fit(X_train, y_train)
        metrics_all.append(evaluate("xgb_solo", xgb_solo, X_test, y_test, test_df))

    metrics_all.append(evaluate("ensemble_calibrated", calibrated_ensemble, X_test, y_test, test_df))

    # ── Print comparison ──
    print(f"\n{'─'*55}")
    print(f" {'Model':<28} {'AUC':>6}  {'LL':>7}  {'BS':>7}  Top-1  Top-3")
    print(f"{'─'*55}")
    for m in metrics_all:
        print(
            f" {m['model']:<28} {m['auc']:>6.4f}  {m['log_loss']:>7.4f}  "
            f"{m['brier_score']:>7.4f}  {m['top1_accuracy']*100:>5.1f}%  {m['top3_accuracy']*100:>5.1f}%"
        )
    print(f"{'─'*55}")

    # ── Save model ──
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "ensemble_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": calibrated_ensemble, "feature_cols": avail}, f)
    print(f"\n[✓] Ensemble model saved → {model_path}")

    # Save feature columns
    feat_col_path = MODEL_DIR / "ensemble_feature_columns.txt"
    feat_col_path.write_text("\n".join(avail))
    print(f"[✓] Feature list saved  → {feat_col_path}")

    # Save calibration metrics
    cal_metrics_path = MODEL_DIR / "ensemble_calibration_metrics.json"
    with open(cal_metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"[✓] Calibration metrics → {cal_metrics_path}")

    return {"metrics": metrics_all, "feature_cols": avail}


def load_ensemble():
    """Load a previously saved ensemble model."""
    model_path = MODEL_DIR / "ensemble_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved ensemble at {model_path} — run without --evaluate-only first")
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    print(f"[✓] Loaded ensemble from {model_path}")
    return obj["model"], obj["feature_cols"]


def evaluate_saved(df: pd.DataFrame) -> None:
    """Load saved ensemble and evaluate on the most recent 20% of data."""
    model, feature_cols = load_ensemble()

    if "won" not in df.columns:
        df["won"] = (df["pos_clean"] == 1).astype(int)

    df = df.sort_values("date_dt").dropna(subset=["pos_clean"]).reset_index(drop=True)
    test_df = df.iloc[int(len(df) * 0.80):].copy()

    avail = [f for f in feature_cols if f in test_df.columns]
    X_test = test_df[avail].fillna(0).values
    y_test = test_df["won"].values

    m = evaluate("ensemble_calibrated_saved", model, X_test, y_test, test_df)
    print(json.dumps(m, indent=2))


def print_feature_importance(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Train a single XGBoost and print top-20 feature importances."""
    if not HAS_XGBOOST:
        print("[!] XGBoost not available for feature importance")
        return

    if "won" not in df.columns:
        df["won"] = (df["pos_clean"] == 1).astype(int)

    df = df.sort_values("date_dt").dropna(subset=["pos_clean"]).reset_index(drop=True)
    train_df = df.iloc[:int(len(df) * 0.80)].copy()
    avail = [f for f in feature_cols if f in train_df.columns]

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
        eval_metric="logloss", use_label_encoder=False, verbosity=0, n_jobs=-1,
    )
    xgb.fit(train_df[avail].fillna(0), train_df["won"])

    imp = pd.Series(xgb.feature_importances_, index=avail).sort_values(ascending=False)
    print("\nTop-20 Feature Importances (XGBoost gain):")
    for feat, score in imp.head(20).items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<40} {score:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train / evaluate stacked ensemble model")
    p.add_argument("--evaluate-only",      action="store_true", help="Load saved model and evaluate (no retraining)")
    p.add_argument("--feature-importance", action="store_true", help="Print feature importances then exit")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data()
    feature_cols = get_feature_cols(df)

    if args.evaluate_only:
        evaluate_saved(df)
        return

    if args.feature_importance:
        print_feature_importance(df, feature_cols)
        return

    # Default: train
    train_ensemble(df, feature_cols)


if __name__ == "__main__":
    main()
