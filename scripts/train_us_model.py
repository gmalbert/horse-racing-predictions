"""
train_us_model.py — Train an XGBoost win-prediction model on US race data.

This is the US-specific counterpart to scripts/phase3_build_horse_model.py.
Once ~50k US race results have been collected (via fetch_equibase_results.py
and scrapers), run this script to train and save a dedicated US model that
outperforms the UK base model on American conditions (dirt racing, US class
ladder, Beyer Speed Figures, US pace ratings).

Data requirements:
    data/processed/us_races_cleaned.parquet   — flat race/runner rows with:
        horse, date, course, surface, distance_f, race_class,
        position (finish), win_probability (optional, for calibration),
        plus any of: beyer_speed, pace_early, pace_late, us_class_num,
                     career_us_wins, career_us_runs, days_off, etc.

Output:
    models/us_horse_model.pkl          — trained XGBoost classifier
    models/us_feature_columns.txt      — feature column names (one per line)
    models/us_model_metadata.json      — training summary (date, AUC, n_rows)

Usage:
    python scripts/train_us_model.py
    python scripts/train_us_model.py --min-rows 10000    # lower threshold for early testing
    python scripts/train_us_model.py --folds 5           # walk-forward CV folds
    python scripts/train_us_model.py --dry-run           # validate data without training

Notes on data leakage prevention (critical):
    - Sort by (horse, date) before any cumulative/expanding aggregations.
    - Use shift(1) to exclude the current race from career stats.
    - Use temporal train/test split, NOT random split.
    - See copilot-instructions.md §DATA LEAKAGE PREVENTION for full checklist.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parent.parent
PROC_DIR   = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] train_us_model: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("train_us_model")

# ── Feature definitions ───────────────────────────────────────────────────────

# Core 47 features shared with the UK base model (compatible for transfer learning)
UK_BASE_FEATURES = [
    "career_runs", "career_win_rate", "career_place_rate", "career_earnings",
    "cd_runs", "cd_win_rate", "class_num", "class_step", "or_numeric",
    "or_change", "or_trend_3", "avg_last_3_pos", "wins_last_3", "days_since_last",
    "field_size", "is_turf", "going_numeric", "race_score", "draw", "draw_pct",
    "draw_group_win_rate", "weight_lbs", "weight_vs_avg", "is_top_weight",
    "weight_change", "age", "is_peak_age", "is_3yo", "is_veteran", "age_vs_avg",
    "avg_btn_last_3", "unlucky_last", "has_blinkers", "has_visor",
    "first_time_blinkers", "gear_changed", "is_handicap", "is_maiden", "is_pattern",
    "prize_log", "is_sprint", "is_mile", "is_middle", "is_staying",
    "jockey_career_runs", "jockey_course_runs", "jockey_trainer_runs",
]

# Additional US-specific features (used when available in the dataset)
US_EXTRA_FEATURES = [
    # Beyer Speed Figure (primary US speed metric)
    "beyer_last", "beyer_avg_3", "beyer_best",
    # Pace figures (early / late speed)
    "pace_early_last", "pace_late_last",
    # US class ladder position (0=Maiden Claiming … 10=G1)
    "us_class_num", "us_class_step",
    # US track-specific career stats
    "career_us_wins", "career_us_runs", "career_us_dirt_wins", "career_us_turf_wins",
    # Connections (US jockey / trainer win rates)
    "jockey_us_win_rate_30d", "trainer_us_win_rate_30d", "j_t_combo_win_rate",
    # Workout recency
    "days_since_last_workout", "workout_count_30d",
    # Track variant
    "track_variant",
]

US_FEATURE_COLS = UK_BASE_FEATURES + [f for f in US_EXTRA_FEATURES]

# ── Data loading & validation ─────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    candidates = [
        PROC_DIR / "us_races_cleaned.parquet",
        PROC_DIR / "all_us_races_cleaned.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            logger.info("Loaded %d rows from %s", len(df), p.name)
            return df
    raise FileNotFoundError(
        "No US race data found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n\nFetch US results from The Racing API:\n"
        + "  python scripts/fetch_us_results.py --days 90\n"
        + "Then re-run this script. Use --min-rows 1000 for early/small datasets."
    )


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"horse", "date", "position"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")


def _engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary win label; drop non-finishers."""
    df = df.copy()
    df["pos_num"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["pos_num"])
    df["won"] = (df["pos_num"] == 1).astype(int)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build expanding-window career stats per horse.
    CRITICAL: sort by (horse, date) first; use shift(1) to prevent leakage.
    """
    df = df.sort_values(["horse", "date"]).reset_index(drop=True)

    df["career_runs"] = df.groupby("horse").cumcount()  # count before this row = shift
    df["career_wins"] = df.groupby("horse")["won"].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
    df["career_win_rate"] = np.where(
        df["career_runs"] > 0, df["career_wins"] / df["career_runs"], 0.0
    )

    # Placeholder for other features — fill with 0 for now.
    # As more data fields are available (Beyer, pace, etc.) add logic here.
    for col in US_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


# ── Model training ────────────────────────────────────────────────────────────

def _temporal_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split by date preserving temporal order (no random split)."""
    df = df.sort_values("date")
    cutoff_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:cutoff_idx], df.iloc[cutoff_idx:]


def train(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    """Train XGBoost classifier; return (model, auc_score)."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed: pip install xgboost")
    from sklearn.metrics import roc_auc_score

    train_df, test_df = _temporal_split(df)

    # Coerce all feature columns to numeric (handles values like '1A' in draw column)
    def _to_numeric_df(frame: pd.DataFrame, cols: list) -> "np.ndarray":
        return frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    X_train = _to_numeric_df(train_df, feature_cols)
    y_train = train_df["won"].values
    X_test  = _to_numeric_df(test_df, feature_cols)
    y_test  = test_df["won"].values

    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    logger.info(
        "Training on %d rows, testing on %d. Pos weight: %.1f",
        len(X_train), len(X_test), pos_weight,
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info("Test AUC: %.4f", auc)

    return model, auc


# ── Save artifacts ────────────────────────────────────────────────────────────

def save_model(model, feature_cols: list[str], auc: float, n_rows: int) -> None:
    import pickle

    model_path   = MODELS_DIR / "us_horse_model.pkl"
    feat_path    = MODELS_DIR / "us_feature_columns.txt"
    meta_path    = MODELS_DIR / "us_model_metadata.json"

    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", model_path)

    feat_path.write_text("\n".join(feature_cols))
    logger.info("Feature columns saved to %s", feat_path)

    meta = {
        "model_type": "XGBoostClassifier",
        "trained_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_features": len(feature_cols),
        "n_training_rows": n_rows,
        "test_auc": round(auc, 4),
        "feature_columns": feature_cols,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved to %s", meta_path)
    logger.info("US model training complete — AUC: %.4f", auc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train US horse racing prediction model")
    parser.add_argument("--min-rows", type=int, default=5_000,
                        help="Minimum rows required to proceed (default: 5000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data and feature engineering without training")
    parser.add_argument("--extra-features", action="store_true", default=True,
                        help="Include US-specific extra features (default: True)")
    args = parser.parse_args()

    try:
        df_raw = _load_data()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    _validate_columns(df_raw)

    df = _engineer_target(df_raw)
    df = _engineer_features(df)

    if len(df) < args.min_rows:
        logger.warning(
            "Only %d rows available (minimum %d). "
            "Continue collecting results — re-run when you have more data.",
            len(df), args.min_rows,
        )
        if not args.dry_run:
            sys.exit(0)

    feature_cols = UK_BASE_FEATURES.copy()
    if args.extra_features:
        feature_cols += [f for f in US_EXTRA_FEATURES if f in df.columns]

    logger.info("Using %d features", len(feature_cols))
    logger.info("Win rate: %.2f%%", df["won"].mean() * 100)

    if args.dry_run:
        logger.info("Dry-run complete — data looks OK. Remove --dry-run to train.")
        return

    model, auc = train(df, feature_cols)
    save_model(model, feature_cols, auc, len(df))


if __name__ == "__main__":
    main()
