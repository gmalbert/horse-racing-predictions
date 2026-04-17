#!/usr/bin/env python3
"""
Walk-Forward Backtesting Engine for Horse Racing Win Predictor.

Implements:
  1. Strict temporal walk-forward cross-validation (12 folds × 1-month test windows)
  2. Level-stakes betting simulation
  3. Value-betting simulation with Kelly criterion staking
  4. Drawdown analysis (max drawdown, recovery time, Sharpe ratio)

Outputs:
  data/processed/backtest_results.json   — full fold-by-fold metrics
  data/processed/backtest_summary.csv    — per-fold summary table
  data/processed/backtest_equity.csv     — daily P&L curves for equity curve plot

Usage:
  python scripts/backtest_walk_forward.py
  python scripts/backtest_walk_forward.py --train-months 24 --test-months 1 --folds 12
  python scripts/backtest_walk_forward.py --train-months 18 --test-months 2 --folds 6 --plot
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ML
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False
    print("[!] XGBoost not installed — falling back to RandomForest")

from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load best available historical dataset."""
    candidates = [
        "race_scores_engineered.parquet",   # saved by phase3 — full 75-feature set
        "race_scores_or_context.parquet",
        "race_scores_going_pref.parquet",
        "race_scores_pedigree.parquet",
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
            print(f"[data] {len(df):,} rows | {df['date_dt'].min().date()} – {df['date_dt'].max().date()}")
            return df
    raise FileNotFoundError("No historical parquet dataset found in data/processed/")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric features present in the dataframe that can be used for training.

    Reads preferred list from models/feature_columns.txt; falls back to all float/int
    columns that are not outcome or ID columns.
    """
    feat_file = MODEL_DIR / "feature_columns.txt"
    if feat_file.exists():
        preferred = [l.strip() for l in feat_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        available = [f for f in preferred if f in df.columns]
        missing = [f for f in preferred if f not in df.columns]
        if missing:
            print(f"[features] {len(missing)} preferred features not in data — skipped: {missing[:5]}")
        print(f"[features] Using {len(available)}/{len(preferred)} preferred features")
        return available

    # Fallback: all numeric columns that are not obviously IDs or outcome cols
    exclude = {
        "won", "placed", "pos_clean", "pos", "btn", "btn_lengths",
        "race_id", "horse_id", "jockey_id", "trainer_id",
        "date", "date_dt", "off", "course", "horse", "jockey", "trainer",
    }
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    print(f"[features] Fallback: {len(cols)} numeric features")
    return cols


# ─────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────

def build_model():
    """Return a fresh, fast XGBoost/RF classifier."""
    if HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=10,   # typical horses per race ≈ 10
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
    )


def predict_proba_col1(model, X) -> np.ndarray:
    """Return P(win) column from any sklearn-compatible model."""
    return model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────
# Accuracy metrics (race-level)
# ─────────────────────────────────────────────────────────────

def race_level_accuracy(df_test: pd.DataFrame, pred_col: str = "pred_prob") -> dict:
    """Calculate top-1 and top-3 accuracy per race.

    Returns dict with top1_accuracy and top3_accuracy (float 0-1).
    """
    grps = df_test.groupby("race_id")
    top1, top3 = [], []
    for _, grp in grps:
        if grp["won"].sum() == 0:
            continue   # race with no recorded winner (void, abandoned)
        top1.append(int(grp.nlargest(1, pred_col)["won"].any()))
        top3.append(int(grp.nlargest(3, pred_col)["won"].any()))
    return {
        "top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "top3_accuracy": float(np.mean(top3)) if top3 else 0.0,
        "n_races": len(top1),
    }


# ─────────────────────────────────────────────────────────────
# Betting simulation helpers
# ─────────────────────────────────────────────────────────────

def bsp_from_win_prob(win_prob: float) -> float:
    """Estimate fair BSP decimal odds from model win probability.

    Applies a ~115% overround correction so the expected ROI for a
    perfectly-calibrated model is approximately -13% (the overround).
    This is used purely as a model calibration test, not a real betting
    simulation (which requires actual bookmaker odds).
    """
    if win_prob <= 0:
        return 100.0
    # 1/p = fair odds; multiply by 0.87 (= 1/1.15) to apply overround
    return round((1.0 / win_prob) * 0.87, 2)


def simulate_level_stakes(
    df_test: pd.DataFrame,
    min_prob: float = 0.15,
    stake: float = 1.0,
    pred_col: str = "pred_prob",
) -> pd.DataFrame:
    """Bet level stakes on top model pick per race when confidence ≥ min_prob.

    Uses estimated BSP (fair odds) for P&L. Returns a DataFrame with one row
    per bet: race_id, horse, stake, odds, won, pnl, cumulative_pnl.
    """
    rows = []
    for race_id, grp in df_test.groupby("race_id"):
        top = grp.nlargest(1, pred_col).iloc[0]
        if top[pred_col] < min_prob:
            continue

        est_odds = bsp_from_win_prob(top[pred_col])
        won = int(top["won"])
        pnl = (est_odds - 1) * stake * won - stake * (1 - won)

        rows.append({
            "race_id": race_id,
            "date": top.get("date", ""),
            "horse": top.get("horse", ""),
            "pred_prob": round(top[pred_col], 4),
            "est_odds": est_odds,
            "stake": stake,
            "won": won,
            "pnl": round(pnl, 4),
        })

    if not rows:
        return pd.DataFrame()

    bets_df = pd.DataFrame(rows)
    bets_df["cumulative_pnl"] = bets_df["pnl"].cumsum()
    bets_df["total_staked"] = stake * np.arange(1, len(bets_df) + 1)
    bets_df["roi_pct"] = (bets_df["cumulative_pnl"] / bets_df["total_staked"] * 100).round(2)
    return bets_df


def kelly_fraction(win_prob: float, decimal_odds: float, f: float = 0.25) -> float:
    """Fractional Kelly stake (default 25% Kelly for safety).

    Returns fraction of bankroll to stake; 0 if no edge.
    """
    b = decimal_odds - 1.0  # net odds
    edge = b * win_prob - (1.0 - win_prob)
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return min(full_kelly * f, 0.10)   # cap at 10% of bankroll


def simulate_value_betting(
    df_test: pd.DataFrame,
    min_edge: float = 0.05,
    bankroll_initial: float = 1000.0,
    kelly_frac: float = 0.25,
    pred_col: str = "pred_prob",
) -> pd.DataFrame:
    """Value betting simulation — requires real bookmaker odds.

    Without actual market odds to compare against the model's probabilities,
    this simulation cannot produce meaningful results. Returns an empty
    DataFrame. Populate df_test with a 'market_odds' column from a live
    odds feed (e.g. The Odds API) to enable this simulation.
    """
    if "market_odds" not in df_test.columns:
        return pd.DataFrame()  # no real odds available

    bankroll = bankroll_initial
    rows = []

    for race_id, grp in df_test.groupby("race_id"):
        top = grp.nlargest(1, pred_col).iloc[0]
        win_prob = top[pred_col]
        market_odds = top["market_odds"]
        if pd.isna(market_odds) or market_odds <= 1.0:
            continue

        implied_market_prob = 1.0 / market_odds
        edge = win_prob - implied_market_prob
        if edge < min_edge:
            continue

        stake_frac = kelly_fraction(win_prob, market_odds, f=kelly_frac)
        stake = max(bankroll * stake_frac, 0.01)
        stake = round(stake, 2)

        won = int(top["won"])
        pnl = (market_odds - 1) * stake * won - stake * (1 - won)
        bankroll += pnl
        bankroll = max(bankroll, 0.01)

        rows.append({
            "race_id": race_id,
            "date": top.get("date", ""),
            "horse": top.get("horse", ""),
            "pred_prob": round(win_prob, 4),
            "market_odds": market_odds,
            "edge": round(edge, 4),
            "stake": stake,
            "won": won,
            "pnl": round(pnl, 4),
            "bankroll": round(bankroll, 2),
        })

    if not rows:
        return pd.DataFrame()

    bets_df = pd.DataFrame(rows)
    bets_df["roi_pct"] = (
        (bets_df["bankroll"] - bankroll_initial) / bankroll_initial * 100
    ).round(2)
    return bets_df


# ─────────────────────────────────────────────────────────────
# Drawdown analysis
# ─────────────────────────────────────────────────────────────

def drawdown_stats(pnl_series: pd.Series) -> dict:
    """Compute max drawdown, recovery, Sharpe, and win-rate from a P&L series."""
    if len(pnl_series) == 0:
        return {"max_drawdown": 0.0, "sharpe": 0.0, "win_rate": 0.0, "total_pnl": 0.0}

    equity = pnl_series.cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    max_dd = float(drawdown.max())

    # Sharpe ratio (annualised assuming 1 bet/day ~ 365 bets/year)
    mean_pnl = float(pnl_series.mean())
    std_pnl = float(pnl_series.std())
    sharpe = (mean_pnl / std_pnl * np.sqrt(365)) if std_pnl > 0 else 0.0

    win_rate = float((pnl_series > 0).mean())
    total_pnl = float(pnl_series.sum())

    return {
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 4),
        "n_bets": int(len(pnl_series)),
    }


# ─────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_months: int = 24,
    test_months: int = 1,
    n_folds: int = 12,
    min_prob_level: float = 0.15,
    min_edge_value: float = 0.05,
    kelly_frac: float = 0.25,
) -> list[dict]:
    """Run walk-forward cross-validation returning per-fold result dicts."""
    df = df.sort_values("date_dt").copy()
    max_date = df["date_dt"].max()

    # Need a unique race identifier
    if "race_id" not in df.columns:
        df["race_id"] = df["date"].astype(str) + "_" + df["course_clean"].astype(str) + "_" + df["off"].astype(str)

    # Binary target
    if "won" not in df.columns:
        df["won"] = (df["pos_clean"] == 1).astype(int)

    results = []
    print(f"\n{'='*65}")
    print(f" WALK-FORWARD CV — {n_folds} folds × {test_months}m test / {train_months}m train")
    print(f"{'='*65}")

    for fold_i in range(n_folds):
        # Define test window (rolling back from max_date)
        test_end   = max_date  - pd.DateOffset(months=fold_i * test_months)
        test_start = test_end  - pd.DateOffset(months=test_months)
        train_end  = test_start
        train_start = train_end - pd.DateOffset(months=train_months)

        train = df[(df["date_dt"] >= train_start) & (df["date_dt"] < train_end)]
        test  = df[(df["date_dt"] >= test_start)  & (df["date_dt"] < test_end)]

        # Need at minimum 500 train and 50 test samples
        if len(train) < 500 or len(test) < 50:
            print(f"[fold {fold_i+1:2d}] ⚠  insufficient data — skipping")
            continue

        # Filter test to races that have an actual winner recorded
        test = test[test.groupby("race_id")["won"].transform("sum") > 0].copy()
        if len(test) < 50:
            continue

        # Select available features only
        avail = [f for f in feature_cols if f in train.columns]
        X_train = train[avail].fillna(0)
        y_train = train["won"]
        X_test  = test[avail].fillna(0)
        y_test  = test["won"]

        # Train model
        model = build_model()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"[fold {fold_i+1:2d}] ✗ training failed: {e}")
            continue

        # Predictions
        test = test.copy()
        test["pred_prob"] = predict_proba_col1(model, X_test)

        # Classification metrics
        try:
            auc = roc_auc_score(y_test, test["pred_prob"])
        except Exception:
            auc = 0.0
        try:
            ll = log_loss(y_test, test["pred_prob"])
        except Exception:
            ll = float("nan")

        acc_metrics = race_level_accuracy(test)

        # Betting simulations
        level_bets = simulate_level_stakes(test, min_prob=min_prob_level)
        value_bets = simulate_value_betting(test, min_edge=min_edge_value, kelly_frac=kelly_frac)

        level_stats = drawdown_stats(level_bets["pnl"]) if len(level_bets) > 0 else {}
        value_stats = drawdown_stats(value_bets["pnl"]) if len(value_bets) > 0 else {}

        level_roi = (
            level_bets["cumulative_pnl"].iloc[-1] / (len(level_bets) * 1.0) * 100
            if len(level_bets) > 0 else 0.0
        )
        value_roi = (
            (value_bets["bankroll"].iloc[-1] - 1000.0) / 1000.0 * 100
            if len(value_bets) > 0 else 0.0
        )

        fold_result = {
            "fold": fold_i + 1,
            "train_start": str(train_start.date()),
            "train_end":   str(train_end.date()),
            "test_start":  str(test_start.date()),
            "test_end":    str(test_end.date()),
            "train_size":  int(len(train)),
            "test_size":   int(len(test)),
            "n_features":  len(avail),
            # Model metrics
            "auc":          round(auc, 4),
            "log_loss":     round(ll, 4) if not np.isnan(ll) else None,
            "top1_accuracy": round(acc_metrics["top1_accuracy"], 4),
            "top3_accuracy": round(acc_metrics["top3_accuracy"], 4),
            "n_races":      acc_metrics["n_races"],
            # Level stakes
            "level_bets":   int(len(level_bets)),
            "level_roi_pct": round(level_roi, 2),
            "level_max_dd": level_stats.get("max_drawdown", 0.0),
            "level_sharpe": level_stats.get("sharpe", 0.0),
            # Value betting
            "value_bets":   int(len(value_bets)),
            "value_roi_pct": round(value_roi, 2),
            "value_max_dd": value_stats.get("max_drawdown", 0.0),
            "value_sharpe": value_stats.get("sharpe", 0.0),
        }

        print(
            f"[fold {fold_i+1:2d}] "
            f"{test_start.date()} – {test_end.date()} | "
            f"AUC={auc:.3f} | "
            f"Top-1={acc_metrics['top1_accuracy']*100:.1f}% | "
            f"Top-3={acc_metrics['top3_accuracy']*100:.1f}% | "
            f"Level ROI={level_roi:+.1f}% | "
            f"Value ROI={value_roi:+.1f}%"
        )

        results.append(fold_result)

    return results


# ─────────────────────────────────────────────────────────────
# Summary reporting
# ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Print aggregate summary across all folds."""
    if not results:
        print("\n[!] No fold results to summarise.")
        return

    df_r = pd.DataFrame(results)
    print(f"\n{'='*65}")
    print(" AGGREGATE SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*65}")
    print(f"  Folds completed:   {len(df_r)}")
    print(f"  AUC — mean={df_r['auc'].mean():.3f}  std={df_r['auc'].std():.3f}  min={df_r['auc'].min():.3f}  max={df_r['auc'].max():.3f}")
    print(f"  Top-1 acc — mean={df_r['top1_accuracy'].mean()*100:.1f}%  std={df_r['top1_accuracy'].std()*100:.1f}%")
    print(f"  Top-3 acc — mean={df_r['top3_accuracy'].mean()*100:.1f}%  std={df_r['top3_accuracy'].std()*100:.1f}%")

    print(f"\n  Level stakes (model calibration test — uses model-implied odds, not real bookmaker odds):")
    print(f"    Avg bets/fold:  {df_r['level_bets'].mean():.1f}")
    print(f"    Avg ROI:        {df_r['level_roi_pct'].mean():+.2f}%  (target: ~-13% for well-calibrated model)")
    print(f"    Max drawdown:   {df_r['level_max_dd'].mean():.2f} units (avg)")
    print(f"    Avg Sharpe:     {df_r['level_sharpe'].mean():.3f}")

    print(f"\n  Value betting (Kelly 25% frac):")
    if df_r['value_bets'].mean() == 0:
        print(f"    N/A — requires real bookmaker odds (add 'market_odds' column to dataset)")
        print(f"          Fetch via The Odds API and join to historical data to enable this.")
    else:
        print(f"    Avg bets/fold:  {df_r['value_bets'].mean():.1f}")
        print(f"    Avg ROI:        {df_r['value_roi_pct'].mean():+.2f}%")
        print(f"    Max drawdown:   {df_r['value_max_dd'].mean():.2f} (avg)")
        print(f"    Avg Sharpe:     {df_r['value_sharpe'].mean():.3f}")

    # Benchmark reference
    print(f"\n  Benchmark reference:")
    print(f"    Random pick Top-1: ~{100/8:.1f}%  (assuming 8-runner avg field)")
    print(f"    Random pick Top-3: ~{300/8:.1f}%")
    print(f"    Level stakes break-even ROI: 0.0%")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward backtesting for horse racing model")
    p.add_argument("--train-months", type=int, default=24, help="Training window months (default 24)")
    p.add_argument("--test-months",  type=int, default=1,  help="Test window months (default 1)")
    p.add_argument("--folds",        type=int, default=12, help="Number of CV folds (default 12)")
    p.add_argument("--min-prob",     type=float, default=0.15, help="Minimum win prob for level staking")
    p.add_argument("--min-edge",     type=float, default=0.05, help="Minimum edge for value betting")
    p.add_argument("--kelly-frac",   type=float, default=0.25, help="Kelly fraction (default 0.25)")
    p.add_argument("--plot",         action="store_true", help="Save equity curve plots to data/processed/")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    df = load_data()
    feature_cols = get_feature_cols(df)

    if not feature_cols:
        print("[!] No feature columns found — aborting.")
        sys.exit(1)

    # Filter to rows with a valid outcome
    df = df[df["pos_clean"].notna()].copy()
    if "won" not in df.columns:
        df["won"] = (df["pos_clean"] == 1).astype(int)

    # Run walk-forward CV
    results = walk_forward_cv(
        df,
        feature_cols=feature_cols,
        train_months=args.train_months,
        test_months=args.test_months,
        n_folds=args.folds,
        min_prob_level=args.min_prob,
        min_edge_value=args.min_edge,
        kelly_frac=args.kelly_frac,
    )

    print_summary(results)

    # Save results
    if results:
        import numpy as _np
        def _json_safe(obj):
            """Convert numpy scalars to plain Python types for JSON."""
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not JSON serializable: {type(obj)}")

        out_json = DATA_DIR / "backtest_results.json"
        with open(out_json, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.utcnow().isoformat(),
                    "config": {
                        "train_months": args.train_months,
                        "test_months": args.test_months,
                        "folds": args.folds,
                        "min_prob": args.min_prob,
                        "min_edge": args.min_edge,
                        "kelly_frac": args.kelly_frac,
                    },
                    "folds": results,
                },
                f,
                indent=2,
                default=_json_safe,
            )
        print(f"\n[✓] Results saved → {out_json}")

        df_summary = pd.DataFrame(results)
        out_csv = DATA_DIR / "backtest_summary.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[✓] Summary table → {out_csv}")

        # Optional plots
        if args.plot:
            _save_equity_plots(results)
    else:
        print("\n[!] No results produced — check data coverage for the requested date range.")


def _save_equity_plots(results: list[dict]) -> None:
    """Save a simple equity-curve PNG using matplotlib (best effort)."""
    try:
        import matplotlib.pyplot as plt

        folds = [r["fold"] for r in results]
        aucs = [r["auc"] for r in results]
        level_rois = [r["level_roi_pct"] for r in results]
        value_rois = [r["value_roi_pct"] for r in results]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(folds, aucs, marker="o", color="steelblue")
        axes[0].axhline(0.65, color="gray", linestyle="--", alpha=0.5, label="baseline 0.65")
        axes[0].set_title("ROC AUC by Fold")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("AUC")
        axes[0].legend()

        axes[1].bar(folds, level_rois, color=["green" if r >= 0 else "red" for r in level_rois])
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Level Stakes ROI % by Fold")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("ROI %")

        axes[2].bar(folds, value_rois, color=["green" if r >= 0 else "red" for r in value_rois])
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_title("Value Betting ROI % by Fold (Kelly 25%)")
        axes[2].set_xlabel("Fold")
        axes[2].set_ylabel("ROI %")

        plt.tight_layout()
        out_png = DATA_DIR / "backtest_equity_curve.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[✓] Equity curve plot → {out_png}")
    except Exception as e:
        print(f"[!] Plot generation failed: {e}")


if __name__ == "__main__":
    main()
