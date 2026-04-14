#!/usr/bin/env python3
"""Merge scraped bookmaker odds (Racing Post + ATR) into a predictions CSV.

After this script runs, the predictions CSV gains:
  - odds_live, odds_decimal, odds_trend, odds_1‥4  (from RP)
  - best_odds_decimal, best_bookmaker, num_bookmakers (from ATR)
  - market_odds          — best available (ATR > RP fallback)
  - implied_prob         — market's win probability = 1 / market_odds
  - edge                 — model win_probability − implied_prob
  - ev_per_unit          — expected value per £1 staked
  - kelly_quarter        — quarter-Kelly stake fraction
  - is_value_bet         — True when edge > 1% AND EV > 1%
  - is_strong_value      — True when edge > 15% AND EV > 5%
  - odds_trend_encoded   — shortening=1, stable=0, drifting=−1 (model feature)
  - edge_adjusted        — edge + trend adjustment (shortening +2%, drifting −2%)

Usage:
    python scripts/merge_scraped_odds.py --date 2026-04-22

Model-feature note:
    odds_trend_encoded and implied_prob (normalised within race) are high-value
    features for future model retraining. They are appended here so they are
    immediately available for value-bet calculations and UI display, and will
    automatically be picked up by phase3_build_horse_model.py if the column
    is added to the feature list.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Columns that may have been merged in a previous run — drop before re-merging
_ODDS_COLS = [
    "odds_live", "odds_decimal", "odds_trend",
    "odds_1", "odds_2", "odds_3", "odds_4",
    "best_odds_decimal", "best_bookmaker", "num_bookmakers",
    "market_odds", "implied_prob",
    "edge", "ev_per_unit", "kelly_quarter",
    "is_value_bet", "is_strong_value",
    "odds_trend_encoded", "edge_adjusted",
]


# ─────────────────────────────────────────────
#  Name normalisation (fuzzy join key)
# ─────────────────────────────────────────────

def _norm(name) -> str:
    """Lowercase, strip country suffix, collapse whitespace.

    Allows matching 'Desert Crown (GB)' ↔ 'Desert Crown'.
    """
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name)
    name = name.lower().strip()
    name = re.sub(r"['\-\.]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


# ─────────────────────────────────────────────
#  Source merges
# ─────────────────────────────────────────────

def merge_rp_odds(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Left-join Racing Post single-bookie odds + trend into predictions."""
    rp_path = RAW_DIR / f"rp_odds_{date_str}.csv"
    if not rp_path.exists():
        print(f"  RP odds not found: {rp_path} — skipping")
        return df

    df_rp = pd.read_csv(rp_path)
    rp_want = [c for c in
               ["horse_name", "odds_live", "odds_decimal", "odds_trend",
                "odds_1", "odds_2", "odds_3", "odds_4"]
               if c in df_rp.columns]

    df_rp["_nk"] = df_rp["horse_name"].apply(_norm)
    df["_nk"] = df["horse"].apply(_norm)

    merged = df.merge(
        df_rp[rp_want + ["_nk"]].drop_duplicates("_nk"),
        on="_nk", how="left",
    ).drop(columns=["_nk", "horse_name"], errors="ignore")
    df.drop(columns=["_nk"], inplace=True, errors="ignore")

    matched = merged["odds_live"].notna().sum() if "odds_live" in merged.columns else 0
    print(f"  RP odds matched: {matched}/{len(merged)} horses")
    return merged


def merge_atr_best_odds(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Left-join ATR best-available bookmaker price into predictions."""
    atr_path = RAW_DIR / f"atr_odds_best_{date_str}.csv"
    if not atr_path.exists():
        print(f"  ATR best-odds not found: {atr_path} — skipping")
        return df

    df_atr = pd.read_csv(atr_path)
    atr_want = [c for c in
                ["horse_name", "best_odds_decimal", "best_bookmaker", "num_bookmakers"]
                if c in df_atr.columns]

    df_atr["_nk"] = df_atr["horse_name"].apply(_norm)
    df["_nk"] = df["horse"].apply(_norm)

    merged = df.merge(
        df_atr[atr_want + ["_nk"]].drop_duplicates("_nk"),
        on="_nk", how="left",
    ).drop(columns=["_nk", "horse_name"], errors="ignore")
    df.drop(columns=["_nk"], inplace=True, errors="ignore")

    matched = merged["best_odds_decimal"].notna().sum() if "best_odds_decimal" in merged.columns else 0
    print(f"  ATR odds matched: {matched}/{len(merged)} horses")
    return merged


# ─────────────────────────────────────────────
#  Value columns
# ─────────────────────────────────────────────

def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute edge, EV, Kelly, and value-bet flags from merged odds.

    Priority: ATR best_odds_decimal > RP odds_decimal (more bookmakers = better).
    All calculations use win_probability from the model.
    """
    # Best available market odds (prioritise ATR multi-bookie price)
    market = pd.Series(pd.NA, index=df.index, dtype=float)
    if "best_odds_decimal" in df.columns:
        market = pd.to_numeric(df["best_odds_decimal"], errors="coerce")
    if "odds_decimal" in df.columns:
        rp = pd.to_numeric(df["odds_decimal"], errors="coerce")
        market = market.where(market.notna(), rp)

    df["market_odds"] = market.where(market > 1.0)

    valid = df["market_odds"].notna()

    # Market-implied win probability
    df["implied_prob"] = pd.NA
    df.loc[valid, "implied_prob"] = 1.0 / df.loc[valid, "market_odds"]

    # Model win probability column (predictions CSV uses 'win_probability')
    if "win_probability" not in df.columns:
        print("  No win_probability column — skipping edge/value calculations")
        return df

    win_p = pd.to_numeric(df["win_probability"], errors="coerce")
    has_both = valid & win_p.notna()

    # Edge: how much better our model is than the market
    df["edge"] = pd.NA
    df.loc[has_both, "edge"] = win_p.loc[has_both] - df.loc[has_both, "implied_prob"]

    # Expected value per £1 staked: win_p * decimal_odds − 1
    df["ev_per_unit"] = pd.NA
    df.loc[has_both, "ev_per_unit"] = (
        win_p.loc[has_both] * df.loc[has_both, "market_odds"] - 1.0
    )

    # Quarter-Kelly stake fraction
    df["kelly_quarter"] = pd.NA
    b = df["market_odds"] - 1.0
    kelly_full = ((win_p * df["market_odds"] - 1.0) / b).clip(lower=0)
    df.loc[has_both, "kelly_quarter"] = kelly_full.loc[has_both] * 0.25

    # Value flags
    edge_s = pd.to_numeric(df.get("edge"), errors="coerce")
    ev_s = pd.to_numeric(df.get("ev_per_unit"), errors="coerce")
    df["is_value_bet"] = has_both & (edge_s > 0.01) & (ev_s > 0.01)
    df["is_strong_value"] = has_both & (edge_s > 0.15) & (ev_s > 0.05)

    # Odds-trend encoded (model-ready feature for future retraining)
    if "odds_trend" in df.columns:
        df["odds_trend_encoded"] = (
            df["odds_trend"]
            .map({"shortening": 1, "stable": 0, "drifting": -1, "no_data": 0})
            .fillna(0)
            .astype(int)
        )
        # Trend-adjusted edge: shortening adds +2% confidence
        trend_adj = df["odds_trend"].map(
            {"shortening": 0.02, "drifting": -0.02, "stable": 0.0, "no_data": 0.0}
        ).fillna(0.0)
        df["edge_adjusted"] = edge_s + trend_adj

    return df


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge scraped bookmaker odds into predictions CSV"
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    pred_path = PROCESSED_DIR / f"predictions_{args.date}.csv"
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        return

    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} predictions for {args.date}")

    # Drop any stale odds columns from a previous merge run
    stale = [c for c in _ODDS_COLS if c in df.columns]
    if stale:
        df.drop(columns=stale, inplace=True)
        print(f"  Dropped {len(stale)} existing odds columns (re-merging fresh)")

    print("Merging Racing Post odds...")
    df = merge_rp_odds(df, args.date)

    print("Merging ATR best-odds...")
    df = merge_atr_best_odds(df, args.date)

    print("Computing value columns...")
    df = add_value_columns(df)

    df.to_csv(pred_path, index=False)
    print(f"\nSaved enriched predictions → {pred_path}")

    # Summary
    odds_cov = df["market_odds"].notna().sum() if "market_odds" in df.columns else 0
    print(f"Odds coverage: {odds_cov}/{len(df)} horses ({odds_cov/max(len(df),1):.0%})")

    if "is_value_bet" in df.columns:
        n_value = int(df["is_value_bet"].sum())
        n_strong = int(df["is_strong_value"].sum()) if "is_strong_value" in df.columns else 0
        print(f"Value bets: {n_value}  |  Strong value bets: {n_strong}")

    if "edge" in df.columns:
        valid_edge = pd.to_numeric(df["edge"], errors="coerce").dropna()
        if len(valid_edge) > 0:
            print(f"Avg edge: {valid_edge.mean():+.1%}"
                  f"  |  Max edge: {valid_edge.max():+.1%}"
                  f"  |  Min edge: {valid_edge.min():+.1%}")


if __name__ == "__main__":
    main()
