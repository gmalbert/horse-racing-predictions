#!/usr/bin/env python3
"""
Bankroll Manager — Kelly Criterion + Adaptive Drawdown Protection.

Implements a production-ready betting bankroll management system:

  1.  Fractional Kelly criterion staking (default 25% Kelly for risk control)
  2.  Drawdown-aware stake reduction (throttles bets after losing streaks)
  3.  Rolling-window performance tracking (last 30 / 90 days)
  4.  Per-confidence-tier sizing (high/medium/low conviction)
  5.  Daily summary report with recommended stakes for today's predictions

Works directly with the predictions CSV produced by predict_todays_races.py.

Usage
-----
  # Today's recommendations (reads latest predictions CSV):
  python scripts/rl_bankroll_manager.py

  # With explicit predictions file:
  python scripts/rl_bankroll_manager.py --date 2026-04-16

  # Review full performance history and update state:
  python scripts/rl_bankroll_manager.py --update-history

  # Show performance report only:
  python scripts/rl_bankroll_manager.py --report

State is stored in data/processed/bankroll_state.json and updated each run.

Staking rules (conservative defaults, adjust via --bankroll and --kelly-frac):
  - Full-Kelly fraction: 0.25  (quarter Kelly — conservative)
  - Minimum stake:  1% of bankroll
  - Maximum stake: 10% of bankroll per bet
  - Drawdown throttle: stakes halved when drawdown > 15%, quartered > 25%
"""

import argparse
import json
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
STATE_FILE = DATA_DIR / "bankroll_state.json"


# ─────────────────────────────────────────────────────────────
# Kelly criterion helpers
# ─────────────────────────────────────────────────────────────

def kelly_stake_fraction(
    win_prob: float,
    decimal_odds: float,
    frac: float = 0.25,
    min_frac: float = 0.01,
    max_frac: float = 0.10,
) -> float:
    """Compute the fractional-Kelly stake fraction.

    Args:
        win_prob:     Model P(win), 0–1.
        decimal_odds: Best available decimal odds (e.g. 4.0).
        frac:         Kelly fraction (0.25 = 25% Kelly).
        min_frac:     Floor stake as fraction of bankroll.
        max_frac:     Ceiling stake as fraction of bankroll.

    Returns:
        Fraction of current bankroll to stake; 0.0 if no positive edge.
    """
    b = decimal_odds - 1.0        # net profit per unit staked if win
    edge = b * win_prob - (1.0 - win_prob)
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    sized = full_kelly * frac
    return float(np.clip(sized, min_frac, max_frac))


def model_odds_from_prob(win_prob: float) -> float:
    """Convert model P(win) to fair decimal odds."""
    if win_prob <= 0:
        return 999.0
    return round(1.0 / win_prob, 2)


# ─────────────────────────────────────────────────────────────
# Bankroll state management
# ─────────────────────────────────────────────────────────────

DEFAULT_STATE = {
    "initial_bankroll": 1000.0,
    "current_bankroll": 1000.0,
    "peak_bankroll": 1000.0,
    "total_bets": 0,
    "total_wins": 0,
    "total_staked": 0.0,
    "total_pnl": 0.0,
    "history": [],          # list of bet records
    "last_updated": None,
}


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Back-fill any missing keys
        for k, v in DEFAULT_STATE.items():
            if k not in state:
                state[k] = v
        return state
    return dict(DEFAULT_STATE)


def save_state(state: dict) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# Drawdown calculation
# ─────────────────────────────────────────────────────────────

def current_drawdown(state: dict) -> float:
    """Return current drawdown as a fraction of peak bankroll (0–1)."""
    if state["peak_bankroll"] <= 0:
        return 0.0
    dd = (state["peak_bankroll"] - state["current_bankroll"]) / state["peak_bankroll"]
    return max(0.0, dd)


def drawdown_multiplier(drawdown_frac: float) -> float:
    """Return a stake-size multiplier based on current drawdown.

    Throttling schedule:
      dd  < 10%  → 1.00  (full size)
      10% ≤ dd < 15%  → 0.75
      15% ≤ dd < 25%  → 0.50  (half size)
      25% ≤ dd < 35%  → 0.25  (quarter size)
      dd  ≥ 35%  → 0.10  (minimal — almost stopped)
    """
    if drawdown_frac < 0.10:
        return 1.00
    if drawdown_frac < 0.15:
        return 0.75
    if drawdown_frac < 0.25:
        return 0.50
    if drawdown_frac < 0.35:
        return 0.25
    return 0.10


def confidence_tier(win_prob: float) -> str:
    """Classify a win probability into a conviction tier."""
    if win_prob >= 0.40:
        return "HIGH"
    if win_prob >= 0.25:
        return "MEDIUM"
    return "LOW"


def tier_multiplier(tier: str) -> float:
    """Additional size multiplier by conviction tier."""
    return {"HIGH": 1.20, "MEDIUM": 1.00, "LOW": 0.70}.get(tier, 1.00)


# ─────────────────────────────────────────────────────────────
# Performance analytics
# ─────────────────────────────────────────────────────────────

def rolling_stats(state: dict, window_days: int = 30) -> dict:
    """Compute stats over the last *window_days* of bet history."""
    if not state["history"]:
        return {"bets": 0, "wins": 0, "roi_pct": 0.0, "pnl": 0.0, "win_rate": 0.0}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    recent = [b for b in state["history"] if b.get("timestamp", "") >= cutoff]

    if not recent:
        return {"bets": 0, "wins": 0, "roi_pct": 0.0, "pnl": 0.0, "win_rate": 0.0}

    bets   = len(recent)
    wins   = sum(1 for b in recent if b["result"] == "win")
    staked = sum(b["stake"] for b in recent)
    pnl    = sum(b["pnl"] for b in recent)

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": round(wins / bets, 4) if bets else 0.0,
        "roi_pct": round(pnl / staked * 100, 2) if staked else 0.0,
        "pnl": round(pnl, 2),
        "total_staked": round(staked, 2),
    }


def max_drawdown_history(state: dict) -> float:
    """Compute maximum historical drawdown from the bankroll history."""
    if not state["history"]:
        return 0.0
    bankrolls = [state["initial_bankroll"]] + [b["bankroll_after"] for b in state["history"]]
    peak = bankrolls[0]
    max_dd = 0.0
    for br in bankrolls:
        if br > peak:
            peak = br
        dd = (peak - br) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def sharpe_ratio_history(state: dict) -> float:
    """Annualised Sharpe ratio from daily PnL histogram."""
    if len(state["history"]) < 10:
        return 0.0
    pnls = np.array([b["pnl"] for b in state["history"]])
    if pnls.std() == 0:
        return 0.0
    return round(float(pnls.mean() / pnls.std() * np.sqrt(365)), 3)


# ─────────────────────────────────────────────────────────────
# Recommendations for today
# ─────────────────────────────────────────────────────────────

def load_predictions(target_date: str) -> pd.DataFrame | None:
    """Load the predictions CSV for a given date."""
    p = DATA_DIR / f"predictions_{target_date}.csv"
    if not p.exists():
        print(f"[!] No predictions found for {target_date} ({p})")
        return None
    df = pd.read_csv(p)
    print(f"[predictions] Loaded {len(df)} horses for {target_date}")
    return df


def generate_recommendations(
    df_preds: pd.DataFrame,
    state: dict,
    kelly_frac: float = 0.25,
    min_win_prob: float = 0.15,
) -> pd.DataFrame:
    """Generate stake recommendations for each race on the card.

    For each race, picks the top model selection and calculates:
      - Kelly stake
      - Drawdown-adjusted stake
      - Confidence tier
      - Expected value

    Returns a DataFrame sorted by recommendation strength.
    """
    bankroll = state["current_bankroll"]
    dd_frac   = current_drawdown(state)
    dd_mult   = drawdown_multiplier(dd_frac)

    rows = []
    # Group by race (course + off time)
    race_key = []
    if "course" in df_preds.columns and "race_time" in df_preds.columns:
        df_preds = df_preds.copy()
        df_preds["_race_key"] = df_preds["course"] + "_" + df_preds["race_time"].astype(str)
        race_key = "_race_key"
    elif "race_id" in df_preds.columns:
        race_key = "race_id"
    else:
        # Treat each row as its own race
        df_preds = df_preds.copy()
        df_preds["_race_key"] = range(len(df_preds))
        race_key = "_race_key"

    for rk, grp in df_preds.groupby(race_key):
        # Top model pick
        if "win_probability" not in grp.columns:
            continue
        top = grp.nlargest(1, "win_probability").iloc[0]
        win_prob = float(top["win_probability"])

        if win_prob < min_win_prob:
            continue

        # Market odds (from scraped data if available, else estimate fair odds)
        if "market_odds" in top.index and pd.notna(top.get("market_odds")):
            try:
                mkt_odds = float(top["market_odds"])
            except (ValueError, TypeError):
                mkt_odds = round(1.0 / win_prob / 0.87, 2)   # estimated
        else:
            mkt_odds = round(1.0 / win_prob / 0.87, 2)

        # Kelly staking
        base_frac = kelly_stake_fraction(win_prob, mkt_odds, frac=kelly_frac)
        if base_frac <= 0:
            continue   # no edge — skip

        tier = confidence_tier(win_prob)
        t_mult = tier_multiplier(tier)
        adj_frac = base_frac * t_mult * dd_mult
        adj_frac = float(np.clip(adj_frac, 0.005, 0.10))
        stake = round(bankroll * adj_frac, 2)

        # Expected value
        ev_pct = round((win_prob * (mkt_odds - 1) - (1 - win_prob)) * 100, 2)
        edge   = round(win_prob - 1.0 / mkt_odds, 4)

        rows.append({
            "race_time":   top.get("race_time", ""),
            "course":      top.get("course", ""),
            "horse":       top.get("horse", ""),
            "win_prob_%":  round(win_prob * 100, 1),
            "mkt_odds":    mkt_odds,
            "edge_%":      round(edge * 100, 1),
            "ev_%":        ev_pct,
            "tier":        tier,
            "kelly_frac_%": round(base_frac * 100, 2),
            "adj_frac_%":   round(adj_frac * 100, 2),
            "recommended_stake": stake,
            "dd_multiplier":     round(dd_mult, 2),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("ev_%", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Record a settled bet (for history tracking)
# ─────────────────────────────────────────────────────────────

def record_bet(
    state: dict,
    horse: str,
    race: str,
    stake: float,
    odds: float,
    won: bool,
) -> dict:
    """Record a settled bet result and update bankroll state."""
    pnl = (odds - 1) * stake if won else -stake
    state["current_bankroll"] += pnl
    state["peak_bankroll"] = max(state["peak_bankroll"], state["current_bankroll"])
    state["total_bets"] += 1
    state["total_wins"] += int(won)
    state["total_staked"] += stake
    state["total_pnl"] += pnl

    state["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "horse": horse,
        "race": race,
        "stake": stake,
        "odds": odds,
        "result": "win" if won else "loss",
        "pnl": round(pnl, 2),
        "bankroll_after": round(state["current_bankroll"], 2),
    })

    return state


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────

def print_report(state: dict) -> None:
    """Print a comprehensive performance report."""
    br = state["current_bankroll"]
    init = state["initial_bankroll"]
    peak = state["peak_bankroll"]
    dd_frac = current_drawdown(state)
    dd_mult = drawdown_multiplier(dd_frac)

    print(f"\n{'═'*60}")
    print(f"  BANKROLL PERFORMANCE REPORT")
    print(f"{'═'*60}")
    print(f"  Initial bankroll:   £{init:,.2f}")
    print(f"  Current bankroll:   £{br:,.2f}  ({(br-init)/init*100:+.1f}%)")
    print(f"  Peak bankroll:      £{peak:,.2f}")
    print(f"  Current drawdown:   {dd_frac*100:.1f}%  (stake multiplier: ×{dd_mult:.2f})")
    print(f"  Max historical DD:  {max_drawdown_history(state)*100:.1f}%")
    print(f"  Sharpe ratio:       {sharpe_ratio_history(state):.3f}")
    print()

    overall_bets  = state["total_bets"]
    overall_wins  = state["total_wins"]
    overall_staked = state["total_staked"]
    overall_pnl   = state["total_pnl"]
    overall_win_rt = overall_wins / overall_bets if overall_bets else 0
    overall_roi    = overall_pnl / overall_staked * 100 if overall_staked else 0

    print(f"  Overall: {overall_bets} bets | {overall_wins} wins ({overall_win_rt*100:.1f}%) | "
          f"Staked £{overall_staked:,.2f} | PnL £{overall_pnl:+,.2f} | ROI {overall_roi:+.1f}%")
    print()

    for window in [30, 90]:
        s = rolling_stats(state, window_days=window)
        if s["bets"] > 0:
            print(f"  Last {window:2d}d: {s['bets']:3d} bets | {s['wins']:3d} wins ({s['win_rate']*100:.1f}%) | "
                  f"PnL £{s['pnl']:+,.2f} | ROI {s['roi_pct']:+.1f}%")

    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────
# Update history from historical prediction CSVs
# ─────────────────────────────────────────────────────────────

def update_history_from_predictions(state: dict) -> dict:
    """Scan all predictions CSVs and back-fill actual outcomes.

    For each predictions_YYYY-MM-DD.csv that has 'actual_pos' column (set by
    a post-race update script), computes which bets would have been placed via
    Kelly and records wins/losses not yet in history.
    """
    existing_races = {b["race"] for b in state["history"]}
    added = 0

    for pred_file in sorted(DATA_DIR.glob("predictions_*.csv")):
        if pred_file.stat().st_size < 10:
            continue
        try:
            df = pd.read_csv(pred_file)
        except Exception:
            continue
        if "actual_pos" not in df.columns and "pos_clean" not in df.columns:
            continue  # No outcome data yet

        actual_col = "actual_pos" if "actual_pos" in df.columns else "pos_clean"
        df["won"] = pd.to_numeric(df[actual_col], errors="coerce") == 1

        if "race_time" not in df.columns or "win_probability" not in df.columns:
            continue

        df["_race_key"] = df["course"].astype(str) + "_" + df["race_time"].astype(str)

        for rk, grp in df.groupby("_race_key"):
            if rk in existing_races:
                continue
            top = grp.nlargest(1, "win_probability")
            if len(top) == 0:
                continue
            row = top.iloc[0]
            win_prob = float(row["win_probability"])
            if win_prob < 0.15:
                continue

            est_odds = round(1.0 / win_prob / 0.87, 2)
            frac = kelly_stake_fraction(win_prob, est_odds)
            if frac <= 0:
                continue

            stake = round(state["current_bankroll"] * frac, 2)
            state = record_bet(
                state,
                horse=str(row.get("horse", "")),
                race=rk,
                stake=stake,
                odds=est_odds,
                won=bool(row["won"]),
            )
            existing_races.add(rk)
            added += 1

    print(f"[history] Added {added} new settled bets")
    return state


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Bankroll manager — Kelly criterion + drawdown protection")
    p.add_argument("--date",          type=str, default=None, help="Predictions date YYYY-MM-DD (default: today)")
    p.add_argument("--bankroll",      type=float, default=None, help="Override starting bankroll (£)")
    p.add_argument("--kelly-frac",    type=float, default=0.25, help="Kelly fraction (default 0.25)")
    p.add_argument("--min-prob",      type=float, default=0.15, help="Minimum win probability to consider a bet")
    p.add_argument("--report",        action="store_true", help="Show performance report only")
    p.add_argument("--update-history", action="store_true", help="Scan prediction CSVs and update history")
    return p.parse_args()


def main():
    args = parse_args()

    # Load / initialise state
    state = load_state()

    if args.bankroll and state["total_bets"] == 0:
        state["initial_bankroll"] = args.bankroll
        state["current_bankroll"] = args.bankroll
        state["peak_bankroll"]    = args.bankroll
        print(f"[bankroll] Initialised with £{args.bankroll:,.2f}")

    if args.update_history:
        state = update_history_from_predictions(state)
        save_state(state)

    if args.report:
        print_report(state)
        return

    # Generate today's recommendations
    target_date = args.date or date.today().isoformat()
    df_preds = load_predictions(target_date)

    if df_preds is None:
        print_report(state)
        sys.exit(0)

    recs = generate_recommendations(df_preds, state, kelly_frac=args.kelly_frac, min_win_prob=args.min_prob)

    if recs.empty:
        print(f"\n[!] No bets recommended for {target_date} (no edges found above thresholds)")
    else:
        dd_frac = current_drawdown(state)
        print(f"\n{'─'*70}")
        print(f"  RECOMMENDED BETS — {target_date}")
        print(f"  Bankroll: £{state['current_bankroll']:,.2f}  |  Drawdown: {dd_frac*100:.1f}%  |  Kelly frac: {args.kelly_frac:.0%}")
        print(f"{'─'*70}")
        for _, r in recs.iterrows():
            tier_sym = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}.get(r["tier"], "")
            print(
                f"  {tier_sym} {r['horse']:<28}  {r['race_time']}@{r['course']}"
            )
            print(
                f"     P(win)={r['win_prob_%']:.1f}%  odds={r['mkt_odds']}  "
                f"edge={r['edge_%']:+.1f}%  EV={r['ev_%']:+.1f}%  "
                f"STAKE: £{r['recommended_stake']:.2f}"
            )
        print(f"{'─'*70}")
        print(f"  Total bets: {len(recs)}  |  Total staked: £{recs['recommended_stake'].sum():.2f}")

        # Save recommendations
        out = DATA_DIR / f"bankroll_recommendations_{target_date}.csv"
        recs.to_csv(out, index=False)
        print(f"\n[✓] Recommendations saved → {out}")

    print_report(state)
    save_state(state)


if __name__ == "__main__":
    main()
