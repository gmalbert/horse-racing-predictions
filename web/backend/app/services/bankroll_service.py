"""Bankroll state, bet logging and Kelly staking.

Defaults to the pipeline's bankroll_state.json and betting_history.csv.
HRP_STATE_DIR can isolate writable state from the read-only race data;
Compose uses this to persist manual betting records in a named volume.
"""
from __future__ import annotations

import json
import threading
from functools import wraps
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .. import config

_LOCK = threading.RLock()


def transaction(fn):
    """Serialize complete read/modify/write operations in the single API worker."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return wrapped

HISTORY_COLUMNS = [
    "bet_date", "race_id", "horse_name", "win_probability",
    "market_odds", "stake", "result", "profit",
]

DEFAULT_STATE: dict[str, Any] = {
    "initial_bankroll": 1000.0,
    "current_bankroll": 1000.0,
    "peak_bankroll": 1000.0,
    "total_bets": 0,
    "total_wins": 0,
    "total_staked": 0.0,
    "total_pnl": 0.0,
    "last_updated": None,
}


# ── persistence ──────────────────────────────────────────────────────────────

def load_state() -> dict[str, Any]:
    with _LOCK:
        if config.STATE_FILE.exists():
            with open(config.STATE_FILE) as fh:
                return {**DEFAULT_STATE, **json.load(fh)}
        return dict(DEFAULT_STATE)


def save_state(state: dict[str, Any]) -> None:
    with _LOCK:
        state["last_updated"] = date.today().isoformat()
        config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = config.STATE_FILE.with_suffix(".json.tmp")
        with open(temporary, "w") as fh:
            json.dump(state, fh, indent=2)
        temporary.replace(config.STATE_FILE)


def load_history() -> pd.DataFrame:
    with _LOCK:
        if config.BETTING_HISTORY_FILE.exists():
            df = pd.read_csv(config.BETTING_HISTORY_FILE, keep_default_na=False)
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[HISTORY_COLUMNS]
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_history(df: pd.DataFrame) -> None:
    with _LOCK:
        config.BETTING_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = config.BETTING_HISTORY_FILE.with_suffix(".csv.tmp")
        df[HISTORY_COLUMNS].to_csv(temporary, index=False)
        temporary.replace(config.BETTING_HISTORY_FILE)


@transaction
def update_bankroll(amount: float) -> dict[str, Any]:
    state = load_state()
    state["current_bankroll"] = amount
    state["peak_bankroll"] = max(float(state.get("peak_bankroll", 0)), amount)
    save_state(state)
    return state


@transaction
def list_bets() -> dict[str, Any]:
    df = load_history().reset_index().rename(columns={"index": "id"})
    return {"bets": df.astype(object).where(df.notna(), None).to_dict("records"), "state": load_state()}


# ── Kelly ────────────────────────────────────────────────────────────────────

def kelly_stake_fraction(win_prob: float, decimal_odds: float, frac: float = 0.25,
                         min_frac: float = 0.01, max_frac: float = 0.10) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    edge = b * win_prob - (1.0 - win_prob)
    if edge <= 0:
        return 0.0
    return float(np.clip((edge / b) * frac, min_frac, max_frac))


def kelly_recommendations(target_date: str, bankroll: float, frac: float, min_edge: float) -> dict[str, Any]:
    pred_file = config.DATA_DIR / f"predictions_{target_date}.csv"
    if not pred_file.exists():
        return {"date": target_date, "found": False, "recommendations": []}
    df = pd.read_csv(pred_file)
    if "market_odds" not in df.columns:
        return {"date": target_date, "found": True, "has_odds": False, "recommendations": []}
    for column in ("market_odds", "win_probability"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["market_odds", "win_probability"])
    df = df[(df["market_odds"] > 1) & df["win_probability"].between(0, 1)]
    df["edge"] = df["win_probability"] - (1.0 / df["market_odds"])
    df = df[df["edge"] >= min_edge]
    recs = []
    total_outlay = 0.0
    total_expected = 0.0
    for _, row in df.iterrows():
        stake_frac = kelly_stake_fraction(row["win_probability"], row["market_odds"], frac)
        stake = round(bankroll * stake_frac, 2)
        expected = round(stake * (row["win_probability"] * (row["market_odds"] - 1) - (1 - row["win_probability"])), 2)
        total_outlay += stake
        total_expected += expected
        recs.append({
            "horse": None if pd.isna(row.get("horse")) else row.get("horse"),
            "race_time": None if pd.isna(row.get("race_time")) else row.get("race_time"),
            "course": None if pd.isna(row.get("course")) else row.get("course"),
            "win_probability": float(row["win_probability"]),
            "market_odds": float(row["market_odds"]),
            "edge": float(row["edge"]),
            "stake": stake,
            "expected_profit": expected,
        })
    recs.sort(key=lambda r: r["edge"], reverse=True)
    return {
        "date": target_date,
        "found": True,
        "has_odds": True,
        "value_bets": len(recs),
        "total_outlay": round(total_outlay, 2),
        "expected_profit": round(total_expected, 2),
        "expected_roi_pct": round(100 * total_expected / total_outlay, 1) if total_outlay else 0.0,
        "recommendations": recs[:50],
    }


# ── performance series (precomputed on each request from history CSV) ────────

@transaction
def performance() -> dict[str, Any]:
    df = load_history()
    state = load_state()
    settled = df[df["result"].isin(["Won", "Lost"])].copy()
    out: dict[str, Any] = {"state": state, "total_bets": int(len(df)), "pending": int((df["result"] == "Pending").sum()) if len(df) else 0}
    if settled.empty:
        out.update({"series": [], "totals": {"bets": 0, "win_rate": None, "pnl": 0.0, "roi_pct": None}})
        return out

    settled["bet_date"] = pd.to_datetime(settled["bet_date"], errors="coerce")
    settled = settled.sort_values("bet_date")
    settled["profit"] = pd.to_numeric(settled["profit"], errors="coerce").fillna(0.0)
    settled["stake"] = pd.to_numeric(settled["stake"], errors="coerce").fillna(0.0)
    settled["cum_pnl"] = settled["profit"].cumsum()
    settled["cum_staked"] = settled["stake"].cumsum()
    settled["cum_roi_pct"] = np.where(settled["cum_staked"] > 0, settled["cum_pnl"] / settled["cum_staked"] * 100, 0.0)
    daily = settled.groupby(settled["bet_date"].dt.date).agg(profit=("profit", "sum"), staked=("stake", "sum")).reset_index()
    daily["rolling_roi_pct"] = np.where(daily["staked"] > 0, daily["profit"].rolling(30, min_periods=1).sum() / daily["staked"].rolling(30, min_periods=1).sum() * 100, 0.0)

    wins = int((settled["result"] == "Won").sum())
    out.update({
        "totals": {
            "bets": int(len(settled)),
            "wins": wins,
            "win_rate": wins / len(settled) if len(settled) else None,
            "pnl": float(settled["profit"].sum()),
            "roi_pct": float(settled["profit"].sum() / settled["stake"].sum() * 100) if settled["stake"].sum() else None,
        },
        "series": [
            {"date": str(r["bet_date"].date()), "cum_pnl": round(float(r["cum_pnl"]), 2), "cum_roi_pct": round(float(r["cum_roi_pct"]), 2)}
            for _, r in settled.iterrows()
        ],
        "rolling": [
            {"date": str(r["bet_date"]), "rolling_roi_pct": round(float(r["rolling_roi_pct"]), 2)}
            for _, r in daily.iterrows()
        ],
    })
    return out


@transaction
def log_bet(payload: dict[str, Any]) -> dict[str, Any]:
    df = load_history()
    state = load_state()
    stake = float(payload["stake"])
    new_row = {
        "bet_date": payload["bet_date"],
        "race_id": payload.get("race_id", ""),
        "horse_name": payload["horse_name"],
        "win_probability": float(payload.get("win_probability") or 0),
        "market_odds": float(payload["market_odds"]),
        "stake": stake,
        "result": payload.get("result", "Pending"),
        "profit": 0.0,
    }
    if new_row["result"] == "Won":
        new_row["profit"] = round(stake * (float(payload["market_odds"]) - 1), 2)
    elif new_row["result"] == "Lost":
        new_row["profit"] = -stake
    new_df = pd.DataFrame([new_row], columns=HISTORY_COLUMNS)
    df = new_df if df.empty else pd.concat([df, new_df], ignore_index=True)

    state["total_bets"] = int(state.get("total_bets", 0)) + 1
    state["total_staked"] = float(state.get("total_staked", 0.0)) + stake
    if new_row["result"] == "Pending":
        state["current_bankroll"] = float(state.get("current_bankroll", 0.0)) - stake
    else:
        state["current_bankroll"] = float(state.get("current_bankroll", 0.0)) + new_row["profit"]
        state["total_pnl"] = float(state.get("total_pnl", 0.0)) + new_row["profit"]
        if new_row["result"] == "Won":
            state["total_wins"] = int(state.get("total_wins", 0)) + 1
    state["peak_bankroll"] = max(float(state.get("peak_bankroll", 0.0)), float(state["current_bankroll"]))
    save_history(df)
    save_state(state)
    return {"ok": True, "index": len(df) - 1, "state": state}


@transaction
def settle_bet(index: int, result: str) -> dict[str, Any]:
    if result not in ("Won", "Lost"):
        return {"ok": False, "error": "result must be Won or Lost"}
    df = load_history()
    if index < 0 or index >= len(df):
        return {"ok": False, "error": "bet index out of range"}
    row = df.iloc[index]
    if row["result"] != "Pending":
        return {"ok": False, "error": "bet already settled"}
    stake = float(row["stake"])
    profit = round(stake * (float(row["market_odds"]) - 1), 2) if result == "Won" else -stake
    df.at[index, "result"] = result
    df.at[index, "profit"] = profit

    state = load_state()
    state["current_bankroll"] = float(state.get("current_bankroll", 0.0)) + stake + profit
    state["total_pnl"] = float(state.get("total_pnl", 0.0)) + profit
    if result == "Won":
        state["total_wins"] = int(state.get("total_wins", 0)) + 1
    state["peak_bankroll"] = max(float(state.get("peak_bankroll", 0.0)), float(state["current_bankroll"]))
    save_history(df)
    save_state(state)
    return {"ok": True, "profit": profit, "state": state}
