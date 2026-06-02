"""
Export daily best bets for the Sports Picks Grid aggregator.

Reads the most recent data/processed/predictions_YYYY-MM-DD.csv, converts
fractional win odds to implied probability, computes edge vs. model
win_probability, and writes data_files/best_bets_today.json.

Usage:
    python scripts/export_best_bets.py              # uses today's date
    python scripts/export_best_bets.py --date 2026-05-19
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
DATA_FILES = ROOT / "data_files"

SPORT = "Horse Racing"
MODEL_VERSION = "1.0.0"
EV_THRESHOLD = 0.02   # minimum edge to include (horse racing edges are typically smaller)
TOP_N_PER_RACE = 3    # max picks exported per race


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_from_edge(edge: float) -> str:
    if edge >= 0.08:
        return "Elite"
    if edge >= 0.04:
        return "Strong"
    if edge >= 0.03:
        return "Good"
    return "Standard"


def _fractional_to_decimal(frac: str) -> float | None:
    """'3/1' → 4.0, '2/5' → 1.4, '10' → 11.0 (evens-style)."""
    try:
        frac = str(frac).strip()
        if "/" in frac:
            num, den = frac.split("/", 1)
            return float(num) / float(den) + 1.0
        # plain integer (evs style)
        return float(frac) + 1.0
    except Exception:
        return None


def _decimal_to_american(dec: float) -> int | None:
    try:
        if dec <= 1.0:
            return None
        if dec >= 2.0:
            return int(round((dec - 1) * 100))
        return int(round(-100 / (dec - 1)))
    except Exception:
        return None


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _find_predictions_csv(for_date: date) -> Path | None:
    """Return today's predictions CSV, or the most recent one if today missing."""
    today_path = DATA_PROC / f"predictions_{for_date}.csv"
    if today_path.exists():
        return today_path
    # Fall back to most recent
    files = sorted(DATA_PROC.glob("predictions_*.csv"), reverse=True)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(for_date: date | None = None) -> None:
    if for_date is None:
        for_date = date.today()

    csv_path = _find_predictions_csv(for_date)
    if csv_path is None:
        print(f"[horse-racing export] No predictions CSV found — writing empty bets")
        _write([], "no predictions CSV found")
        return

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[horse-racing export] Failed to read {csv_path}: {e}")
        _write([], f"CSV read error: {e}")
        return

    required = {"horse", "win_probability", "win_odds_fractional", "course", "race_time"}
    missing = required - set(df.columns)
    if missing:
        print(f"[horse-racing export] Missing columns {missing} — writing empty bets")
        _write([], f"missing columns: {missing}")
        return

    bets: list[dict] = []

    # Group by race (course + race_time)
    group_cols = [c for c in ("course", "race_time", "race_name", "date") if c in df.columns]
    for race_key, race_df in df.groupby(group_cols):
        # Sort by win_probability descending, take top N
        race_df = race_df.sort_values("win_probability", ascending=False).head(TOP_N_PER_RACE)

        for _, row in race_df.iterrows():
            horse = str(row.get("horse", "")).strip()
            model_prob = _safe_float(row.get("win_probability", 0))
            frac_odds = str(row.get("win_odds_fractional", "")).strip()
            course = str(row.get("course", "")).strip()
            race_time = str(row.get("race_time", "")).strip()
            race_name = str(row.get("race_name", course)).strip()
            row_date = str(row.get("date", str(for_date)))[:10]

            if not horse or model_prob <= 0:
                continue

            dec_odds = _fractional_to_decimal(frac_odds)
            if dec_odds is None or dec_odds <= 1.0:
                continue

            market_implied = 1.0 / dec_odds
            edge = model_prob - market_implied

            if edge < EV_THRESHOLD:
                continue

            american_odds = _decimal_to_american(dec_odds)
            tier = _tier_from_edge(edge)

            game_label = f"{race_name} ({course})" if race_name != course else course
            extra_parts = []
            for col in ("jockey", "trainer", "distance_f", "race_class"):
                if col in row and pd.notna(row[col]):
                    extra_parts.append(f"{col}: {row[col]}")
            notes = "  |  ".join(extra_parts) if extra_parts else ""

            bets.append(
                {
                    "game_date": row_date,
                    "game": game_label,
                    "game_time": race_time or None,
                    "bet_type": "win",
                    "pick": horse,
                    "confidence": round(min(max(model_prob, 0.0), 1.0), 4),
                    "edge": round(edge, 4),
                    "odds": american_odds,
                    "tier": tier,
                    "notes": notes,
                }
            )

    bets.sort(key=lambda b: b["edge"], reverse=True)
    _write(bets)


def _write(bets: list[dict], notes: str = "") -> None:
    DATA_FILES.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": str(date.today().year),
            "notes": notes,
        },
        "bets": bets,
    }
    out = DATA_FILES / "best_bets_today.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[horse-racing export] Wrote {len(bets)} bets → {out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD override (default: today)")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else None
    export(for_date=run_date)
