# Horse Racing Predictions — 12-Month Feature Roadmap

> Generated: 2026-07-31 | Horizon: August 2026 – July 2027

---

## Executive Summary

This roadmap expands the horse racing platform from a UK-focused XGBoost classifier into
a full-coverage, multi-jurisdiction platform spanning UK, Ireland, US, France, and
Australia. Key investments: real-time speed figure modeling, breeding analytics,
jockey form surfaces, and an automated nightly staking engine.

---

## Q1 (Aug–Oct 2026) — US Expansion & Speed Figures

### Feature 1 — Beyer Speed Figure Integration

Ingest DRF Beyer Speed Figures via Equibase CSV exports. Use speed figures
as the primary pace feature for US dirt/turf races.

```python
# scripts/us_speed_figures.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def parse_equibase_speed_figures(csv_path: str) -> pd.DataFrame:
    """Parse Equibase past performance CSV for Beyer figures."""
    df = pd.read_csv(csv_path, low_memory=False)
    # Typical Equibase column mapping
    speed_df = df.rename(columns={
        "HORSE_NAME": "horse",
        "RACE_DATE": "race_date",
        "TRACK": "track",
        "BEYER_SPEED": "beyer_speed",
        "DISTANCE_FURLONGS": "distance_f",
        "SURFACE": "surface",
        "FINAL_POSITION": "position",
    })[["horse", "race_date", "track", "beyer_speed", "distance_f", "surface", "position"]]
    speed_df = speed_df[speed_df["beyer_speed"].notna()]
    speed_df["beyer_speed"] = pd.to_numeric(speed_df["beyer_speed"], errors="coerce")
    return speed_df

def build_speed_figure_features(df: pd.DataFrame, horse: str, n: int = 5) -> dict:
    """Rolling Beyer speed figure features for a horse."""
    recent = (
        df[df["horse"] == horse]
        .sort_values("race_date", ascending=False)
        .head(n)
    )
    if recent.empty:
        return {"avg_beyer_l5": None, "best_beyer_l5": None, "beyer_trend": None}
    return {
        "avg_beyer_l5": recent["beyer_speed"].mean(),
        "best_beyer_l5": recent["beyer_speed"].max(),
        "beyer_trend": recent["beyer_speed"].diff(-1).mean(),  # positive = improving
        "beyer_consistency": recent["beyer_speed"].std(),
    }
```

### Feature 2 — Breeding Sire/Dam Analytics

Build a sire × distance × surface performance database. Identify which
sires produce winners at specific distance/surface combinations.

```python
# scripts/breeding_analytics.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def build_sire_profile(results: pd.DataFrame) -> pd.DataFrame:
    """Compute sire performance by distance bracket and surface."""
    df = results.copy()
    df["distance_bracket"] = pd.cut(
        df["distance_f"],
        bins=[0, 6, 8, 10, 14, 20],
        labels=["Sprint", "Mile", "9-10f", "11-12f", "Staying"],
    )
    sire_stats = (
        df.groupby(["sire", "distance_bracket", "surface"])
        .agg(
            runs=("horse", "count"),
            wins=("position", lambda x: (x == 1).sum()),
            places=("position", lambda x: (x <= 3).sum()),
            avg_beyer=("beyer_speed", "mean"),
        )
        .reset_index()
    )
    sire_stats["win_rate"] = sire_stats["wins"] / sire_stats["runs"].clip(lower=1)
    sire_stats["place_rate"] = sire_stats["places"] / sire_stats["runs"].clip(lower=1)
    sire_stats = sire_stats[sire_stats["runs"] >= 10]
    out = DATA_DIR / "processed/sire_profiles.parquet"
    sire_stats.to_parquet(out, index=False)
    return sire_stats

def get_sire_advantage(
    sire: str, distance_bracket: str, surface: str,
    sire_df: pd.DataFrame, league_avg_win_rate: float = 0.125,
) -> float:
    """Return sire win rate vs league average for this distance/surface."""
    row = sire_df[
        (sire_df["sire"] == sire) &
        (sire_df["distance_bracket"] == distance_bracket) &
        (sire_df["surface"] == surface)
    ]
    if row.empty:
        return 0.0
    return row.iloc[0]["win_rate"] - league_avg_win_rate
```

### Feature 3 — Jockey Agent / Yard Intelligence

Correlate jockey-trainer combinations to compute partnership win rates.
Identify hot stable / jockey pairings and flag cold-stable switches.

```python
# scripts/jockey_trainer.py
import pandas as pd

def build_jockey_trainer_partnerships(results: pd.DataFrame) -> pd.DataFrame:
    """Win rate per jockey-trainer combination."""
    combos = (
        results.groupby(["jockey", "trainer"])
        .agg(
            runs=("horse", "count"),
            wins=("win", "sum"),
            last_winner_date=("race_date", lambda x: x[results.loc[x.index, "win"] == 1].max()),
        )
        .reset_index()
    )
    combos["win_rate"] = combos["wins"] / combos["runs"].clip(lower=1)
    combos["is_hot"] = (combos["win_rate"] > 0.25) & (combos["runs"] >= 10)
    return combos.sort_values("win_rate", ascending=False)

def detect_yard_form(results: pd.DataFrame, trainer: str, days: int = 14) -> str:
    """Classify trainer as 'hot', 'cold', or 'neutral' over last N days."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = results[(results["trainer"] == trainer) & (results["race_date"] >= cutoff)]
    if recent.empty:
        return "neutral"
    win_rate = recent["win"].mean()
    if win_rate > 0.30:
        return "hot"
    elif win_rate < 0.05:
        return "cold"
    return "neutral"
```

### Feature 4 — Going Preference Model (Machine Learning)

Train a per-horse model predicting performance improvement/decline based
on ground conditions (Firm → Soft → Heavy transitions).

```python
# scripts/going_preference.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

GOING_CODES = {
    "Firm": 0, "Good to Firm": 1, "Good": 2, "Good to Soft": 3,
    "Soft": 4, "Heavy": 5, "Standard": 2, "Fast": 0, "Yielding": 4,
}

def compute_going_preference_features(
    results: pd.DataFrame, horse: str, current_going: str
) -> dict:
    """Return historical performance on similar going types."""
    horse_races = results[results["horse"] == horse].copy()
    horse_races["going_code"] = horse_races["going"].map(GOING_CODES).fillna(2)
    current_code = GOING_CODES.get(current_going, 2)

    # Races on similar going (within 1 code)
    similar = horse_races[abs(horse_races["going_code"] - current_code) <= 1]
    firm = horse_races[horse_races["going_code"] <= 1]
    soft = horse_races[horse_races["going_code"] >= 4]

    return {
        "win_rate_similar_going": similar["win"].mean() if len(similar) > 0 else None,
        "win_rate_firm": firm["win"].mean() if len(firm) >= 3 else None,
        "win_rate_soft": soft["win"].mean() if len(soft) >= 3 else None,
        "going_preference_code": _infer_preference(horse_races),
        "going_mismatch_flag": _detect_mismatch(horse_races, current_code),
    }

def _infer_preference(df: pd.DataFrame) -> str:
    """Determine preferred going by win rate."""
    df = df.copy()
    df["going_code"] = df["going"].map(GOING_CODES).fillna(2)
    wr = df.groupby("going_code")["win"].mean()
    if wr.empty:
        return "any"
    best_code = wr.idxmax()
    if best_code <= 1: return "fast"
    if best_code >= 4: return "soft"
    return "good"

def _detect_mismatch(df: pd.DataFrame, target_code: int) -> bool:
    """True if horse has never won on going this soft/firm."""
    df = df.copy()
    df["going_code"] = df["going"].map(GOING_CODES).fillna(2)
    winners = df[df["win"] == 1]
    if winners.empty:
        return False
    worst_won = winners["going_code"].max()
    return target_code > worst_won + 1
```

### Feature 5 — Automated Nightly Prediction Generation

GitHub Action fetches racecards at 6 PM UTC daily. Runs predictions for
next day. Emails a formatted picks report by 7 PM UTC.

```yaml
# .github/workflows/nightly_predictions.yml
name: Horse Racing Nightly Predictions
on:
  schedule:
    - cron: '0 18 * * *'   # 6 PM UTC
  workflow_dispatch:

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - name: Fetch tomorrow's racecards
        env:
          RACING_API_USERNAME: ${{ secrets.RACING_API_USERNAME }}
          RACING_API_PASSWORD: ${{ secrets.RACING_API_PASSWORD }}
        run: python scripts/fetch_racecards.py --date tomorrow
      - name: Generate predictions
        run: python scripts/predict_todays_races.py --date tomorrow
      - name: Send email report
        env:
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
        run: python scripts/send_picks_email.py --date tomorrow
      - name: Commit prediction artifacts
        run: |
          git config user.name "github-actions"
          git add data/processed/predictions_*.csv
          git commit -m "Auto: predictions for $(date -d tomorrow +%Y-%m-%d)" || echo "No changes"
          git push
```

---

## Q2 (Nov 2026 – Jan 2027) — Value Betting & Kelly Engine

### Feature 6 — Kelly Criterion Staking Plan

Implement full Kelly, half Kelly, and fractional Kelly stake sizing.
Account for model edge, bettor bankroll, and maximum bet cap.

```python
# scripts/kelly_staking.py
import pandas as pd
import numpy as np

def kelly_stake(
    win_prob: float, decimal_odds: float,
    fraction: float = 0.5, bankroll: float = 1000.0,
    max_bet_pct: float = 0.05,
) -> dict:
    """Compute Kelly stake recommendation."""
    b = decimal_odds - 1  # net odds
    q = 1 - win_prob
    kelly_full = (win_prob * b - q) / b
    kelly_fraction = kelly_full * fraction
    stake = bankroll * max(0, kelly_fraction)
    stake = min(stake, bankroll * max_bet_pct)  # cap at max_bet_pct
    return {
        "full_kelly_pct": round(kelly_full, 4),
        "fractional_kelly_pct": round(kelly_fraction, 4),
        "recommended_stake": round(stake, 2),
        "ev": round(win_prob * b - q, 4),
        "profitable": kelly_fraction > 0,
    }

def apply_kelly_to_predictions(
    predictions: pd.DataFrame,
    bankroll: float = 1000.0,
    fraction: float = 0.5,
) -> pd.DataFrame:
    """Add Kelly stake columns to predictions DataFrame."""
    results = []
    for _, row in predictions.iterrows():
        dec_odds = 1 / row.get("dk_implied_prob", 0.1) if row.get("dk_implied_prob", 0) > 0 else 4.0
        kelly = kelly_stake(
            row.get("model_win_prob", 0.1), dec_odds, fraction, bankroll
        )
        results.append({**row.to_dict(), **kelly})
    return pd.DataFrame(results)
```

### Feature 7 — Class Movement Feature Engineering

Horses moving up/down in class is one of the strongest signals.
Compute class relative to race history and flag class jump/drop.

```python
# scripts/class_movement.py
import pandas as pd

CLASS_HIERARCHY = {
    "Group 1": 10, "Group 2": 9, "Group 3": 8,
    "Listed": 7, "Handicap Class 1": 6, "Handicap Class 2": 5,
    "Handicap Class 3": 4, "Handicap Class 4": 3,
    "Maiden": 2, "Claiming": 1,
}

def compute_class_movement(horse: str, current_race_class: str,
                             history: pd.DataFrame) -> dict:
    """Detect class rise/drop and compute win rate at this class level."""
    current_code = CLASS_HIERARCHY.get(current_race_class, 5)
    horse_history = history[history["horse"] == horse].copy()
    if horse_history.empty:
        return {"class_movement": 0, "win_rate_at_class": None}

    horse_history["class_code"] = horse_history["race_class"].map(CLASS_HIERARCHY).fillna(5)
    last_class = horse_history.sort_values("race_date").iloc[-1]["class_code"]

    at_class = horse_history[horse_history["class_code"] == current_code]
    win_rate = at_class["win"].mean() if len(at_class) > 0 else None

    return {
        "class_movement": current_code - last_class,  # positive = class rise
        "win_rate_at_class": win_rate,
        "class_drop": current_code < last_class,
        "class_rise": current_code > last_class,
        "previous_class_code": last_class,
        "current_class_code": current_code,
    }
```

### Feature 8 — Draw Bias (Barrier Analysis)

Compute historical win rates by draw (barrier position) for each track
and distance combination. Flag favorable/unfavorable draws.

```python
# scripts/draw_bias.py
import pandas as pd
import numpy as np

def build_draw_bias_table(results: pd.DataFrame) -> pd.DataFrame:
    """Win rate by draw position for each track × distance bracket."""
    df = results.copy()
    df["distance_bracket"] = pd.cut(
        df["distance_f"], bins=[0, 6, 8, 10, 14, 20],
        labels=["Sprint", "Mile", "9-10f", "11-12f", "Staying"]
    )
    draw_stats = (
        df.groupby(["track", "distance_bracket", "draw"])
        .agg(runs=("horse", "count"), wins=("win", "sum"))
        .reset_index()
    )
    draw_stats["win_rate"] = draw_stats["wins"] / draw_stats["runs"].clip(lower=1)
    # League average per track/distance
    avg = draw_stats.groupby(["track", "distance_bracket"])["win_rate"].transform("mean")
    draw_stats["draw_edge"] = draw_stats["win_rate"] - avg
    return draw_stats

def get_draw_advantage(
    track: str, distance_bracket: str, draw: int,
    bias_table: pd.DataFrame,
) -> float:
    """Return draw edge vs field average (+ = favorable, - = unfavorable)."""
    row = bias_table[
        (bias_table["track"] == track) &
        (bias_table["distance_bracket"] == distance_bracket) &
        (bias_table["draw"] == draw)
    ]
    return row.iloc[0]["draw_edge"] if not row.empty else 0.0
```

### Feature 9 — Live Scratchings & Late Changes Handler

Monitor racecards for late scratchings via Racing API webhooks (or polling).
Re-rank predictions when a horse is scratched from the field.

```python
# scripts/handle_scratchings.py
import requests, pandas as pd
from pathlib import Path
import os

def fetch_final_runners(race_id: str) -> list[str]:
    """Fetch confirmed runners for a race just before off-time."""
    resp = requests.get(
        "https://api.theracingapi.com/v1/runners",
        params={"race_id": race_id},
        auth=(os.environ["RACING_API_USERNAME"], os.environ["RACING_API_PASSWORD"]),
        timeout=10,
    )
    return [r["horse"] for r in resp.json().get("runners", []) if r.get("status") == "Active"]

def rerank_after_scratching(
    predictions: pd.DataFrame, scratched_horse: str
) -> pd.DataFrame:
    """Remove scratched horse and normalize remaining win probabilities."""
    preds = predictions[predictions["horse"] != scratched_horse].copy()
    if preds.empty:
        return preds
    removed_prob = predictions[predictions["horse"] == scratched_horse]["win_prob"].sum()
    # Redistribute probability proportionally
    preds["win_prob"] = preds["win_prob"] + (preds["win_prob"] / preds["win_prob"].sum()) * removed_prob
    preds["win_rank"] = preds["win_prob"].rank(ascending=False).astype(int)
    return preds.sort_values("win_rank")
```

### Feature 10 — Multi-Track Arbitrage Finder

When the same horse runs in different jurisdictions at overlapping times,
find the best odds across books. Detect any cross-market arb opportunities.

```python
# scripts/cross_book_odds.py
import pandas as pd, requests, os
from itertools import combinations

BOOKS_CHECKED = ["betfair", "bet365", "william_hill", "draftkings", "paddypower"]

def find_best_odds_per_horse(race_id: str, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Find best available odds for each horse across all books."""
    best = (
        odds_df[odds_df["race_id"] == race_id]
        .groupby("horse")
        .apply(lambda g: g.loc[g["decimal_odds"].idxmax()])
        .reset_index(drop=True)
    )
    return best[["horse", "book", "decimal_odds", "fractional_odds"]]

def check_arb_opportunity(odds_df: pd.DataFrame, race_id: str) -> dict | None:
    """Check if arbitrage exists across books for a race."""
    race_odds = odds_df[odds_df["race_id"] == race_id]
    best_per_horse = find_best_odds_per_horse(race_id, race_odds)
    implied_sum = (1 / best_per_horse["decimal_odds"]).sum()
    if implied_sum < 1.0:
        profit_pct = (1 - implied_sum) * 100
        return {
            "race_id": race_id,
            "arb_margin": round(profit_pct, 2),
            "horses": best_per_horse.to_dict("records"),
        }
    return None
```

---

## Q3 (Feb–Apr 2027) — Analytics & Dashboard

### Feature 11 — Course & Distance Winner Database

Build a database of horses with proven course-and-distance form.
Course-and-distance winners have materially higher win rates.

```python
# scripts/course_distance_db.py
import pandas as pd

def build_cd_winners(results: pd.DataFrame) -> pd.DataFrame:
    """Mark horses with course-and-distance winning form."""
    cd_wins = (
        results[results["win"] == 1]
        .groupby(["horse", "track", "distance_f"])
        .size()
        .reset_index(name="cd_wins")
    )
    results = results.merge(cd_wins, on=["horse", "track", "distance_f"], how="left")
    results["cd_wins"] = results["cd_wins"].fillna(0).astype(int)
    results["has_cd_win"] = results["cd_wins"] > 0
    return results

def get_cd_score(horse: str, track: str, distance_f: float,
                  cd_df: pd.DataFrame) -> int:
    row = cd_df[(cd_df["horse"] == horse) & (cd_df["track"] == track)
                & (abs(cd_df["distance_f"] - distance_f) < 0.5)]
    return int(row["cd_wins"].sum()) if not row.empty else 0
```

### Feature 12 — Odds Movement (Steam Move) Detector

Track opening vs current market odds. Identify steam moves (rapid shortening)
as a signal of informed money. Correlate with model predictions.

```python
# scripts/steam_move_detector.py
import pandas as pd, numpy as np
from datetime import datetime

STEAM_THRESHOLD = -0.20  # >20% odds contraction = steam move

def detect_steam_moves(
    opening_odds: pd.DataFrame, current_odds: pd.DataFrame
) -> pd.DataFrame:
    """Identify horses whose odds have moved significantly since open."""
    merged = opening_odds.merge(
        current_odds, on=["race_id", "horse"], suffixes=("_open", "_current")
    )
    merged["odds_change_pct"] = (
        (merged["decimal_odds_current"] - merged["decimal_odds_open"])
        / merged["decimal_odds_open"]
    )
    steam_moves = merged[merged["odds_change_pct"] < STEAM_THRESHOLD].copy()
    steam_moves["steam_direction"] = "shortening (steam)"
    drifters = merged[merged["odds_change_pct"] > 0.30].copy()
    drifters["steam_direction"] = "drifting (bad sign)"
    return pd.concat([steam_moves, drifters]).sort_values("odds_change_pct")
```

### Feature 13 — Race-Day Dashboard (Live Updates)

Streamlit dashboard updating every 60 seconds on race days. Shows confirmed
runners, current odds, model prediction, and steam moves.

```python
# predictions.py (race day mode)
import streamlit as st
import pandas as pd
import time
from datetime import date
from scripts.predict_todays_races import load_todays_predictions
from scripts.steam_move_detector import detect_steam_moves

def render_race_day_dashboard() -> None:
    st.title(f"🏇 Race Day Dashboard — {date.today().strftime('%B %d, %Y')}")
    preds = load_todays_predictions()
    if preds is None or preds.empty:
        st.warning("No predictions available. Fetch racecards first.")
        return

    # Auto-refresh every 60 seconds
    placeholder = st.empty()
    with placeholder.container():
        for race_id, race_group in preds.groupby("race_id"):
            with st.expander(f"🏟️ {race_group.iloc[0]['venue']} — Race {race_group.iloc[0]['race_num']}"):
                st.dataframe(
                    race_group[["horse", "jockey", "trainer", "weight", "win_prob",
                                "place_prob", "recommended_stake", "tier"]],
                    width="stretch",
                )
    time.sleep(60)
    st.rerun()
```

### Feature 14 — Trainer Win Rate by Month / Season Trend

Identify trainers who start slow (March form) vs those who peak at
specific times of year. Adjust predictions by seasonal trainer form.

```python
# scripts/trainer_seasonality.py
import pandas as pd

def build_trainer_monthly_win_rates(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["race_date"] = pd.to_datetime(results["race_date"])
    results["month"] = results["race_date"].dt.month
    monthly = (
        results.groupby(["trainer", "month"])
        .agg(runs=("horse", "count"), wins=("win", "sum"))
        .reset_index()
    )
    monthly["win_rate"] = monthly["wins"] / monthly["runs"].clip(lower=1)
    # Compute seasonal adjustment vs annual average
    annual_avg = results.groupby("trainer")["win"].mean()
    monthly["annual_avg"] = monthly["trainer"].map(annual_avg)
    monthly["seasonal_adj"] = monthly["win_rate"] - monthly["annual_avg"]
    return monthly

def get_trainer_seasonal_adj(
    trainer: str, month: int, seasonal_df: pd.DataFrame
) -> float:
    row = seasonal_df[
        (seasonal_df["trainer"] == trainer) & (seasonal_df["month"] == month)
    ]
    return row.iloc[0]["seasonal_adj"] if not row.empty else 0.0
```

### Feature 15 — Raceform AI Narrative Generator

Use GPT-4o-mini to auto-generate a human-readable race preview for top picks.
Include form summary, class, going preference, and model edge.

```python
# scripts/ai_preview.py
import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PREVIEW_PROMPT = """
Generate a concise horse racing preview (100 words) for this horse's selection:

Horse: {horse}
Race: {race_name}, {track}, {distance_f}f, {going}
Form: Last 5 races: {form_string}
Trainer: {trainer} ({trainer_strike_rate:.0%} SR)
Jockey: {jockey} ({jockey_strike_rate:.0%} SR)
Model Win Prob: {win_prob:.1%}
Edge vs market: {edge:+.1%}
Key factors: {key_factors}

Style: concise, factual, punter-focused. Mention if course or going specialist.
"""

def generate_race_preview(selection: dict) -> str:
    prompt = PREVIEW_PROMPT.format(**selection)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150, temperature=0.6,
    )
    return resp.choices[0].message.content
```

---

## Q4 (May–Jul 2027) — Advanced ML & International Markets

### Feature 16 — Australian Racing Integration

Add Victorian/NSW racing via TAB/Equibase AU. Separate model calibrated
on Australian form (graded system, distances in metres, softer going scales).

```python
# scripts/australian_racing.py
import requests, pandas as pd, os

AU_TAB_BASE = "https://api.tab.com.au/v1/tab-info-service"

def fetch_au_racecards(date: str) -> pd.DataFrame:
    """Fetch TAB Australia race cards."""
    resp = requests.get(
        f"{AU_TAB_BASE}/racing/dates/{date}/meetings",
        params={"jurisdiction": "VIC"},
        timeout=15,
    )
    meetings = resp.json().get("meetings", [])
    rows = []
    for m in meetings:
        for race in m.get("races", []):
            for runner in race.get("runners", []):
                rows.append({
                    "track": m["meetingName"],
                    "race_num": race["raceNumber"],
                    "horse": runner["runnerName"],
                    "barrier": runner.get("barrier"),
                    "jockey": runner.get("riderDriverName", ""),
                    "trainer": runner.get("trainerName", ""),
                    "weight_kg": runner.get("weightAllocated", 58),
                    "distance_m": race.get("distance", 1200),
                    "race_class": race.get("raceClassConditions", ""),
                })
    return pd.DataFrame(rows)
```

### Feature 17 — Exacta / Trifecta Probability Calculator

Compute exact probability for any exacta or trifecta combination using
model win, place, and show probabilities (conditional sequential sampling).

```python
# scripts/exotic_probs.py
import numpy as np

def exacta_probability(
    horse1: str, horse2: str,
    win_probs: dict[str, float],
) -> float:
    """P(horse1 wins, horse2 finishes 2nd)."""
    p1_win = win_probs.get(horse1, 0)
    p2_win = win_probs.get(horse2, 0)
    p2_second_given_p1_won = p2_win / max(1 - p1_win, 0.001)
    return p1_win * p2_second_given_p1_won

def trifecta_probability(
    h1: str, h2: str, h3: str, win_probs: dict[str, float]
) -> float:
    """P(h1 wins, h2 2nd, h3 3rd)."""
    p1 = win_probs.get(h1, 0)
    remaining_after_1 = {k: v for k, v in win_probs.items() if k != h1}
    total_remaining = sum(remaining_after_1.values())
    p2_given_1 = remaining_after_1.get(h2, 0) / max(total_remaining, 0.001)
    remaining_after_2 = {k: v for k, v in remaining_after_1.items() if k != h2}
    total_rem2 = sum(remaining_after_2.values())
    p3_given_12 = remaining_after_2.get(h3, 0) / max(total_rem2, 0.001)
    return p1 * p2_given_1 * p3_given_12

def top_exacta_bets(
    win_probs: dict[str, float], min_prob: float = 0.02
) -> list[dict]:
    """Return most likely exacta combinations."""
    horses = list(win_probs.keys())
    results = []
    for h1 in horses:
        for h2 in horses:
            if h1 == h2:
                continue
            p = exacta_probability(h1, h2, win_probs)
            if p >= min_prob:
                results.append({"h1": h1, "h2": h2, "probability": round(p, 4)})
    return sorted(results, key=lambda x: -x["probability"])[:10]
```

### Feature 18 — Racecourse Map & Track Profile Visualizer

For each track, display an SVG/Plotly visualization of the course layout,
entry points, and historical advantage zones (inside, outside rail preference).

```python
# pages/track_profiles.py
import streamlit as st, plotly.graph_objects as go, numpy as np

TRACK_PROFILES = {
    "Cheltenham": {"shape": "oval", "length_m": 3218, "rail_bias": "inner",
                   "gradient_m": 65, "ground_type": "turf"},
    "Ascot": {"shape": "triangle", "length_m": 2800, "rail_bias": "outer",
               "gradient_m": 40, "ground_type": "turf"},
    "Kempton": {"shape": "triangle", "length_m": 2000, "rail_bias": "none",
                 "gradient_m": 0, "ground_type": "aw"},
}

def render_track_profile(track: str) -> None:
    profile = TRACK_PROFILES.get(track, {})
    if not profile:
        st.info(f"No profile data for {track}.")
        return
    st.subheader(f"Track Profile: {track}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Length", f"{profile['length_m']}m")
    c2.metric("Gradient", f"{profile['gradient_m']}m climb")
    c3.metric("Rail Bias", profile["rail_bias"].title())
    st.caption(f"Surface: {profile['ground_type'].upper()}")
```

### Feature 19 — Model Confidence Calibration Plot

After settling results, generate probability calibration curves and
ECE (Expected Calibration Error) to monitor model health over time.

```python
# scripts/calibration_monitor.py
import pandas as pd, numpy as np
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)

def plot_calibration(df: pd.DataFrame) -> go.Figure:
    frac_pos, mean_pred = calibration_curve(
        df["win"].values, df["win_prob"].values, n_bins=10, strategy="quantile"
    )
    ece = compute_ece(df["win"].values, df["win_prob"].values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, name="Model",
                              mode="lines+markers", line=dict(color="#2196F3")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect Calibration",
                              mode="lines", line=dict(dash="dash", color="grey")))
    fig.update_layout(title=f"Model Calibration — ECE: {ece:.4f}",
                       xaxis_title="Predicted Probability",
                       yaxis_title="Actual Win Rate", template="plotly_dark")
    return fig
```

### Feature 20 — Jockey Body Weight & Claim Allowance Tracker

Track jockey claim allowances (7 lb, 5 lb, 3 lb). A big claim allowance
effectively lowers weight carried, improving win probability.

```python
# scripts/jockey_claims.py
import pandas as pd

CLAIM_THRESHOLDS = {
    "7lb Claimer": 7, "5lb Claimer": 5, "3lb Claimer": 3,
}

def get_jockey_claim(jockey: str, career_wins: int) -> int:
    """UK/Ireland: riders gain/lose claims based on career wins."""
    if career_wins < 20:
        return 7
    elif career_wins < 50:
        return 5
    elif career_wins < 95:
        return 3
    return 0  # no longer a claimer

def effective_weight(allocated_weight: float, jockey: str,
                      career_wins: int) -> float:
    """Return actual weight after claim allowance."""
    claim = get_jockey_claim(jockey, career_wins)
    return allocated_weight - claim

def weight_to_performance_adjustment(effective_wt: float, std_weight: float = 126.0) -> float:
    """Estimate lap-time equivalent of weight difference (1 lb ≈ 0.3 lengths per mile)."""
    lb_diff = effective_wt - std_weight
    return lb_diff * 0.3  # lengths adjustment
```

### Feature 21 — Risk-Adjusted Performance Metrics

Compute Sharpe ratio and maximum drawdown for staking strategies.
Compare flat stake vs Kelly vs fractional Kelly performance.

```python
# scripts/risk_metrics.py
import numpy as np, pandas as pd

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    return float(excess.mean() / excess.std(ddof=1)) if excess.std() > 0 else 0.0

def max_drawdown(cumulative_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / np.maximum(peak, 1)
    return float(drawdown.min())

def compare_staking_methods(picks: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    """Compare flat, half-Kelly, and full-Kelly over historical picks."""
    results = []
    for method, stake_fn in [
        ("Flat £10", lambda r: 10.0),
        ("1% Bank", lambda r: bankroll * 0.01),
        ("Half Kelly", lambda r: max(0, r["fractional_kelly_pct"]) * 0.5 * bankroll),
    ]:
        cumulative = bankroll
        rets = []
        for _, row in picks.iterrows():
            stake = min(stake_fn(row), bankroll * 0.05)
            if row["win"]:
                pnl = stake * (row["decimal_odds"] - 1)
            else:
                pnl = -stake
            cumulative += pnl
            rets.append(pnl / stake if stake > 0 else 0)
        arr = np.array(rets)
        results.append({
            "method": method,
            "final_bankroll": round(cumulative, 2),
            "sharpe": round(sharpe_ratio(arr), 3),
            "max_drawdown": round(max_drawdown(np.cumsum([bankroll] + list(arr))), 3),
            "roi": round((cumulative - bankroll) / bankroll, 4),
        })
    return pd.DataFrame(results)
```

### Feature 22 — French Racing Integration (PMU / Unibet)

Add Chantilly, Longchamp, Deauville meetings via PMU API.
Separate model weights for French turf dynamics (firmer ground, different weights).

```python
# scripts/french_racing.py
import requests, pandas as pd

PMU_BASE = "https://offline.turf.fr/pmuss/rest"

def fetch_french_programme(date: str) -> pd.DataFrame:
    """Fetch French racing programme from PMU."""
    resp = requests.get(
        f"{PMU_BASE}/programme/{date.replace('-', '')}",
        timeout=15,
    )
    data = resp.json()
    rows = []
    for reunion in data.get("programme", {}).get("reunions", []):
        track = reunion.get("hippodrome", {}).get("libelleLong", "")
        for course in reunion.get("courses", []):
            for participant in course.get("participants", []):
                rows.append({
                    "track": track,
                    "race_num": course.get("numOrdre"),
                    "horse": participant.get("nom", ""),
                    "jockey": participant.get("driver", ""),
                    "weight_kg": participant.get("poidsJockey", 58),
                    "distance_m": course.get("distance", 2000),
                    "going": course.get("terrain", {}).get("libelle", "Bon"),
                    "prize_eur": course.get("montantTotalOffert", 0),
                })
    return pd.DataFrame(rows)
```

---

## Timeline Summary

| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 Aug–Oct 2026 | US expansion | Beyer speed figures, breeding analytics, jockey-trainer partnerships, going model, nightly pipeline |
| Q2 Nov 2026–Jan 2027 | Value betting | Kelly staking, class movement, draw bias, scratchings handler, cross-book arbitrage |
| Q3 Feb–Apr 2027 | Analytics | CD winners, steam moves, race-day dashboard, trainer seasonality, AI previews |
| Q4 May–Jul 2027 | International | Australian racing, exacta/trifecta probabilities, track visualizer, calibration monitor, jockey claims, risk metrics, French racing |
