# Horse Racing Predictions — Architecture

## Overview
ML-based horse racing outcome predictor. Data pipeline fetches racecards from The Racing API, engineers features with expanding-window career stats (no lookahead bias), trains an XGBoost ensemble, and generates daily predictions. A multi-page Streamlit app displays results.

## Data Flow
```
The Racing API (racecards, results)    The Odds API (market odds)
        ↓                                       ↓
scripts/fetch_racecards.py           scripts/fetch_odds.py
        ↓                                       ↓
    data/raw/racecards_YYYY-MM-DD.json      data/raw/
        ↓
scripts/phase2_score_races.py
    → race profitability scoring (0–100)
        ↓
scripts/build_engineered_dataset.py
    → data/processed/race_scores_engineered.parquet
        ↓
scripts/ensemble_model.py
    → models/ensemble_model.pkl (XGBoost stack, AUC ~0.69)
        ↓
scripts/predict_todays_races.py
    → data/processed/predictions_YYYY-MM-DD.csv
        ↓
predictions.py (Streamlit entry)
        ↓
scripts/export_best_bets.py → data_files/best_bets_today.json
```

## ML Model
- **Phase 1**: Data cleaning (630K races → 245K after Class 5–7 filter)
- **Phase 2**: Race profitability scoring (class, prize, course tier, field size)
- **Phase 3**: XGBoost classifier, 75 engineered features, ensemble AUC 0.6892
  - Career expanding-window stats (always `shift(1)` before cumulative ops)
  - Going preferences, OR context, pedigree, jockey/trainer features
  - Weight features for handicap races: `weight_lbs`, `weight_vs_avg`, `is_top_weight`
- **Phase 4**: Kelly criterion bet sizing + value detection vs. market odds

## Critical: Leakage Prevention
1. Sort by `[group_var, date]` before any rolling/expanding calculation
2. Use `shift(1)` to exclude current race from all windows
3. Re-sort after any filtering operation
4. Run `scripts/verify_no_leakage.py` after adding features
5. Always use temporal split for evaluation (never random split)

## API Integrations
| Source | Purpose | Auth | Limit |
|--------|---------|------|-------|
| The Racing API | Racecards, results | HTTP Basic (`RACING_API_USERNAME`, `RACING_API_PASSWORD`) | 500 calls/month |
| The Odds API | Market odds | `?apiKey=ODDS_API_KEY` | 500 calls/month |

Both APIs: cache aggressively in `data/raw/`, never call live in tests.

## Key Components
- `predictions.py` — entry, Today/Tomorrow predictions, Model Insights tabs
- `pages/data_explorer.py` — historical data explorer (horses, courses, jockeys)
- `shared/utils.py` — `load_model()`, `load_data()`, `get_dataframe_height()`
- `scripts/predict_todays_races.py` — single-day prediction runner
- `scripts/rl_bankroll_manager.py` — bankroll state + history reports
- `scripts/backtest_walk_forward.py` — walk-forward backtesting
- `scripts/export_best_bets.py` — writes `data_files/best_bets_today.json`

## Odds Conversion
- `scripts/odds_converter.py` — decimal ↔ fractional ↔ American
- Edge = model implied probability − bookmaker implied probability
- Value bet threshold (EV_THRESHOLD): 0.02 (2% minimum edge)

## Storage
- `data/raw/` — cached API JSON responses
- `data/processed/` — Parquet datasets + prediction CSVs
- `models/` — `.pkl` model artifacts (gitignored)
- `tests/fixtures/` — saved API responses for offline testing
