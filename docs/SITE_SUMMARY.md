> **AI Onboarding Guide** — See also `.github/copilot-instructions.md` for full coding conventions.

# Horse Racing Predictions — Site Summary

## What This App Does

Multi-page Streamlit app for horse racing predictions (UK/Ireland). Uses XGBoost models trained on Racing API historical racecards with 75 engineered features including career stats, going preferences, jockey/trainer performance, and weight data. Generates predictions for today's and tomorrow's races with Kelly-criterion bet sizing.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Fetch today's racecards and generate predictions
python scripts/predict_todays_races.py --date YYYY-MM-DD

# 3. Run the app
streamlit run predictions.py
```

For a full data rebuild:
```bash
python scripts/build_engineered_dataset.py   # Build feature matrix
python scripts/ensemble_model.py             # Train stacked ensemble
python scripts/backtest_walk_forward.py --folds 6   # Validate
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (main `predictions.py` + `pages/data_explorer.py`) |
| ML | XGBoost ensemble + stacking (`scripts/ensemble_model.py`) |
| Data source | The Racing API (HTTP Basic Auth, 500 calls/month) |
| Odds | The Odds API |
| Data storage | Parquet (primary), JSON (raw racecards) |
| Backtesting | Walk-forward (`scripts/backtest_walk_forward.py`) |
| Bankroll | Kelly Criterion (`scripts/rl_bankroll_manager.py`) |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Main Streamlit page (733 lines) — Today/Tomorrow predictions, Model Insights |
| `pages/data_explorer.py` | Historical data explorer (767 lines) — Horses, Courses, Jockeys, Overall |
| `shared/utils.py` | Common utilities — `load_model()`, `load_data()`, `get_dataframe_height()` |
| `scripts/predict_todays_races.py` | Single-day prediction runner |
| `scripts/batch_generate_predictions.py` | Batch runner — scans `data/raw/` for all racecards |
| `scripts/phase3_build_horse_model.py` | Core XGBoost training with 75 features |
| `scripts/ensemble_model.py` | Stacked ensemble training and calibration |
| `scripts/backtest_walk_forward.py` | Walk-forward backtesting |
| `scripts/rl_bankroll_manager.py` | Bankroll state and betting history reports |
| `examples/api_example.py` | Racing API usage patterns and authentication example |

## Data Flow

1. **Racecards**: `scripts/fetch_racecards.py --date YYYY-MM-DD` → The Racing API → `data/raw/racecards_YYYY-MM-DD.json`
2. **Feature engineering**: expanding-window career stats (no lookahead bias) → `data/processed/race_scores_engineered.parquet`
3. **Training**: `scripts/phase3_build_horse_model.py` → XGBoost (75 features) → `models/*.pkl`
4. **Predictions**: `scripts/predict_todays_races.py` → `data/processed/predictions_YYYY-MM-DD.csv`
5. **UI**: Streamlit reads predictions CSV → renders today/tomorrow picks with confidence and Kelly sizing

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `RACING_API_USERNAME` | The Racing API HTTP Basic Auth | Required |
| `RACING_API_PASSWORD` | The Racing API HTTP Basic Auth | Required |
| `ODDS_API_KEY` | The Odds API — market odds for value bet detection | Optional |

## Critical API Rate Limit

**Both The Racing API and The Odds API are limited to 500 calls/month.** Cache aggressively. Never call live APIs in tests — use fixtures from `data/raw/` instead.

## Data Leakage Prevention

This is critical — all features must use **expanding windows** with `shift(1)` to prevent lookahead bias:
1. Sort by `[horse/jockey/trainer, date]` before any cumulative calculation
2. Use `shift(1)` to exclude the current race
3. After filtering (e.g., by going type), re-sort before rolling calculations
4. Run `python scripts/verify_no_leakage.py` after adding any new feature

## Common Gotchas

- Use `width='stretch'` for all charts/dataframes — `use_container_width` is removed in newer Streamlit
- `APP_TIMEZONE` env var controls which timezone is used for "today"/"tomorrow" determination
- Tests must not make live API calls — monkeypatch `requests.get` in `tests/conftest.py`
- The `data_explorer.py` is 767 lines — large changes should be tested thoroughly
