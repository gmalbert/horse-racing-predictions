# Horse Racing Predictions - AI Coding Agent Instructions

## Project Overview
Python ML project predicting UK horse racing outcomes using The Racing API (HTTP Basic Auth: `RACING_API_USERNAME`, `RACING_API_PASSWORD`) and The Odds API (key: `ODDS_API_KEY`). Phases: data cleaning (630K→245K races), race scoring (0-100), XGBoost win prediction (ROC AUC 0.671), betting strategy.

## Environment Setup
- Virtual env: `.venv/` (activate: `.venv\Scripts\Activate.ps1`)
- Install: `pip install -r requirements.txt`
- Secrets: `.env` (never commit; update `.env.example`)

## Key Patterns
- **API Limits**: 500 calls/month each; cache responses in `data/raw/`, use offline data in tests
- **Feature Engineering**: Use `groupby().cumsum().shift(1)` for expanding windows to avoid lookahead bias
- **Testing**: `pytest` with fixtures; mock `requests.get` via `tests/conftest.py`
- **Data Flow**: Raw API → `data/raw/` → Processed Parquet in `data/processed/` → Models in `models/` (gitignored)
- **UI**: Streamlit app (`predictions.py`); replace `use_container_width=True` with `width='stretch'` (newer versions)

## Developer Workflows
- Run UI: `streamlit run predictions.py`
- Generate predictions: `python scripts/predict_todays_races.py [--date YYYY-MM-DD]`
- Test: `pytest tests/`
- API verify: `python examples/api_example.py`

## Architecture
- **Scripts**: Data processing in `scripts/` (phase1-3), predictions in `predict_todays_races.py`
- **UI Features**: Today/Tomorrow predictions, Exacta/Trifecta estimates, cumulative probabilities (Top 2/3 %)
- **Odds**: Convert probabilities to decimal/fractional via `scripts/odds_converter.py`; value bet when model odds < bookmaker odds

## Files to Consult
- Patterns: `examples/api_example.py`, `scripts/phase3_build_horse_model.py`
- Conventions: `README.md`, `requirements.txt`
- UI: `predictions.py` (timezone via `APP_TIMEZONE` env var)

Conserve API usage; cache aggressively. Ask owner before live pulls.