# Horse Racing Predictions - AI Coding Agent Instructions

## Project Overview

This is a **horse racing predictions** project using Python. Data is sourced from **The Racing API** (api.theracingapi.com) with HTTP Basic Authentication and **The Odds API** for live betting odds.

## Development Environment

- **Python Virtual Environment**: Use `.venv/` for all Python dependencies
- **Activation**: 
  - Windows PowerShell: `.venv\Scripts\Activate.ps1`
  - Windows CMD: `.venv\Scripts\activate.bat`
- **Install deps**: `pip install -r requirements.txt`
- **Environment variables**: Use `python-dotenv` with `.env` file (never commit); update `.env.example` for new vars

## AI Agent Instructions — Horse Racing Predictions

This file contains concise, actionable guidance for AI coding agents working in this repo.

## Quick context
- Purpose: ML project to predict horse racing outcomes using The Racing API and The Odds API for value betting
- Repo layout: `examples/`, `data/{raw,processed}`, `models/`, `scripts/`, `tests/`, `src/` (future)
- Current phase: Phase 3 (ML model) complete, Phase 4 (betting strategy) in progress

## Environment & setup (explicit)
- Use virtualenv at `.venv/` and add any new deps to `requirements.txt`
- Windows PowerShell activation: `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Use `python-dotenv` and `.env` for secrets; update `.env.example` if adding new env vars

## API integration (concrete)
- **The Racing API**: HTTP Basic Auth. Credentials: `RACING_API_USERNAME` and `RACING_API_PASSWORD`
- **The Odds API**: API key in `ODDS_API_KEY`, passed as query param `?apiKey=key`
- **Rate limits**: BOTH APIs limited to 500 calls/month — cache aggressively, never call live in tests
- **Examples**: `examples/api_example.py` (Racing API), `examples/odds_api_example.py` (Odds API)

## Important constraints & patterns for edits
- Never commit credentials. `.env` must be gitignored; add new vars to `.env.example`
- Prefer offline/cached data: store raw API responses in `data/raw/`, read from them in development/tests
- No live API calls in unit tests. Use fixtures in `tests/fixtures/` or saved responses in `data/raw/`
- Network code patterns: Racing API uses `auth=(username, password)`; Odds API uses `params={'apiKey': key}`

## Architecture notes (what to look for)
- **Data pipeline phases**:
  - Phase 1: Data cleaning/validation (630K+ races → 245K after removing Class 5-7)
  - Phase 2: Race profitability scoring (0-100 based on class, prize, course tier, field size, pattern races)
  - Phase 3: ML horse win prediction (XGBoost classifier, 18 features, ROC AUC 0.671)
  - Phase 4: Betting strategy (Kelly criterion, value betting)
- **Data flow**: Raw API data → `data/raw/` → Processed datasets → `data/processed/` (Parquet preferred)
- **Model artifacts**: Stored in `models/` (gitignored); training scripts in `scripts/`
- **Feature engineering**: Pure functions preferred; career stats use expanding windows to avoid lookahead bias
- **UI**: Streamlit app in `predictions.py` with tabs for data exploration, ML predictions, value betting
  - Predictions tab: `Today & Tomorrow` — generates and displays predictions for the current date and the next day.
  - Behavior: the UI shows fetch/generate controls only for days that don't yet have generated predictions (keeps the interface minimal once data exists).
  - New: Detailed race-level metrics (Exacta/Trifecta) and per-horse cumulative probabilities
    - `predictions.py` now computes and displays: Exacta (1-2 in order) and Trifecta (1-2-3 in order) probability estimates for the top 3 model picks in the race detail view.
    - For each horse in the race, the app now exposes *cumulative* finish probabilities:
      - `Top 2 %` = P(1st) + P(2nd)
      - `Top 3 %` = P(1st) + P(2nd) + P(3rd)
    - The UI shows the incremental increases (e.g., "60% (+25%)" for Top 2 meaning +25% added by placing) so reviewers can see what was added at each step.
    - Implementation note: Exacta/Trifecta probabilities are approximated via renormalization of the model's win/place/show probabilities (conditional probabilities: P(A) * P(B)/(1-P(A)) * P(C)/(1-P(A)-P(B))). This is documented in `predictions.py` and used for displaying "Fair Trifecta Odds".

## Developer workflows (commands you can run)
- **Verify API access**: `python examples/api_example.py` or `python examples/odds_api_example.py`
- **Run tests**: `pytest tests/`
- **Add dependencies**: Edit `requirements.txt`, then `pip install -r requirements.txt`
- **Run UI**: `streamlit run predictions.py` (loads `data/processed/race_scores.parquet` and `data/logo.png`)
- **Generate predictions**: `python scripts/predict_todays_races.py` or `python scripts/predict_todays_races.py --date 2025-12-31` (outputs `data/processed/predictions_YYYY-MM-DD.csv`)
- **Fetch racecards**: `python scripts/fetch_racecards.py --date 2025-12-31` (saves to `data/raw/racecards_YYYY-MM-DD.json`)
- **Pre-compile check**: `python -c "import py_compile,glob; [py_compile.compile(p, doraise=True) for p in glob.glob('**/*.py', recursive=True)]"`

## Odds conversion and value betting
- **Module**: `scripts/odds_converter.py` provides probability ↔ odds conversions
- **Formats**: Decimal (4.0), Fractional ("3/1"), American (+300)
- **UK fractional odds**: Simple denominators (max 2): 1/2, 1/1, 3/2, 2/1, 5/2, etc.
- **Predictions CSV**: 6 odds columns per horse: win/place/show in decimal + fractional
- **Value betting**: Compare model implied odds to bookmaker odds; bet when bookmaker odds > model odds
- **Example**: Model predicts 25% win (3/1), bookmaker offers 5/1 → 8.33% edge → VALUE BET

## Testing and safety
- Use `pytest`; place tests in `tests/` with fixtures
- Avoid network calls in tests; use `tests/conftest.py` `mock_requests_get` to monkeypatch `requests.get`
- Test fixtures: `tests/fixtures/race_sample.json` for saved API responses

## When you change code
- Update `requirements.txt` for new packages
- Update `.env.example` for new env vars
- Document API call counts in script headers for long-running data pulls
- For UI changes: Test with `streamlit run predictions.py`

## Files to consult first
- **Implementation patterns**: `examples/api_example.py`, `examples/odds_api_example.py`
- **Project conventions**: `README.md`, `.env.example`, `requirements.txt`
- **Data layout**: `data/raw/` (API responses), `data/processed/` (clean datasets in Parquet)
- **UI details**: `predictions.py` — filters (Year, Course, Horse contains, Finish Position); displays normalized dates; raw `pos` column shown as "Finish Position"
- **ML pipeline**: `scripts/phase2_score_races.py` (race scoring), `scripts/phase3_build_horse_model.py` (XGBoost training)

## Streamlit API changes (note: effective after 2025)
- The `use_container_width` parameter has been removed in newer Streamlit versions.
- **Replacement mapping**:
  - `use_container_width=True` → `width='stretch'`
  - `use_container_width=False` → `width='content'`
- **Action required**: Search repo for `use_container_width` and update `st.dataframe`, `st.plotly_chart`, `st.button`, etc.
- **Prefer**: `width='stretch'` for full-width tables/charts in the app
- **Test after update**: `streamlit run predictions.py` to confirm UI renders

## Timezone handling in the UI

- The Streamlit app runs server-side and will use the server's system timezone by default. When the server is in UTC/GMT this can make "Today" and "Tomorrow" differ from the user's local browser date.
- To control which timezone the app uses for determining "today"/"tomorrow", set the environment variable `APP_TIMEZONE` to a valid IANA timezone string (for example, `Europe/London`, `Europe/Dublin`, `America/New_York`).
- The code in `predictions.py` will use `APP_TIMEZONE` when present; otherwise it falls back to the server's local timezone. If you want browser-side timezone detection, consider adding a small client-side component to pass the browser timezone to the server (not implemented by default).

## Final note
Be conservative with API usage (cache, batch, count calls). If changes trigger API calls, ask repo owner before running live pulls.

---
If any part is unclear or needs expansion (caching helpers, test fixtures, CI steps), specify the area.