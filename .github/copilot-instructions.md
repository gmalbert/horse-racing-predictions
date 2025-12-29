# Horse Racing Predictions - AI Coding Agent Instructions

## Project Overview

This is a **horse racing predictions** project using Python. Data is sourced from **The Racing API** (api.theracingapi.com) with HTTP Basic Authentication.

## Development Environment

- **Python Virtual Environment**: Use `.venv/` for all Python dependencies
- **Activation**: 
  - Windows PowerShell: `.venv\Scripts\Activate.ps1`
  - Windows CMD: `.venv\Scripts\activate.bat`
# AI Agent Instructions — Horse Racing Predictions

This file contains concise, actionable guidance for AI coding agents working in this repo.

## Quick context
- Purpose: ML project to predict horse racing outcomes using The Racing API and The Odds API.
- Repo layout: `examples/`, `data/{raw,processed}`, `models/`, `src/`, `notebooks/`, `tests/`.

## Environment & setup (explicit)
- Use a virtualenv at `.venv/` and add any new deps to `requirements.txt`.
- Windows PowerShell activation: `.venv\Scripts\Activate.ps1` (use this for CI-local steps).
- Install deps: `pip install -r requirements.txt`.
- Use `python-dotenv` and `.env` for secrets; update `.env.example` if adding new env vars.

## API integration (concrete)
- The Racing API: HTTP Basic Auth. Credentials stored in `RACING_API_USERNAME` and `RACING_API_PASSWORD`.
- The Odds API: API key stored in `ODDS_API_KEY` and passed as a query param.
- Rate limits: BOTH APIs are limited to ~500 calls/month — do not run live calls recklessly.
- Examples: inspect `examples/api_example.py` (Racing API usage) and `examples/odds_api_example.py` (Odds API usage).

## Important constraints & patterns for edits
- Never commit credentials. `.env` must be gitignored; add new vars to `.env.example`.
- Prefer offline / cached data for development: store raw API responses in `data/raw/` and read from them in tests.
- Do not make live API calls in unit tests. Use recorded responses or fixtures under `tests/` or `data/raw/`.
- When adding network code, follow the existing `requests` pattern: Racing API uses `auth=(username, password)`; Odds API uses `params={'apiKey': key}`.

## Architecture notes (what to look for)
- Data collection vs model training: keep `src/data/` code for collection/preprocessing and `src/models/` for training/inference.
- Models and artifacts belong in `models/` (gitignored); processed datasets belong in `data/processed/`.
- Feature engineering logic typically lives in `src/features/` — prefer pure functions for easier testing.

## Developer workflows (commands you can run)
- Run examples to verify API access:
   - `python examples/api_example.py`
   - `python examples/odds_api_example.py`
- Run unit tests: `pytest tests/`
- Add dependencies: edit `requirements.txt`, then `pip install -r requirements.txt` locally.

- Run the Streamlit UI: `streamlit run predictions.py` (app loads `data/processed/uk_horse_races.csv` and `data/logo.png`)
- Generate daily predictions with odds: `python scripts/predict_todays_races.py` (outputs to `data/processed/predictions_YYYY-MM-DD.csv`)

## Odds conversion and value betting
- Module: `scripts/odds_converter.py` provides probability ↔ odds conversions
- Formats: Decimal (e.g., 4.0), Fractional (e.g., "3/1"), American (e.g., "+300")
- Fractional odds use simple denominators (max 2) for UK racing: 1/2, 1/1, 3/2, 2/1, 5/2, etc.
- Predictions CSV includes 6 odds columns: win/place/show in both decimal and fractional formats
- Value betting workflow: Model generates implied odds → User compares to live bookmaker odds → Bet when bookmaker odds > model odds
- Example: Model says 3/1 (25%), bookmaker offers 5/1 → edge = 8.33% → VALUE BET

## Testing and safety
- Use `pytest` and place new tests in `tests/` alongside any fixtures.
- Avoid direct network calls in tests; use fixtures or save responses to `data/raw/` for deterministic tests.

Notes about test fixtures added:
- `tests/fixtures/race_sample.json` contains a saved API response used by `tests/conftest.py`.
- `tests/conftest.py` includes `mock_requests_get` which monkeypatches `requests.get` to return saved JSON — prefer this pattern for deterministic tests.

## When you change code
- Update `requirements.txt` for new packages.
- Update `.env.example` for new env vars.
- If you add long-running scripts (data pulls), document expected cost and API call counts in the script header and README.

## Files to consult first
- Implementation patterns: `examples/api_example.py`, `examples/odds_api_example.py`
- Project conventions and setup: `README.md`, `.env.example`, `requirements.txt`
- Data layout: `data/raw/` and `data/processed/`

- Streamlit UI: `predictions.py` (root) — shows filters: Year, Course, Horse (contains), Finish Position; main page number-of-results dropdown; uses cached CSV and `data/logo.png`.
- Daily predictions table displays: Horse, Jockey, Win Rank, Win %, Win Odds (fractional), Place Rank, Place %, Place Odds, Show Rank, Show %, Show Odds, Age, Weight, OR, Recent Form.
- Note: the raw CSV column `pos` is exposed in the app as `Finish Position`. The app normalises date display to `YYYY-MM-DD` when time is 00:00:00.

## Final note
Be conservative with external API usage (caching, batching, call counters). If unsure about a change that triggers API calls, ask the repo owner before running live pulls.

---
If any part of this is unclear or you'd like more detail (e.g., suggested caching helpers, test fixtures, or CI steps), tell me which area and I will expand.
│   └── utils/         # Utility functions
├── tests/             # Unit and integration tests
└── requirements.txt   # Python dependencies
```

## Key Commands

**Install Dependencies** (create requirements.txt first if missing):
```bash
pip install -r requirements.txt
```

**Common ML Libraries** for horse racing predictions typically include:
- requests (API calls)
- python-dotenv (environment variables)
- pandas, numpy (data manipulation)
- scikit-learn (ML models)
- xgboost/lightgbm (gradient boosting)
- matplotlib/seaborn (visualization)

## Testing

- Place tests in `tests/` directory
- Use `pytest` as the testing framework
- Run tests: `pytest tests/`

## Notes for AI Agents

- This project is in **initial setup phase** - establish foundational structure first
- Focus on **data quality** and **feature engineering** for racing predictions
- Consider **time-series aspects** of racing data (historical performance)
- Handle **missing data** appropriately (horses without full racing history)

## Streamlit API changes (Dec 2025)

- The Streamlit parameter `use_container_width` will be removed after 2025-12-31.
- Replacement mapping:
   - `use_container_width=True` -> `width='stretch'`
   - `use_container_width=False` -> `width='content'`
- Action: Search the repo for `use_container_width` and update `st.dataframe`, `st.table`, or other components accordingly. Prefer `width='stretch'` for full-width tables in the app and `width='content'` when fixed sizing is desired.
- After updating, run a quick smoke test: `streamlit run predictions.py` to confirm the UI renders.

## Developer tip: pre-compile Python to catch syntax errors

- Before running the app or tests, pre-compile changed Python files to catch syntax errors early. This avoids runtime Streamlit failures caused by simple syntax mistakes.
- Examples:
   - Pre-compile a single file: `python -m py_compile predictions.py`
   - Pre-compile all files (POSIX shell): `python -m py_compile $(git ls-files '*.py')`
   - Or run a small Python one-liner to compile recursively:
      `python -c "import py_compile,glob; [py_compile.compile(p, doraise=True) for p in glob.glob('**/*.py', recursive=True)]"`

Add `py_compile` as a quick local check in your workflow or CI to reduce simple syntax-related errors.
