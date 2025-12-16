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
- Note: the raw CSV column `pos` is exposed in the app as `Finish Position`. The app normalises date display to `YYYY-MM-DD` when time is 00:00:00.

## Final note
Be conservative with external API usage (caching, batching, call counters). If unsure about a change that triggers API calls, ask the repo owner before running live pulls.

---
If any part of this is unclear or you'd like more detail (e.g., suggested caching helpers, test fixtures, or CI steps), tell me which area and I will expand.
│   └── utils/         # Utility functions
