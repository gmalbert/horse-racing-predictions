# PR Summary: Parser Corpus Hardening + Odds Join Diagnostics

Date: 2026-06-06
Branch: fix/odds-errors

## Scope

This change set hardens parser regression coverage for US track-site ingestion and adds actionable diagnostics for market-odds joins used by backtesting.

## What Changed

### 1) Real-capture parser corpus expansion (TB/QH/H)
- Added a larger real snapshot fixture corpus:
  - tests/fixtures/t1_real_capture_snapshots.json
- Added/expanded parser hardening tests:
  - tests/test_t1_parser_hardening.py

Coverage now includes:
- TB: TAM, LS, PRX, HOU, CBY, CT, ELP, MNR, PEN, PID
- QH: RUD, ZIA, LA, SUN
- Harness: RSC, PLN, NTH, RUN
- Date range represented in fixtures: 2026-05-09 through 2026-05-11

Cases covered:
- Successful summary extraction
- Results row parsing for TAM snapshots
- Null/empty best_page fallback behavior
- Access denied / blocked pages
- 404 and low-quality fallback pages

### 2) parse_date integration tests with real snapshots
- Added end-to-end parse_date tests for TB/QH/H parser modules using fixture-backed snapshots.
- Verifies canonical CSV/JSON output and expected event rows.

### 3) Backtest market-odds join diagnostics
- Updated scripts/backtest_walk_forward.py:
  - Expanded horse-name normalization for join keys:
    - strips country suffixes
    - removes diacritics
    - removes punctuation/apostrophes
    - normalizes whitespace
  - Added structured odds-join diagnostics, including:
    - total rows, matched rows, coverage
    - source duplicate key rows
    - date-key and horse-key coverage
    - top unmatched dates
  - Stores diagnostics in DataFrame attrs and writes JSON by default to:
    - data/processed/backtest_market_odds_diagnostics.json
  - Added CLI option:
    - --odds-diagnostics-file

### 4) Odds diagnostics unit tests
- Added tests:
  - tests/test_backtest_walk_forward_odds.py
- Validates:
  - diagnostics generation and key metrics
  - required-column guard behavior
  - name normalization matching with punctuation/diacritics variations

### 5) CI hook for parser hardening and odds diagnostics
- Added workflow:
  - .github/workflows/parser_hardening_checks.yml
- Runs on push/PR when parser/fixture/diagnostics files change.
- Executes:
  - python -m pytest tests/test_t1_parser_hardening.py tests/test_backtest_walk_forward_odds.py -q

## Validation Evidence

Executed locally:
- python -m pytest tests/test_t1_parser_hardening.py -q
  - 32 passed (after corpus expansion)
- python -m pytest tests/test_backtest_walk_forward_odds.py tests/test_t1_parser_hardening.py -q
  - 35 passed (final state)

## Notes

- No raw captures earlier than 2026-05-09 were present in the current workspace. Corpus growth was therefore completed across all available dates (2026-05-09 to 2026-05-11).
- This summary intentionally focuses on parser/odds diagnostics scope and does not claim ownership of unrelated modified files currently present in the worktree.
