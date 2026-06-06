## Summary

This PR hardens US track-site parser reliability and improves visibility into backtest market-odds joins.

## Changes

### Parser hardening
- Expanded real-capture fixtures in tests/fixtures/t1_real_capture_snapshots.json.
- Extended tests/test_t1_parser_hardening.py with:
  - richer TB/QH/H snapshot coverage
  - malformed/fallback/null-page behavior checks
  - parse_date integration tests using real snapshots
- Added coverage across available capture dates: 2026-05-09 to 2026-05-11.

### Backtest odds diagnostics
- Updated scripts/backtest_walk_forward.py:
  - improved horse normalization for joins (country suffix, diacritics, punctuation, whitespace)
  - structured market-odds diagnostics (coverage, duplicates, unmatched date concentration)
  - optional diagnostics output path via --odds-diagnostics-file
  - default diagnostics output file: data/processed/backtest_market_odds_diagnostics.json
- Added tests/test_backtest_walk_forward_odds.py for:
  - diagnostics generation
  - required-column guard behavior
  - normalization matching behavior

### CI
- Added .github/workflows/parser_hardening_checks.yml to run parser/odds hardening tests on push/PR path changes.

## Why

- Reduce parser fragility from real-world HTML/content failures.
- Keep parser behavior stable as fixture corpus grows.
- Make market-odds join quality observable so value-betting backtests are diagnosable.

## Validation

Executed locally:
- python -m pytest tests/test_t1_parser_hardening.py tests/test_backtest_walk_forward_odds.py -q
- Result: 35 passed

## Notes

- Raw captures earlier than 2026-05-09 were not present in this workspace, so fixture growth covers all available dates.
- This PR intentionally focuses on parser hardening and odds-join diagnostics scope.
