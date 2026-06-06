# Commit Message Drafts (Parser + Odds Diagnostics)

Use these as non-interactive commit templates.

## Option A: Two commits

1. Parser corpus and CI hook

chore(tests): expand real-capture parser corpus and add CI hardening checks

- add TB/QH/H real snapshot fixtures across 2026-05-09..2026-05-11
- extend parser hardening tests with summary/error/fallback scenarios
- add parse_date integration tests using real snapshot fixtures
- add parser_hardening_checks GitHub Actions workflow

2. Odds diagnostics and normalization

feat(backtest): add market-odds join diagnostics and stronger name normalization

- improve horse normalization (country suffix, diacritics, punctuation, spacing)
- add structured join diagnostics (coverage, duplicate keys, unmatched concentration)
- add --odds-diagnostics-file output option and default diagnostics JSON emit
- add odds diagnostics unit tests

## Option B: Single squashed commit

feat(parser+backtest): harden track parsers and add odds-join diagnostics

- expand real-capture parser fixtures/tests for TB/QH/H
- add parse_date integration tests for real snapshot paths
- add parser/odds diagnostics CI workflow
- improve horse-name normalization for odds joins
- emit structured market-odds join diagnostics and add unit tests
- include PR summary documentation

## Optional commit body footer

Validation:
- python -m pytest tests/test_t1_parser_hardening.py tests/test_backtest_walk_forward_odds.py -q
- 35 passed
