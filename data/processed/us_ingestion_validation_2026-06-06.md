# US Ingestion Validation Report

- Date: 2026-06-06
- Status: **FAIL**
- Generated At (UTC): 2026-06-06T19:45:29.883603+00:00

## Metrics

- Total races: 354
- Total runners: 2902
- Unique tracks: 34
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data\raw\us_racecards_2026-06-06.json
- [FAIL] source_report_exists: data\raw\us_source_report_2026-06-06.json
- [PASS] min_total_races: 354 >= 8
- [PASS] min_total_runners: 2902 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=354 vs median=26.5, min_allowed=13.2
