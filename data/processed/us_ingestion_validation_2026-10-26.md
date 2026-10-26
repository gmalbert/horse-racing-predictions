# US Ingestion Validation Report

- Date: 2026-10-26
- Status: **PASS**
- Generated At (UTC): 2026-10-26T15:10:22.775872+00:00

## Metrics

- Total races: 50
- Total runners: 435
- Unique tracks: 5
- ML odds coverage: 95.9%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-10-26.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-10-26.json
- [PASS] min_total_races: 50 >= 8
- [PASS] min_total_runners: 435 >= 50
- [PASS] min_ml_odds_coverage: 95.9% >= 50.0%
- [PASS] coverage_drift_guard: Insufficient history for drift baseline; check skipped
- [PASS] harness_source_coverage: 33 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=50, racecards_total=50
