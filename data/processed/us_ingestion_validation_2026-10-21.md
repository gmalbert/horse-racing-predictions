# US Ingestion Validation Report

- Date: 2026-10-21
- Status: **PASS**
- Generated At (UTC): 2026-10-22T02:50:03.475566+00:00

## Metrics

- Total races: 8
- Total runners: 62
- Unique tracks: 1
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data\raw\us_racecards_2026-10-21.json
- [PASS] source_report_exists: data\raw\us_source_report_2026-10-21.json
- [PASS] min_total_races: 8 >= 8
- [PASS] min_total_runners: 62 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: Insufficient history for drift baseline; check skipped
- [PASS] source_report_total_matches_racecards: report_total=8, racecards_total=8
