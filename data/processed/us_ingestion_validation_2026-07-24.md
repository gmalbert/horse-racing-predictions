# US Ingestion Validation Report

- Date: 2026-07-24
- Status: **PASS**
- Generated At (UTC): 2026-07-24T07:08:12.847060+00:00

## Metrics

- Total races: 0
- Total runners: 0
- Unique tracks: 0
- ML odds coverage: 0.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-07-24.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-07-24.json
- [PASS] min_total_races: 0 >= 8 (empty racecards allowed)
- [PASS] min_total_runners: 0 >= 50 (empty racecards allowed)
- [PASS] min_ml_odds_coverage: 0.0% >= 50.0% (empty racecards allowed)
- [PASS] coverage_drift_guard: Insufficient history for drift baseline; check skipped
- [PASS] source_report_total_matches_racecards: report_total=0, racecards_total=0
