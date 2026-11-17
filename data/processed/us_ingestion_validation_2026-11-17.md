# US Ingestion Validation Report

- Date: 2026-11-17
- Status: **PASS**
- Generated At (UTC): 2026-11-17T15:59:45.828585+00:00

## Metrics

- Total races: 62
- Total runners: 506
- Unique tracks: 6
- ML odds coverage: 96.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-17.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-17.json
- [PASS] min_total_races: 62 >= 8
- [PASS] min_total_runners: 506 >= 50
- [PASS] min_ml_odds_coverage: 96.0% >= 50.0%
- [PASS] coverage_drift_guard: today=62 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 35 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=62, racecards_total=62
