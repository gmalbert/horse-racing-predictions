# US Ingestion Validation Report

- Date: 2026-11-13
- Status: **PASS**
- Generated At (UTC): 2026-11-12T15:13:00.753224+00:00

## Metrics

- Total races: 115
- Total runners: 1002
- Unique tracks: 12
- ML odds coverage: 98.9%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-13.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-13.json
- [PASS] min_total_races: 115 >= 8
- [PASS] min_total_runners: 1002 >= 50
- [PASS] min_ml_odds_coverage: 98.9% >= 50.0%
- [PASS] coverage_drift_guard: today=115 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 38 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=115, racecards_total=115
