# US Ingestion Validation Report

- Date: 2026-11-12
- Status: **PASS**
- Generated At (UTC): 2026-11-12T15:13:00.679261+00:00

## Metrics

- Total races: 103
- Total runners: 870
- Unique tracks: 11
- ML odds coverage: 96.9%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-12.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-12.json
- [PASS] min_total_races: 103 >= 8
- [PASS] min_total_runners: 870 >= 50
- [PASS] min_ml_odds_coverage: 96.9% >= 50.0%
- [PASS] coverage_drift_guard: today=103 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 47 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=103, racecards_total=103
