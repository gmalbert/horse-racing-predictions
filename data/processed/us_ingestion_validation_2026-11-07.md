# US Ingestion Validation Report

- Date: 2026-11-07
- Status: **PASS**
- Generated At (UTC): 2026-11-06T14:21:02.806138+00:00

## Metrics

- Total races: 121
- Total runners: 1073
- Unique tracks: 12
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-07.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-07.json
- [PASS] min_total_races: 121 >= 8
- [PASS] min_total_runners: 1073 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=121 vs median=88.5, min_allowed=44.2
- [PASS] harness_source_coverage: 26 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=121, racecards_total=121
