# US Ingestion Validation Report

- Date: 2026-11-20
- Status: **PASS**
- Generated At (UTC): 2026-11-19T15:46:28.809748+00:00

## Metrics

- Total races: 129
- Total runners: 1133
- Unique tracks: 14
- ML odds coverage: 99.3%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-20.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-20.json
- [PASS] min_total_races: 129 >= 8
- [PASS] min_total_runners: 1133 >= 50
- [PASS] min_ml_odds_coverage: 99.3% >= 50.0%
- [PASS] coverage_drift_guard: today=129 vs median=92.5, min_allowed=46.2
- [PASS] harness_source_coverage: 37 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=129, racecards_total=129
