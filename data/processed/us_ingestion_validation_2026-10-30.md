# US Ingestion Validation Report

- Date: 2026-10-30
- Status: **PASS**
- Generated At (UTC): 2026-10-30T12:56:39.728295+00:00

## Metrics

- Total races: 132
- Total runners: 1157
- Unique tracks: 13
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-10-30.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-10-30.json
- [PASS] min_total_races: 132 >= 8
- [PASS] min_total_runners: 1157 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=132 vs median=73.5, min_allowed=36.8
- [PASS] harness_source_coverage: 47 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=132, racecards_total=132
