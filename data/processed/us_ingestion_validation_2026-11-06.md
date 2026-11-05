# US Ingestion Validation Report

- Date: 2026-11-06
- Status: **PASS**
- Generated At (UTC): 2026-11-05T14:42:15.031913+00:00

## Metrics

- Total races: 120
- Total runners: 1062
- Unique tracks: 13
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-06.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-06.json
- [PASS] min_total_races: 120 >= 8
- [PASS] min_total_runners: 1062 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=120 vs median=82.0, min_allowed=41.0
- [PASS] harness_source_coverage: 36 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=120, racecards_total=120
