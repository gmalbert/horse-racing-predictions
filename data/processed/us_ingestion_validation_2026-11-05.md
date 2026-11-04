# US Ingestion Validation Report

- Date: 2026-11-05
- Status: **PASS**
- Generated At (UTC): 2026-11-04T14:33:41.202492+00:00

## Metrics

- Total races: 111
- Total runners: 958
- Unique tracks: 12
- ML odds coverage: 99.9%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-05.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-05.json
- [PASS] min_total_races: 111 >= 8
- [PASS] min_total_runners: 958 >= 50
- [PASS] min_ml_odds_coverage: 99.9% >= 50.0%
- [PASS] coverage_drift_guard: today=111 vs median=82.0, min_allowed=41.0
- [PASS] harness_source_coverage: 46 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=111, racecards_total=111
