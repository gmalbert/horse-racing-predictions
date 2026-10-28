# US Ingestion Validation Report

- Date: 2026-10-28
- Status: **PASS**
- Generated At (UTC): 2026-10-28T12:51:52.405299+00:00

## Metrics

- Total races: 95
- Total runners: 814
- Unique tracks: 10
- ML odds coverage: 97.1%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-10-28.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-10-28.json
- [PASS] min_total_races: 95 >= 8
- [PASS] min_total_runners: 814 >= 50
- [PASS] min_ml_odds_coverage: 97.1% >= 50.0%
- [PASS] coverage_drift_guard: today=95 vs median=51.0, min_allowed=25.5
- [PASS] harness_source_coverage: 39 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=95, racecards_total=95
