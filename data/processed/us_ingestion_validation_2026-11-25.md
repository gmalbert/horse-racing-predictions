# US Ingestion Validation Report

- Date: 2026-11-25
- Status: **PASS**
- Generated At (UTC): 2026-11-26T03:24:41.574067+00:00

## Metrics

- Total races: 74
- Total runners: 544
- Unique tracks: 9
- ML odds coverage: 98.7%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-25.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-25.json
- [PASS] min_total_races: 74 >= 8
- [PASS] min_total_runners: 544 >= 50
- [PASS] min_ml_odds_coverage: 98.7% >= 50.0%
- [PASS] coverage_drift_guard: today=74 vs median=86.5, min_allowed=43.2
- [PASS] harness_source_coverage: 7 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=74, racecards_total=74
