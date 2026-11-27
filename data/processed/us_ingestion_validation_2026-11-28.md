# US Ingestion Validation Report

- Date: 2026-11-28
- Status: **PASS**
- Generated At (UTC): 2026-11-27T16:41:09.608209+00:00

## Metrics

- Total races: 134
- Total runners: 1258
- Unique tracks: 13
- ML odds coverage: 98.2%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-28.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-28.json
- [PASS] min_total_races: 134 >= 8
- [PASS] min_total_runners: 1258 >= 50
- [PASS] min_ml_odds_coverage: 98.2% >= 50.0%
- [PASS] coverage_drift_guard: today=134 vs median=81.5, min_allowed=40.8
- [PASS] harness_source_coverage: 29 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=134, racecards_total=134
