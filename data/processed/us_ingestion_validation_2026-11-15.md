# US Ingestion Validation Report

- Date: 2026-11-15
- Status: **PASS**
- Generated At (UTC): 2026-11-14T13:33:06.541964+00:00

## Metrics

- Total races: 80
- Total runners: 688
- Unique tracks: 8
- ML odds coverage: 98.5%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-15.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-15.json
- [PASS] min_total_races: 80 >= 8
- [PASS] min_total_runners: 688 >= 50
- [PASS] min_ml_odds_coverage: 98.5% >= 50.0%
- [PASS] coverage_drift_guard: today=80 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 16 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=80, racecards_total=80
