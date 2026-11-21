# US Ingestion Validation Report

- Date: 2026-11-21
- Status: **PASS**
- Generated At (UTC): 2026-11-21T14:44:15.363055+00:00

## Metrics

- Total races: 142
- Total runners: 1226
- Unique tracks: 14
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-21.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-21.json
- [PASS] min_total_races: 142 >= 8
- [PASS] min_total_runners: 1226 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=142 vs median=92.5, min_allowed=46.2
- [PASS] harness_source_coverage: 28 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=142, racecards_total=142
