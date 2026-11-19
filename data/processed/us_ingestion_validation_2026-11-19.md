# US Ingestion Validation Report

- Date: 2026-11-19
- Status: **PASS**
- Generated At (UTC): 2026-11-19T15:46:28.748000+00:00

## Metrics

- Total races: 121
- Total runners: 1083
- Unique tracks: 13
- ML odds coverage: 97.1%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-19.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-19.json
- [PASS] min_total_races: 121 >= 8
- [PASS] min_total_runners: 1083 >= 50
- [PASS] min_ml_odds_coverage: 97.1% >= 50.0%
- [PASS] coverage_drift_guard: today=121 vs median=92.5, min_allowed=46.2
- [PASS] harness_source_coverage: 48 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=121, racecards_total=121
