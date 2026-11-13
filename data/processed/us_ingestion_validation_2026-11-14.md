# US Ingestion Validation Report

- Date: 2026-11-14
- Status: **PASS**
- Generated At (UTC): 2026-11-13T14:39:03.251598+00:00

## Metrics

- Total races: 127
- Total runners: 1147
- Unique tracks: 12
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-14.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-14.json
- [PASS] min_total_races: 127 >= 8
- [PASS] min_total_runners: 1147 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=127 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 28 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=127, racecards_total=127
