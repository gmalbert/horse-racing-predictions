# US Ingestion Validation Report

- Date: 2026-11-21
- Status: **PASS**
- Generated At (UTC): 2026-11-20T15:28:13.906117+00:00

## Metrics

- Total races: 142
- Total runners: 1259
- Unique tracks: 14
- ML odds coverage: 99.4%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-21.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-21.json
- [PASS] min_total_races: 142 >= 8
- [PASS] min_total_runners: 1259 >= 50
- [PASS] min_ml_odds_coverage: 99.4% >= 50.0%
- [PASS] coverage_drift_guard: today=142 vs median=92.5, min_allowed=46.2
- [PASS] harness_source_coverage: 28 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=142, racecards_total=142
