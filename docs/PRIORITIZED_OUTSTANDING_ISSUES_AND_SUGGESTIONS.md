# Prioritized Outstanding Issues and Suggestions (Docs Consolidation)

Date: 2026-06-06
Scope: Consolidated from all Markdown files under docs/ (54 files reviewed)

## Prioritization Framework

- P0 Critical: Blocks reliable production operation, risks bad predictions, or creates high operational risk.
- P1 High: Strongly impacts model quality, coverage, or maintainability but is not an immediate production blocker.
- P2 Medium: Important enhancements with meaningful upside.
- P3 Low: Nice-to-have, exploratory, or longer-horizon initiatives.

## Consolidated Backlog

### P0 Critical

1. Production-grade validation/backtesting gate is still incomplete
- Why it matters: Docs repeatedly flag that model quality can look good offline but fail in production without strict temporal and walk-forward validation plus betting simulation.
- Suggested action: Make walk-forward backtest and calibration metrics a release gate before deploying daily model or feature changes.
- Evidence: VALIDATION_BACKTESTING.md, IMMEDIATE_ACTION_PLAN.md, MODEL_IMPROVEMENT_ROADMAP.md
- Confidence: High

2. US scraping foundation tasks remain largely backlog with no completed foundation row
- Why it matters: Core schema, adapter, idempotent storage, and validation are listed as critical-path tasks; gaps here create fragile ingestion and poor downstream reliability.
- Suggested action: Complete Foundation F-01 through F-05 first, then unlock tier rollouts.
- Evidence: US_TRACK_SCRAPING_TRACKER.md, US_TRACK_SCRAPING_EXECUTION_PLAN.md
- Confidence: High

3. Daily operational guardrails for US ingestion are not fully closed
- Why it matters: Execution plan calls out nightly runs plus morning validation to avoid silent gaps; tracker still shows pending integration tasks for some pipelines.
- Suggested action: Add enforced daily scheduler + validation report with fail-fast alerts and non-zero exit on coverage drift.
- Evidence: US_TRACK_SCRAPING_EXECUTION_PLAN.md, US_TRACK_SCRAPING_TRACKER.md
- Confidence: High

4. Inference/training feature parity risk is still documented as an active failure mode
- Why it matters: Several docs call out feature mismatch issues between training and prediction paths; this causes runtime errors and degraded predictions.
- Suggested action: Add automated feature-contract check between training artifact expected columns and prediction pipeline output before scoring.
- Evidence: IMMEDIATE_ACTION_PLAN.md, PREDICTION_UPDATES_COMPLETE.md, MODEL_V2.1_RESULTS.md, MODEL_RETRAINING_COMPLETE.md
- Confidence: High

### P1 High

5. Backtesting of value-betting strategy and bankroll policy needs formal completion
- Why it matters: Value betting logic exists, but end-to-end profitability and drawdown behavior are still treated as pending/roadmap in multiple docs.
- Suggested action: Run rolling historical simulation with level and fractional Kelly variants, publish monthly ROI, max drawdown, and calibration drift.
- Evidence: BETTING_STRATEGY.md, VALIDATION_BACKTESTING.md, MODEL_SUGGESTED_ENHANCEMENTS.md
- Confidence: High

6. Remaining weight-feature depth is explicitly incomplete
- Why it matters: Basic weight features are present, but higher-signal handicap features are still marked missing.
- Suggested action: Implement and evaluate weight_for_age, weight_trend, lb_per_length, and handicap_efficiency with leakage checks.
- Evidence: CRITICAL_DATA_GAPS.md
- Confidence: High

7. US track parser hardening and fixtures/tests are still incomplete for in-progress tracks
- Why it matters: Several track cards are in progress but still missing fixture captures and parser unit tests, increasing breakage risk.
- Suggested action: For each in-progress track, complete fixture capture and parser tests before promoting status to done.
- Evidence: US_TRACK_SCRAPING_TRACKER.md
- Confidence: High

8. US odds coverage and source quality verification remains open in planning docs
- Why it matters: If US market odds coverage is partial, value-bet outputs and comparison tools become unreliable.
- Suggested action: Run explicit US odds coverage audit by track/date and add fallback source hierarchy.
- Evidence: US_RACING_DATA_API.md, US_RACING_IMPLEMENTATION.md, us_horse_racing_scraping_guide.md
- Confidence: Medium

9. Data leakage prevention is fixed in audited scripts but should be continuously enforced
- Why it matters: Audit docs show fixes, but future feature additions can reintroduce leakage if guardrails are not automated.
- Suggested action: Add leakage verification script to CI for feature-engineering changes and require pass before merge.
- Evidence: DATA_LEAKAGE_AUDIT_V2.3.md, DATA_LEAKAGE_VERIFICATION_V2.1.md, SITE_SUMMARY.md
- Confidence: High

10. Operational workflow robustness around large artifacts and caching remains a recurring concern
- Why it matters: Multiple docs discuss LFS/cache/bandwidth constraints that can silently disrupt automated runs.
- Suggested action: Standardize cache keys, avoid LFS-heavy paths in automation, and add workflow health checks.
- Evidence: FIX_LFS_BANDWIDTH_QUOTA.md, GITHUB_ACTIONS_LFS_FIX.md
- Confidence: Medium

### P2 Medium

11. Documentation drift and stale guidance across roadmap docs
- Why it matters: Some docs still present already-implemented items as upcoming, which can mislead prioritization.
- Suggested action: Mark docs as Active, Completed, or Historical and add last-verified date with owner.
- Evidence: NEXT_FEATURES.md, UI_BETTING_ENHANCEMENTS.md, OUTSTANDING_FEATURES.md, IMPLEMENTATION_COMPLETE.md
- Confidence: High

12. Model architecture improvements and specialized model strategy remain partially roadmap-only
- Why it matters: Potential AUC/ROI gains are identified but not systematically closed.
- Suggested action: Create phase-based experiment board with acceptance metrics per architecture change.
- Evidence: MODEL_ARCHITECTURE_IMPROVEMENTS.md, MODEL_IMPROVEMENT_ROADMAP.md, MODEL_SUGGESTED_ENHANCEMENTS.md
- Confidence: Medium

13. Market-data enrichment and external source integration plans are broad but under-sequenced
- Why it matters: Many enhancement docs propose additional data feeds, but sequencing and validation budget are not tightly scoped.
- Suggested action: Prioritize one source at a time with measurable uplift target and rollback criteria.
- Evidence: FREE_DATA_SOURCES.md, ODDS_SCRAPING_REPOS_ANALYSIS.md, LONG_TERM_DATA_ENHANCEMENTS.md
- Confidence: Medium

### P3 Low

14. Long-horizon RL and advanced NLP/trouble-line extraction are promising but expensive
- Why it matters: High complexity and data quality demands; likely lower immediate ROI than core validation and ingestion hardening.
- Suggested action: Keep as research track with strict milestone gates and defer until P0/P1 closure.
- Evidence: LONG_TERM_DATA_ENHANCEMENTS.md, US_Horseracing roadmap.md
- Confidence: Medium

15. UI experimentation and feature-rich display ideas should follow data-quality stabilization
- Why it matters: UI value is limited if core data coverage and validation are not locked.
- Suggested action: Continue incremental UI changes only when backed by validated model/data improvements.
- Evidence: UI_BETTING_ENHANCEMENTS.md, PREDICTION_UPDATES_COMPLETE.md
- Confidence: Medium

## Top 10 Now (Ordered)

1. Implement hard validation/backtesting gate for production changes (P0)
2. Finish US foundation tasks F-01 to F-05 (P0)
3. Enforce daily US ingestion monitoring and morning validation report (P0)
4. Add feature-contract parity checks for training vs inference (P0)
5. Complete value-betting and bankroll historical backtest framework (P1)
6. Complete remaining advanced weight features (P1)
7. Close US in-progress track fixtures/tests before expansion (P1)
8. Validate US odds coverage and define fallback hierarchy (P1)
9. Add leakage verification to CI for feature changes (P1)
10. Stabilize workflow cache/LFS operational checks (P1)

## Cross-Cutting Conflicts and Inconsistencies

- Planning docs vs current implementation status are inconsistent in several places.
- Some docs state core automation or odds integration as pending while workflow docs and newer implementation docs show those parts running.
- Some enhancement docs are effectively historical snapshots but are not labeled as such, causing duplicate or stale backlog entries.

Recommended normalization policy:
- Add a status banner to each doc: Active, Historical, or Completed.
- Add Last Verified date and owner to each roadmap doc.
- Maintain this file as the single live backlog and link out to detail docs.

## Notes on Coverage

- All 54 Markdown files under docs/ were included in the review scope.
- Items marked completed in source docs were excluded unless they imply residual risk, follow-up, or inconsistency.
- Priority mapping was normalized to P0/P1/P2/P3 across heterogeneous doc styles.

## Files Reviewed (54)

- 6_MONTH_FEATURES.md
- APPENDING_NEW_DATA.md
- architecture.md
- AUTO_START_WATCHER.md
- BATCH_PREDICTIONS.md
- BETTING_STRATEGY.md
- CACHE_CORRUPTION_FIX.md
- CRITICAL_DATA_GAPS.md
- DATA_LEAKAGE_AUDIT.md
- DATA_LEAKAGE_AUDIT_V2.3.md
- DATA_LEAKAGE_VERIFICATION_V2.1.md
- feature_engineering_roo.md
- FEATURE_ENGINEERING_V2.md
- FEATURE_IMPLEMENTATION_SUMMARY.md
- FEATURE_IMPLEMENTATION_V2.2.md
- FEATURE_IMPLEMENTATION_V2.3.md
- FIX_LFS_BANDWIDTH_QUOTA.md
- FREE_DATA_SOURCES.md
- GITHUB_ACTIONS_LFS_FIX.md
- IMMEDIATE_ACTION_PLAN.md
- IMPLEMENTATION_COMPLETE.md
- LONG_TERM_DATA_ENHANCEMENTS.md
- macos-emoji-filename-fix.md
- MEDIUM_TERM_DATA_ENHANCEMENTS.md
- MODEL_ARCHITECTURE_IMPROVEMENTS.md
- MODEL_CALIBRATION_AND_DIAGNOSTICS.md
- MODEL_IMPROVEMENT_ROADMAP.md
- MODEL_RETRAINING_COMPLETE.md
- MODEL_SUGGESTED_ENHANCEMENTS.md
- MODEL_V2.1_RESULTS.md
- NEXT_FEATURES.md
- ODDS_SCRAPING_REPOS_ANALYSIS.md
- OUTSTANDING_FEATURES.md
- PIPELINE_REGENERATION.md
- PREDICTION_UPDATES_COMPLETE.md
- pulsescore_horse_racing_guide.md
- QUICK_START_AUTOMATION.md
- QUICK_START_MODEL_V2.md
- RACING_EDGE_UK_SIGNAL_ENHANCEMENTS.md
- RACING_EDGE_UK_SIGNAL_ENHANCEMENTS_US.md
- REFACTORING_SUMMARY.md
- SHORT_TERM_DATA_ENHANCEMENTS.md
- SITE_SUMMARY.md
- UI_BETTING_ENHANCEMENTS.md
- US_Horseracing roadmap.md
- us_horse_racing_scraping_guide.md
- US_RACING_CODE_EXAMPLES.md
- US_RACING_DATA_API.md
- US_RACING_EXPANSION_OVERVIEW.md
- US_RACING_FEATURES.md
- US_RACING_IMPLEMENTATION.md
- US_TRACK_SCRAPING_EXECUTION_PLAN.md
- US_TRACK_SCRAPING_TRACKER.md
- VALIDATION_BACKTESTING.md
