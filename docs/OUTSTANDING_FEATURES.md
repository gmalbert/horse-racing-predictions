# Outstanding Features & Documentation Tasks

This file consolidates active open tasks from the repo documentation and current code state.
Use it as a single reference for the remaining work items across data, model, UI, and operational roadmaps.

## High Priority Open Tasks

- Data leakage audit
  - Run `scripts/add_pedigree_features_no_leakage.py`
  - Regenerate the full feature dataset
  - Update training input paths in `phase3_build_horse_model.py`
  - Verify the no-leakage pedigree pipeline end-to-end

- Backtesting
  - Backtest on Oct-Dec 2024 historical races
  - Confirm model performance with realistic replay and holdout data

- Production deployment
  - Finalize deployment process for daily predictions and monitoring
  - Ensure the Streamlit UI is stable in the target environment

## Betting Strategy Roadmap

From `docs/BETTING_STRATEGY.md`, the following checklist items remain actionable:
- Clean and validate historical database
- Build aggregation scripts for horse/jockey/trainer statistics
- Create exploratory analysis notebook for betting strategy
- Implement race scoring and fixture ingestion automation
- Build the watchlist generator and value bet identifier
- Add jockey/trainer features and integrate bookmaker odds
- Train and backtest value-betting strategy models

<!-- ## US Racing Support

From `docs/US_RACING_IMPLEMENTATION.md`:
- Verify The Racing API US data coverage
- Confirm US class, surface, and going format support
- Add region-specific loading and filters
- Validate predictions for US sample dates -->

## Operational/Infrastructure Tasks

From `docs/FIX_LFS_BANDWIDTH_QUOTA.md`:
- Identify Git LFS tracked files that should be simplified
- Update GitHub Actions workflows and cache steps
- Test the fix locally and in CI

## Long-Term Enhancements

From `docs/LONG_TERM_DATA_ENHANCEMENTS.md` and related files:
- Implement market-data/Betfair exchange integration
- Add weather and ground condition modeling
- Build ensemble and ranker models
<!-- - Add video or image-based race analysis if feasible -->
- Build continuous retraining and RL bankroll management pipelines

## Notes

- Some items are already implemented in code and docs were updated accordingly.
- Continue updating this file when resolved tasks are cleared.
