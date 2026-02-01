# Model Improvement Roadmap

## Executive Summary

This document provides a comprehensive improvement roadmap for the horse racing prediction system. Analysis of current predictions and model metrics reveals several critical gaps that need addressing.

### Current Model Metrics (Baseline)
- **ROC AUC: 0.671** (test set) — marginally better than random (0.5)
- **Training Accuracy: 88.7%** — misleadingly high due to class imbalance
- **47 features** — many showing 0 importance (blinkers, visor, gear_changed, distance bands)
- **Top feature: field_size (11.5%)** — a race-level feature, not horse-specific

### Key Problems Identified

1. **Cold Start Problem**: Many horses have `career_runs=0`, meaning no historical form data
2. **Class Imbalance**: ~10% winners in typical race → 88% accuracy by always predicting "lose"
3. **Missing Predictive Features**: No pace data, pedigree analysis, or market signals
4. **Feature Leakage Risk**: Some features may inadvertently encode outcome information
5. **Limited Data Sources**: Currently only The Racing API; no odds integration

### Roadmap Documents

| Document | Priority | Timeline | Focus |
|----------|----------|----------|-------|
| [0. Immediate Action Plan](IMMEDIATE_ACTION_PLAN.md) | 🔴 START HERE | Days 1-10 | Quick wins with code |
| [1. Critical Data Gaps](CRITICAL_DATA_GAPS.md) | 🔴 Urgent | 1-2 weeks | Essential missing data |
| [2. Free Data Sources](FREE_DATA_SOURCES.md) | 🔴 Urgent | 1-2 weeks | Where to get more data |
| [3. Feature Engineering V2](FEATURE_ENGINEERING_V2.md) | 🟠 High | 2-4 weeks | New feature ideas |
| [4. Model Architecture Improvements](MODEL_ARCHITECTURE_IMPROVEMENTS.md) | 🟠 High | 2-4 weeks | Better ML approaches |
| [5. Validation and Backtesting](VALIDATION_BACKTESTING.md) | 🟡 Medium | 3-6 weeks | Proper evaluation |
| [6. UI and Betting Enhancements](UI_BETTING_ENHANCEMENTS.md) | 🟡 Medium | 4-8 weeks | User experience |

---

## Priority Action Items

### Week 1-2: Quick Wins
1. **Fix cold start**: Use sire/dam statistics for horses with < 3 career runs
2. **Add pace analysis**: Front-runner vs closer classification
3. **Integrate Betfair SP data**: Free historical starting prices
4. **Recalibrate probability outputs**: Current probabilities may be poorly calibrated

### Week 3-4: Data Expansion
1. **Scrape Racing Post ratings**: RPR is more predictive than OR alone
2. **Add trainer form**: 14-day and 30-day trainer strike rates
3. **Weather integration**: Going prediction using Met Office data
4. **Market movement signals**: Early price vs SP movement

### Month 2-3: Model Overhaul
1. **Ensemble approach**: Combine XGBoost, LightGBM, and neural network
2. **Specialized models**: Different models for maidens, handicaps, Group races
3. **Rank-based training**: Optimize for ranking horses within race, not just win/lose
4. **Temporal validation**: Proper walk-forward testing

---

## Success Metrics

| Metric | Current | Target (3mo) | Target (6mo) |
|--------|---------|--------------|--------------|
| ROC AUC | 0.671 | 0.72 | 0.75 |
| Top-1 Accuracy | ~15% | 22% | 28% |
| Top-3 Accuracy | ~45% | 55% | 65% |
| Profit (1pt level stakes) | Negative | Breakeven | +5% ROI |
| Calibration Error | Unknown | < 0.05 | < 0.03 |

---

## Navigation

See individual documents for detailed implementation:

0. [Immediate Action Plan](IMMEDIATE_ACTION_PLAN.md) — **START HERE** — Quick wins with code samples
1. [Critical Data Gaps](CRITICAL_DATA_GAPS.md) — What's missing that would help predictions
2. [Free Data Sources](FREE_DATA_SOURCES.md) — Where to get more data for free
3. [Feature Engineering V2](FEATURE_ENGINEERING_V2.md) — New features to implement
4. [Model Architecture Improvements](MODEL_ARCHITECTURE_IMPROVEMENTS.md) — Better ML techniques
5. [Validation and Backtesting](VALIDATION_BACKTESTING.md) — Proper testing methodology
6. [UI and Betting Enhancements](UI_BETTING_ENHANCEMENTS.md) — User-facing improvements
