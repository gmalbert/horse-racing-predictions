# US Racing Expansion - Overview

**Last Updated**: January 1, 2026  
**Status**: Planning Phase

## Executive Summary

This document outlines the strategy and requirements for expanding the horse racing predictions system to include **US racing** alongside the current UK/Ireland coverage.

### Key Challenges

The UK and US racing systems have fundamental structural differences that require careful handling:

| Aspect | UK Racing | US Racing |
|--------|-----------|-----------|
| **Grading System** | Class 1-7 (higher = better) | Claiming $2.5k - Stakes (higher $ = better) |
| **Surface Types** | Turf, All-Weather (Polytrack) | Dirt, Turf, Synthetic (various) |
| **Distance Units** | Furlongs (5f-20f typical) | Furlongs + yards (common), or just furlongs |
| **Race Types** | Handicaps, Maidens, Conditions | Claiming, Allowance, Stakes, Maiden |
| **Going** | Firm to Heavy (8 categories) | Fast to Sloppy (6 categories) |
| **Premium Courses** | Ascot, Newmarket, York, Doncaster | Churchill Downs, Belmont, Santa Anita, Keeneland |
| **Pattern Races** | Group 1/2/3, Listed | Grade I/II/III (Graded Stakes) |
| **Data Source** | The Racing API (region='GB') | The Racing API (region='US') |

### Strategic Approach

**Multi-Region Architecture** - Don't replace the UK system, extend it:

1. **Shared Core**: Keep ML model architecture, betting logic, UI framework
2. **Region-Specific**: Add region configuration for scoring, features, mappings
3. **Unified Interface**: Single UI with region filter/selector
4. **Separate Models**: Train UK model and US model independently (different feature distributions)

### Documentation Structure

This expansion plan is broken into focused sections:

- **[US_RACING_DATA_API.md](./US_RACING_DATA_API.md)** - Data source changes, API parameters, field mapping
- **[US_RACING_FEATURES.md](./US_RACING_FEATURES.md)** - Feature engineering adaptations, race scoring differences
- **[US_RACING_IMPLEMENTATION.md](./US_RACING_IMPLEMENTATION.md)** - Step-by-step roadmap, code changes, migration strategy
- **[US_RACING_CODE_EXAMPLES.md](./US_RACING_CODE_EXAMPLES.md)** - Complete code samples for key components

## Quick Wins vs. Long-Term Work

### Phase 1: API & Data (Quick Win - 1-2 days)
- ✅ The Racing API already supports `region='US'`
- Add US data fetching scripts (clone UK versions, change region param)
- Validate US data structure matches expectations
- **Deliverable**: `data/raw/us_races_YYYY.csv`

### Phase 2: Data Cleaning & Mapping (Medium - 3-5 days)
- Map US class system → numeric scoring (see [US_RACING_FEATURES.md](./US_RACING_FEATURES.md))
- Map US going conditions → numeric scale
- Handle dirt vs. turf vs. synthetic surfaces
- Standardize US course names
- **Deliverable**: `data/processed/us_races_cleaned.parquet`

### Phase 3: Feature Engineering (Medium - 3-5 days)
- Adapt career stats (US horses race more frequently)
- Modify course-distance form (dirt vs. turf specialists)
- Adjust class-step logic (claiming → allowance → stakes progression)
- Add surface-switch penalty (turf→dirt or vice versa)
- **Deliverable**: Updated `scripts/phase3_build_horse_model.py` with region support

### Phase 4: Race Scoring (Quick Win - 1-2 days)
- Create US-specific scoring criteria
- Map premium US tracks (see course tier list below)
- Adjust field size expectations (US fields typically smaller)
- **Deliverable**: `scripts/phase2_score_us_races.py`

### Phase 5: ML Model Training (Medium - 2-3 days)
- Train separate US model on US historical data
- Validate features are properly encoded
- Compare UK vs. US model performance
- **Deliverable**: `models/us_horse_model.pkl`

### Phase 6: UI Integration (Medium - 3-4 days)
- Add region selector to Streamlit UI
- Update filters for US-specific values (claiming prices, dirt/turf)
- Dual-load UK and US datasets
- Region-aware predictions display
- **Deliverable**: Updated `predictions.py` with region toggle

### Phase 7: Betting Strategy (Long-Term - 1 week)
- Research US odds formats and bookmaker availability
- Adapt Kelly criterion for US market
- Identify profitable US race types (claiming races? turf stakes?)
- **Deliverable**: US-specific betting recommendations

## US Premium Courses (Tier Classification)

Based on prize money, graded stakes frequency, and racing quality:

### Tier 1 - Elite Tracks
- **Churchill Downs** (KY) - Kentucky Derby, Breeders Cup host
- **Belmont Park** (NY) - Belmont Stakes, premier East Coast
- **Santa Anita** (CA) - Major West Coast stakes
- **Keeneland** (KY) - Spring/Fall meets, high-quality fields
- **Saratoga** (NY) - Summer meet, historic prestige

### Tier 2 - Major Tracks
- **Del Mar** (CA) - Summer meet, Pacific Classic
- **Gulfstream Park** (FL) - Winter racing capital
- **Aqueduct** (NY) - Winter NYC racing
- **Oaklawn Park** (AR) - Arkansas Derby, growing stakes program
- **Fair Grounds** (LA) - Louisiana Derby

### Tier 3 - Regional Tracks
- **Monmouth Park**, **Laurel Park**, **Woodbine** (Canada), **Pimlico** (Preakness only)

## Success Metrics

To validate the US expansion is working:

1. **Data Quality**: Can we clean and standardize 100k+ US races?
2. **Model Performance**: Does US model achieve ROC AUC > 0.65? (UK model: 0.671)
3. **Feature Importance**: Do logical features emerge? (class quality, surface form, track bias)
4. **Profitability Signals**: Do premium stakes races score higher than low-level claimers?

## Next Steps

1. Read [US_RACING_DATA_API.md](./US_RACING_DATA_API.md) for API integration details
2. Review [US_RACING_FEATURES.md](./US_RACING_FEATURES.md) for feature engineering changes
3. Follow [US_RACING_IMPLEMENTATION.md](./US_RACING_IMPLEMENTATION.md) for step-by-step implementation

## Open Questions

- **How frequent is US data?** (Verify API has historical depth comparable to UK)
- **Odds availability**: Does The Odds API cover US tracks comprehensively?
- **Class mapping**: How to score $10k claiming vs. $50k allowance vs. Grade III stakes?
- **Track bias**: US tracks have known biases (speed-favoring vs. closer-favoring) - capture this?

---

See linked documents for detailed technical specifications.
