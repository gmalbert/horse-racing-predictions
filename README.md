# Horse Racing Predictions

<p align="left">
  <img src="data/logo.png" alt="Equine Edge" width="300" />
</p>

A comprehensive machine learning system for UK horse racing analysis and betting strategy optimization. Combines historical race data, intelligent scoring algorithms, and predictive models to identify high-value betting opportunities.

---

## Table of Contents

- [Features](#features)
  - [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
  - [Race Profitability Scorer](#race-profitability-scorer)
  - [ML Win Prediction Model](#ml-win-prediction-model)
  - [Fixture Calendar with Predictions](#fixture-calendar-with-predictions)
- [Project Components](#project-components)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Prediction System](#prediction-system)
  - [Betting Strategy Tools](#betting-strategy-tools)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Data Sources](#api-data-sources)
- [Installation](#installation)
- [Development](#development)

---

## Features

### Interactive Streamlit Dashboard

**Launch:** `streamlit run predictions.py`

A full-featured web interface for exploring race data, analyzing predictions, and identifying betting opportunities:

- **📊 Data Explorer**
  - Filter by year, course, horse name, and finish position
  - View detailed race results with jockey, trainer, and time information
  - Summary statistics: horse performance, course trends, jockey stats

- **🏟️ Course Analytics**
  - Race counts and performance by venue
  - Course tier classifications (Premium/Major/Minor)
  - Regional distribution analysis

- **🔮 ML Model Tab**
  - Live win probability predictions using XGBoost
  - 18-feature model (ROC AUC 0.671)
  - Feature importance visualization
  - One-click model retraining

- **� Value Betting with Implied Odds**
  - Automatic conversion of probabilities to betting odds
  - Displays decimal and fractional odds (UK format)
  - Simple fractional odds (denominators ≤2): 1/2, 1/1, 3/2, 2/1, 5/2, etc.
  - Compare model odds vs bookmaker odds to find value bets
  - **Example**: Model shows 3/1, bookmaker offers 5/1 → VALUE BET!

- **�📅 Predicted Fixtures**
  - 1,474 upcoming races scored (Dec 2025 - Dec 2026)
  - 29 Tier 1 Focus races identified
  - Interactive filters: tier, course, minimum score
  - Score distribution visualization
  - Top courses by predicted value

- **🎯 Top Predictive Races**
  - Upcoming Tier 1 Focus races (score ≥70)
  - Sorted chronologically with soonest races first
  - Combines actual + predicted race data
  - Shows date, course, class, prize, distance, score

### Race Profitability Scorer

**Phase 2 Algorithm** - Identifies the most profitable races to bet on:

**Scoring Factors (0-100 points):**
- **Class Quality** (0-30 pts): Class 1 = 30pts, Class 4 = 8pts
- **Prize Money** (0-25 pts): Higher prizes = better fields
- **Course Tier** (0-20 pts): Premium courses = 20pts
- **Field Size** (0-15 pts): Competitive fields preferred
- **Pattern Race Bonus** (0-10 pts): Group/Listed races

**Tier Classification:**
- **Tier 1: Focus** (score ≥70) - 36,838 races - Best betting value
- **Tier 2: Value** (50-69) - 110,801 races - Moderate opportunities
- **Tier 3: Avoid** (<50) - 97,659 races - Low profitability

**Validation:** 3.6 percentage point improvement in top-tier race identification

### ML Win Prediction Model

**Phase 3: XGBoost Classifier** for predicting horse win probability

**Model Performance:**
- Training Accuracy: 88.7%
- Test Accuracy: 88.5%
- ROC AUC: 0.671 (test set)
- Training Data: 203,736 races (Class 1-4, 2015-2025)

**Top 5 Features:**
1. `field_size` - 17.41% importance
2. `avg_last_3_pos` - 7.70%
3. `class_num` - 7.21%
4. `career_place_rate` - 6.34%
5. `or_change` - 5.90%

**18 Total Features:**
- Horse form: last 3 positions, days since last run, career stats
- Race characteristics: class, field size, distance
- Ratings: official rating, RPR, changes over time
- Performance metrics: win rate, place rate, consistency

**Fallback Logic:** Uses RandomForest if XGBoost unavailable

### Fixture Calendar with Predictions

**Intelligent Race Characteristic Prediction** for 1,474 upcoming fixtures

**How It Works:**
1. **Course Profiles**: Built from 245,298 historical races (37 courses)
2. **Smart Predictions**:
   - **Weekend Races**: Upgraded to better class + higher prizes
   - **Seasonal Going**: Winter=softer, summer=firmer
   - **Course-Specific**: Distance, typical class, field size
   - **Prize Money**: 75th percentile for weekends, median for weekdays

**Prediction Quality:**
- Based on 10+ years of data per course
- Weekend/weekday distinction improves accuracy
- Seasonal adjustments match UK weather patterns

**Upcoming Opportunities:**
- **Ascot**: 14 Tier 1 races (score 85.4)
- **York**: 8 Tier 1 races (score 82+)
- **Goodwood**: 7 Tier 1 races (score 74+)

**Run Predictions:** `python scripts/score_fixture_calendar.py`

[Back to Top](#table-of-contents)

---

## Project Components

### Data Processing Pipeline

**Phase 1: Data Validation and Cleaning**
- 630,000+ historical UK races (2015-2025)
- Database optimization: Removed Class 5-7 (61% reduction → 245,298 races)
- Cleaned columns: distance, course names, class, surface, prize money
- Normalized finish positions and going conditions

**Data Files:**
- `data/processed/race_scores.parquet` - Historical scored races
- `data/processed/scored_fixtures_calendar.csv` - Predicted upcoming races
- `data/processed/lookups/course_tiers.csv` - Course quality tiers
- `data/raw/` - Raw API responses (cached)

### Prediction System

**Multi-Layer Approach:**
1. **Race Scoring** (Phase 2): Identifies valuable races
2. **Horse Prediction** (Phase 3): Predicts win probability per horse
3. **Odds Conversion** (Phase 3.5): Converts probabilities to implied odds
4. **Fixture Prediction**: Estimates characteristics for upcoming races

**Scripts:**
- `scripts/phase2_score_races.py` - Score historical races
- `scripts/phase3_build_horse_model.py` - Train ML model
- `scripts/odds_converter.py` - Probability/odds conversion utilities
- `scripts/predict_todays_races.py` - Generate daily predictions with odds
- `scripts/score_fixture_calendar.py` - Predict & score fixtures

**Odds Conversion:**
- **Decimal Odds**: 1 / probability (e.g., 25% → 4.0)
- **Fractional Odds**: Simplified to UK standard formats (e.g., 25% → 3/1)
- **American Odds**: ±100 × probability ratio (e.g., 25% → +300)
- **Value Bet Detection**: Identifies when bookmaker odds exceed model odds

### Betting Strategy Tools

**Upcoming (Phase 4-6):**
- Kelly Criterion staking system
- Bankroll management
- Performance backtesting
- Production deployment automation

[Back to Top](#table-of-contents)

---

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd horse-racing-predictions

# 2. Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch dashboard
streamlit run predictions.py
```

The app will open in your browser at `http://localhost:8501`

Note: The dashboard now includes a "Today & Tomorrow" predictions tab. It can generate predictions for the current date and for the next day; the UI only shows fetch/generate controls for days that still need data.


[Back to Top](#table-of-contents)

---

## Project Structure

```
horse-racing-predictions/
├── predictions.py              # Streamlit dashboard (main app)
├── requirements.txt            # Python dependencies
├── .env.example               # API credentials template
│
├── data/
│   ├── raw/                   # Cached API responses
│   │   ├── 2025_racing_fixture_list.ics
│   │   └── 2026_fixture_list.xlsx
│   ├── processed/             # Cleaned datasets
│   │   ├── race_scores.parquet          # 245K scored races
│   │   ├── scored_fixtures_calendar.csv # 1,474 predicted fixtures
│   │   └── lookups/
│   │       └── course_tiers.csv         # Course classifications
│   └── logo.png               # App branding
│
├── models/                    # Trained ML models
│   ├── horse_win_predictor.pkl         # XGBoost classifier
│   ├── model_metadata.pkl              # Training info
│   ├── feature_importance.csv          # Feature rankings
│   └── feature_columns.txt             # Model features
│
├── scripts/                   # Data processing & training
│   ├── phase2_score_races.py           # Race profitability scorer
│   ├── phase3_build_horse_model.py     # ML model training
│   ├── odds_converter.py               # Probability/odds conversion
│   ├── predict_todays_races.py         # Daily predictions with odds
│   ├── score_fixture_calendar.py       # Predict upcoming races
│   ├── extract_2025_recent_fixtures.py # Parse .ics calendar
│   └── extract_bha_2026_four_courses.py # Parse Excel fixtures
│
├── examples/                  # API usage demos
│   ├── api_example.py         # The Racing API
│   └── odds_api_example.py    # The Odds API
│
├── tests/                     # Unit tests
│   ├── conftest.py           # Test fixtures
│   └── fixtures/             # Mock API responses
│
└── src/                       # Source modules (future)
    ├── data/                 # Data collection
    ├── features/             # Feature engineering
    ├── models/               # Model definitions
    └── utils/                # Helper functions
```

[Back to Top](#table-of-contents)

---

## API Data Sources

### The Racing API

[The Racing API](https://api.theracingapi.com/documentation) provides comprehensive UK horse racing data.

- **Authentication**: HTTP Basic Auth
- **Credentials**: `RACING_API_USERNAME` and `RACING_API_PASSWORD` in `.env`
- **Rate Limit**: 500 calls/month
- **Data Coverage**:
  - Race schedules and results
  - Horse/jockey/trainer information
  - Historical performance data
  - Course details and conditions

### The Odds API

[The Odds API](https://the-odds-api.com/liveapi/guides/v4/) provides live betting odds.

- **Authentication**: API Key (`ODDS_API_KEY` in `.env`)
- **Rate Limit**: 500 calls/month
- **Data Coverage**: Real-time odds (live only - no historical data)
- **Use Case**: Combine with predictions to identify value bets

⚠️ **Important**: Both APIs are limited to 500 calls/month. Implement caching and batching to avoid quota issues.

**Best Practices:**
- Save raw responses to `data/raw/` for reuse
- Use cached data in tests (see `tests/fixtures/`)
- Batch API calls where possible
- Monitor usage with `examples/api_example.py`

[Back to Top](#table-of-contents)

---

## Installation

### Prerequisites

- Python 3.8+ (tested on 3.12-3.13)
- pip package manager
- Git

### Full Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd horse-racing-predictions
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   
   # Windows CMD
   .venv\Scripts\activate.bat
   
   # Unix/macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Key Packages:**
   - `streamlit` - Web dashboard
   - `pandas`, `numpy` - Data processing
   - `xgboost` - Machine learning
   - `plotly` - Interactive charts
   - `scikit-learn` - ML utilities
   - `openpyxl` - Excel file support
   - `python-dotenv` - Environment variables

4. **Configure API Credentials (Optional)**
   
   Only needed if pulling live data from APIs:
   
   ```bash
   # Copy template
   copy .env.example .env  # Windows
   cp .env.example .env    # Unix/macOS
   
   # Edit .env and add:
   RACING_API_USERNAME=your_username
   RACING_API_PASSWORD=your_password
   ODDS_API_KEY=your_api_key
   ```

5. **Run the Dashboard**
   ```bash
   streamlit run predictions.py
   ```
   
   Opens at: `http://localhost:8501`

### Data Files

The repository includes pre-processed data, so API credentials are **optional** unless you want to fetch new data.

**Included:**
- Historical races (2015-2025): `data/processed/race_scores.parquet`
- Predicted fixtures: `data/processed/scored_fixtures_calendar.csv`
- Trained ML model: `models/horse_win_predictor.pkl`

**To regenerate:**
```bash
# Score historical races (if you have new data)
python scripts/phase2_score_races.py

# Retrain ML model
python scripts/phase3_build_horse_model.py

# Score upcoming fixtures
python scripts/score_fixture_calendar.py

# Generate daily predictions (script now accepts a `--date` parameter)
python scripts/predict_todays_races.py            # predicts for today (default)
python scripts/predict_todays_races.py --date 2026-01-01  # predict for a specific date
```

[Back to Top](#table-of-contents)

---

## Development

### Testing

```bash
# Run all tests
pytest tests/

# Test with coverage
pytest --cov=src tests/
```

**Test Strategy:**
- Use saved fixtures in `tests/fixtures/` (no live API calls)
- Mock `requests.get()` with `tests/conftest.py`
- Add new test fixtures to `data/raw/` for deterministic tests

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document complex algorithms
- Keep functions focused and testable

### Adding Features

1. **Data Collection**: Add scripts to `scripts/`
2. **Feature Engineering**: Create pure functions in `src/features/`
3. **Models**: Place training logic in `src/models/`
4. **UI**: Update `predictions.py` with new tabs/visualizations

### Pre-commit Checks

```bash
# Syntax check all Python files
python -m py_compile predictions.py

# Or check all files
python -c "import py_compile,glob; [py_compile.compile(p, doraise=True) for p in glob.glob('**/*.py', recursive=True)]"
```

### Rate Limit Management

Both APIs have 500 calls/month limits:

- **Cache aggressively**: Save all responses to `data/raw/`
- **Batch requests**: Combine related calls
- **Use examples sparingly**: Run `examples/*.py` only when needed
- **Monitor usage**: Track API call counts

### Contributing

- Create feature branches from `main`
- Test thoroughly before merging
- Update README for significant features
- Document API usage patterns

---

**Project Status:** Active Development

[Back to Top](#table-of-contents)

**Current Phase:** Phase 3 complete (ML model), Phase 4 in progress (betting strategy)

[Back to Top](#table-of-contents)

**License:** See repository for license information

[Back to Top](#table-of-contents)