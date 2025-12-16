# Horse Racing Predictions

<p align="left">
  <img src="data/logo.png" alt="Equine Edge" width="300" />
</p>
Machine learning project and interactive Streamlit app for predicting horse racing outcomes using data from The Racing API.

Predictions app overview

- Purpose: provide an interactive UI to explore historical UK race results, compute simple summary statistics, and serve as a place to plug in prediction models.
- How it works: the app loads `data/processed/uk_horse_races.csv`, caches it for performance, exposes sidebar filters (Year, Course, Horse name, Finish Position) and a main-page results limiter. Summary tabs compute aggregated statistics from the full set of filtered data (wins, place, show, total prize, races per course, jockey stats, etc.).
- Where to look: the app source is `predictions.py` (root). Example API usage is in `examples/api_example.py` and `examples/odds_api_example.py`.

Run the app locally:

```bash
streamlit run predictions.py
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd horse-racing-predictions
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   
   # Unix/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API credentials**
   ```bash
   # Copy the example environment file
   copy .env.example .env  # Windows
   # cp .env.example .env  # Unix/macOS
   
   # Edit .env and add your API credentials:
   # - RACING_API_USERNAME and RACING_API_PASSWORD (The Racing API)
   # - ODDS_API_KEY (The Odds API)
   ```

## Project Structure

```
horse-racing-predictions/
├── .env                    # API credentials (not in git)
├── .env.example           # Template for environment variables
├── requirements.txt       # Python dependencies
├── examples/              # Example scripts
│   ├── api_example.py    # Racing API usage demonstration
│   └── odds_api_example.py # Odds API usage demonstration
├── data/                  # Data storage (gitignored)
│   ├── raw/              # Raw API responses
│   └── processed/        # Cleaned/transformed data
├── models/               # Trained models (gitignored)
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── data/            # Data collection and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model definitions and training
│   └── utils/           # Utility functions
└── tests/               # Unit and integration tests
```

## API Data Sources

### The Racing API

This project uses [The Racing API](https://api.theracingapi.com/documentation) for race data and results.

- **Authentication**: HTTP Basic Auth
- **Credentials**: `RACING_API_USERNAME` and `RACING_API_PASSWORD` in `.env` file
- **Rate Limit**: 500 calls per month
- **Data**: Race schedules, results, horse/jockey/trainer information

### The Odds API

This project uses [The Odds API](https://the-odds-api.com/liveapi/guides/v4/) for live betting odds.

- **Authentication**: API Key as query parameter
- **Credentials**: `ODDS_API_KEY` in `.env` file
- **Rate Limit**: 500 calls per month
- **Limitation**: Live odds only - **no historical data available**
- **Use Case**: Real-time odds to complement predictions

⚠️ **Important**: Both APIs are limited to 500 calls per month. Implement caching and rate limiting to avoid exceeding quotas.

## Usage

Run the example scripts to test API connectivity:

```bash
# Test The Racing API
python examples/api_example.py

# Test The Odds API
python examples/odds_api_example.py
```

## Streamlit Predictions UI

A simple Streamlit app is included to explore the processed dataset and run placeholder predictions.

- App file: `predictions.py` (run from the repository root)
- Data source: `data/processed/uk_horse_races.csv` (loaded and cached by the app)
- Logo: `data/logo.png` is shown at the top of the app if present
- Sidebar filters: Year, Course, Horse Name (contains), Finish Position
- Main page: choose number of results to display (25, 50, 75, 100, All) and view the top results sorted by `Date` (descending)
- Summary tabs: Horse performance (wins/place/show, avg finish, total prize), Course statistics (races per course), Jockey performance, and Overall statistics (totals and distributions). All summaries respond to filters.

Run the app:

```bash
streamlit run predictions.py
```

## Development

- Follow PEP 8 style guidelines
- Write tests for new functionality
- Keep data collection separate from model training
- Document API rate limits and implement throttling
