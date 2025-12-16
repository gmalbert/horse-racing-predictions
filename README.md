# Horse Racing Predictions

Machine learning project for predicting horse racing outcomes using data from The Racing API.

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

## Development

- Follow PEP 8 style guidelines
- Write tests for new functionality
- Keep data collection separate from model training
- Document API rate limits and implement throttling

## Testing

```bash
pytest tests/
```

## License

[Add license information]
