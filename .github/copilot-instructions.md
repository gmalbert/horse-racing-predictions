# Horse Racing Predictions - AI Coding Agent Instructions

## Project Overview

This is a **horse racing predictions** project using Python. Data is sourced from **The Racing API** (api.theracingapi.com) with HTTP Basic Authentication.

## Development Environment

- **Python Virtual Environment**: Use `.venv/` for all Python dependencies
- **Activation**: 
  - Windows PowerShell: `.venv\Scripts\Activate.ps1`
  - Windows CMD: `.venv\Scripts\activate.bat`
  - Unix/macOS: `source .venv/bin/activate`

## API Configuration

### The Racing API
**Documentation**: https://api.theracingapi.com/documentation
- **Authentication**: HTTP Basic Auth
- **Username**: Store in environment variable `RACING_API_USERNAME`
- **Password**: Store in environment variable `RACING_API_PASSWORD`
- **Rate Limit**: 500 calls per month
- **CRITICAL**: Never commit credentials to git. Use `.env` file (add to `.gitignore`) or environment variables

### The Odds API
**Documentation**: https://the-odds-api.com/liveapi/guides/v4/
- **Authentication**: API Key (query parameter)
- **API Key**: Store in environment variable `ODDS_API_KEY`
- **Rate Limit**: 500 calls per month
- **Limitation**: No historical data available (live odds only)
- **Use Case**: Complement racing predictions with current betting odds
- **CRITICAL**: Never commit credentials to git. Use `.env` file (add to `.gitignore`) or environment variables

## Project Conventions

### When Adding New Code

1. **Python Version**: Verify Python version requirements before adding dependencies
2. **Dependencies**: Add all new packages to `requirements.txt` (create if missing)
3. **Environment Variables**: Use `python-dotenv` to load `.env` file for API credentials
4. **API Calls**: 
   - Racing API: Use `requests` with basic auth: `auth=(username, password)`
   - Odds API: Use `requests` with API key as query parameter: `params={'apiKey': api_key}`
5. **Data Files**: Keep training data, race data, and models in appropriate directories (e.g., `data/`, `models/`)
6. **Notebooks**: If using Jupyter notebooks for exploration, place in `notebooks/` directory
7. **Rate Limiting**: **CRITICAL** - Both APIs limited to 500 calls/month. Implement:
   - Call counting/tracking mechanism
   - Caching of API responses
   - Batch requests when possible
   - Avoid redundant API calls

### Machine Learning Patterns (Expected)

- **Data Pipeline**: Separate data collection, preprocessing, and feature engineering
- **Model Training**: Keep training scripts separate from prediction/inference code
- **Configuration**: Use configuration files (JSON/YAML) for hyperparameters and model settings
- **Evaluation**: Include separate evaluation/validation scripts with metrics

### Code Organization (Recommended)

```
horse-racing-predictions/
├── data/              # Raw and processed data
├── models/            # Saved model files
├── notebooks/         # Exploratory analysis
├── src/               # Source code
│   ├── data/          # Data collection and preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Model definitions and training
│   └── utils/         # Utility functions
├── tests/             # Unit and integration tests
└── requirements.txt   # Python dependencies
```

## Key Commands

**Install Dependencies** (create requirements.txt first if missing):
```bash
pip install -r requirements.txt
```

**Common ML Libraries** for horse racing predictions typically include:
- requests (API calls)
- python-dotenv (environment variables)
- pandas, numpy (data manipulation)
- scikit-learn (ML models)
- xgboost/lightgbm (gradient boosting)
- matplotlib/seaborn (visualization)

## Testing

- Place tests in `tests/` directory
- Use `pytest` as the testing framework
- Run tests: `pytest tests/`

## Notes for AI Agents

- This project is in **initial setup phase** - establish foundational structure first
- Focus on **data quality** and **feature engineering** for racing predictions
- Consider **time-series aspects** of racing data (historical performance)
- Handle **missing data** appropriately (horses without full racing history)
