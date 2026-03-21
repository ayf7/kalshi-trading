# Kalshi Trader

AI-driven automated trading system for [Kalshi](https://kalshi.com) prediction markets. Ingests market data and news, trains ML models to estimate event probabilities, and backtests trading strategies against historical data. Focused on NBA/NFL sports markets.

## Prerequisites

- Python 3.10+
- A Kalshi account with API credentials (for live data ingestion; not needed for synthetic data / backtesting)

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv ~/.env
source ~/.env/bin/activate
```

### 2. Install the package

```bash
cd kalshi-trader
pip install -e ".[dev]"
```

This installs all runtime dependencies (kalshi-python, gdeltdoc, pandas, scikit-learn, xgboost, etc.) and dev dependencies (pytest, ruff).

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your Kalshi API credentials:

```
KALSHI_API_KEY_ID=your-api-key-id-here
KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem
KALSHI_DEMO=true
DB_PATH=data/kalshi.db
```

- Get your API key ID and RSA private key from [Kalshi Account Settings](https://kalshi.com/account/profile).
- Set `KALSHI_DEMO=true` to use Kalshi's demo environment (`demo-api.kalshi.co`) instead of production.
- If you only plan to use synthetic data, you can skip this step entirely.

## Quick Start (No API credentials needed)

```bash
# Generate synthetic market data
python scripts/seed_synthetic.py

# Train a model
python scripts/train_model.py --model logistic -v

# Run a backtest
python scripts/run_backtest.py --start 2025-01-15 --end 2025-02-10 --model logistic
```

## Usage

### Data Ingestion

```bash
# Snapshot market prices once
python scripts/ingest_snapshot.py --once

# Run continuously (every 30s)
python scripts/ingest_snapshot.py

# Fetch news articles
python scripts/ingest_news.py --keywords "Celtics" "Lakers"
```

### Model Training

```bash
python scripts/train_model.py --model logistic      # Logistic regression
python scripts/train_model.py --model xgboost        # XGBoost
```

Models are saved to `data/models/`.

### Backtesting

```bash
python scripts/run_backtest.py --start 2025-01-01 --end 2025-03-01 --model xgboost
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
kalshi-trader/
├── src/kalshi_trader/       # Main package
│   ├── config.py            # App configuration (reads .env)
│   ├── data/                # Data layer (models, DB, API clients, ingestion)
│   ├── features/            # Feature extraction (market + news features)
│   ├── models/              # ML models (logistic regression, XGBoost)
│   ├── strategy/            # Trading strategy + risk management
│   └── backtest/            # Backtesting engine + simulated exchange
├── scripts/                 # CLI entry points
├── config/                  # Default settings + keyword mappings
├── tests/                   # pytest tests
└── data/                    # SQLite database (gitignored)
```
