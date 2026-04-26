# stock-selection

HSI (Hang Seng Index) quantitative stock selection system for **long book** and **short book** construction.

## Overview

This project implements a factor-based stock selection framework for HSI constituent stocks.  
It combines **technical signals** (momentum, moving averages, RSI) with **fundamental signals** (P/E, ROE, earnings growth, leverage) to rank stocks and produce a long book and a short book.

### Long book selection

Favours stocks with:
- Positive price momentum (1m, 3m, 6m)
- Price trading above the 50-day and 200-day moving averages
- RSI below 80 (not extremely overbought)
- Low-to-moderate P/E ratio (cheap relative to peers)
- High return on equity (ROE)
- Positive earnings and revenue growth
- Lower debt-to-equity ratio

### Short book selection

Favours stocks with:
- Negative price momentum (1m, 3m, 6m)
- Price trading below the 50-day and 200-day moving averages
- RSI above 20 (not extremely oversold — mean-reversion risk)
- High P/E ratio (expensive relative to peers)
- Low or negative ROE
- Declining earnings and revenue
- High debt-to-equity ratio

---

## Project structure

```
stock-selection/
├── data/
│   ├── hsi_constituents.csv      # HSI constituent tickers and sectors
│   ├── price_data.csv            # Daily OHLCV data for all constituents
│   └── fundamental_data.csv      # Per-ticker fundamental metrics
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Data loading and technical indicator helpers
│   ├── long_selection.py         # Long book factor scoring and ranking
│   ├── short_selection.py        # Short book factor scoring and ranking
│   └── stock_selector.py         # Orchestration (HSIStockSelector class)
├── tests/
│   └── test_stock_selection.py   # Unit and integration tests (pytest)
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with bundled sample data
python main.py

# Customise book size
python main.py --long-n 15 --short-n 15

# Use your own CSV files
python main.py --price my_prices.csv --fundamental my_fund.csv

# Save results to CSV
python main.py --output-dir results/
```

### CSV file formats

**`price_data.csv`** — one row per ticker-date:

| Column  | Type        | Description              |
|---------|-------------|--------------------------|
| date    | YYYY-MM-DD  | Trading date             |
| ticker  | string      | e.g. `0005.HK`           |
| open    | float       | Opening price (HKD)      |
| high    | float       | Intraday high            |
| low     | float       | Intraday low             |
| close   | float       | Closing price            |
| volume  | int         | Shares traded            |

**`fundamental_data.csv`** — one row per ticker:

| Column               | Type   | Description                   |
|----------------------|--------|-------------------------------|
| ticker               | string | e.g. `0005.HK`                |
| pe_ratio             | float  | Price-to-earnings ratio       |
| pb_ratio             | float  | Price-to-book ratio           |
| roe                  | float  | Return on equity (decimal)    |
| earnings_growth_yoy  | float  | YoY earnings growth (decimal) |
| revenue_growth_yoy   | float  | YoY revenue growth (decimal)  |
| debt_to_equity       | float  | Debt-to-equity ratio          |
| dividend_yield       | float  | Dividend yield (decimal)      |
| market_cap_hkd_bn    | float  | Market cap in HKD billions    |

**`hsi_constituents.csv`** — one row per constituent:

| Column | Type   | Description            |
|--------|--------|------------------------|
| ticker | string | e.g. `0005.HK`         |
| name   | string | Company name           |
| sector | string | GICS sector            |

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```
