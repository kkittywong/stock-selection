"""
Utility functions for HSI stock selection.
"""

import pandas as pd
import numpy as np
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_constituents(path: str = None) -> pd.DataFrame:
    """Load HSI constituent stocks from CSV."""
    if path is None:
        path = os.path.join(DATA_DIR, "hsi_constituents.csv")
    df = pd.read_csv(path)
    return df


def load_price_data(path: str = None) -> pd.DataFrame:
    """Load historical price data from CSV.

    Returns a DataFrame with columns:
        date, ticker, open, high, low, close, volume
    """
    if path is None:
        path = os.path.join(DATA_DIR, "price_data.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_fundamental_data(path: str = None) -> pd.DataFrame:
    """Load fundamental data from CSV.

    Returns a DataFrame with columns:
        ticker, pe_ratio, pb_ratio, roe, earnings_growth_yoy,
        revenue_growth_yoy, debt_to_equity, dividend_yield, market_cap_hkd_bn
    """
    if path is None:
        path = os.path.join(DATA_DIR, "fundamental_data.csv")
    df = pd.read_csv(path)
    return df


def compute_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Return simple moving average of *prices* over *window* periods."""
    return prices.rolling(window=window, min_periods=window).mean()


def compute_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    """Return price return over *lookback* periods (momentum signal)."""
    return prices.pct_change(periods=lookback)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    # When avg_loss is 0 and avg_gain > 0, RSI = 100; when both are 0, RSI = 50
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Fill cases where avg_loss == 0: pure uptrend -> RSI = 100
    rsi = rsi.where(avg_loss != 0, other=100.0)
    return rsi


def compute_technical_signals(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for every ticker.

    Returns a DataFrame indexed by ticker with the latest values of:
        close, ma50, ma200, momentum_1m, momentum_3m, momentum_6m, rsi14,
        above_ma50, above_ma200
    """
    records = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        closes = grp["close"]

        ma50 = compute_moving_average(closes, 50).iloc[-1]
        ma200 = compute_moving_average(closes, 200).iloc[-1]
        mom_1m = compute_momentum(closes, 21).iloc[-1]
        mom_3m = compute_momentum(closes, 63).iloc[-1]
        mom_6m = compute_momentum(closes, 126).iloc[-1]
        rsi14 = compute_rsi(closes, 14).iloc[-1]
        last_close = closes.iloc[-1]

        records.append({
            "ticker": ticker,
            "close": last_close,
            "ma50": ma50,
            "ma200": ma200,
            "momentum_1m": mom_1m,
            "momentum_3m": mom_3m,
            "momentum_6m": mom_6m,
            "rsi14": rsi14,
            "above_ma50": int(last_close > ma50) if not np.isnan(ma50) else np.nan,
            "above_ma200": int(last_close > ma200) if not np.isnan(ma200) else np.nan,
        })

    return pd.DataFrame(records).set_index("ticker")


def rank_normalize(series: pd.Series) -> pd.Series:
    """Rank-normalise a series to [0, 1] (higher rank = higher value)."""
    return series.rank(pct=True)
