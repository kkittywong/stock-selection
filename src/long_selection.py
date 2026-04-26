"""
Long book selection for HSI stock selection strategy.

Selection criteria (higher composite score = stronger long candidate):
  - Positive price momentum (1m, 3m, 6m)
  - Price trading above 50-day and 200-day moving averages
  - RSI not in extreme overbought territory (< 75)
  - Low-to-moderate P/E ratio (cheap relative to peers)
  - High return on equity (ROE)
  - Positive earnings and revenue growth
  - Lower debt-to-equity ratio
"""

import pandas as pd
import numpy as np

from .utils import rank_normalize


# ---------------------------------------------------------------------------
# Weights for each factor (must sum to 1.0 for interpretability)
# ---------------------------------------------------------------------------
LONG_FACTOR_WEIGHTS = {
    "momentum_composite": 0.30,
    "trend_composite": 0.20,
    "pe_score": 0.15,
    "roe_score": 0.15,
    "earnings_growth_score": 0.10,
    "revenue_growth_score": 0.05,
    "debt_score": 0.05,
}


def _momentum_composite(tech: pd.DataFrame) -> pd.Series:
    """Combine short-, medium-, and long-term momentum into one score."""
    mom_1m = rank_normalize(tech["momentum_1m"].dropna())
    mom_3m = rank_normalize(tech["momentum_3m"].dropna())
    mom_6m = rank_normalize(tech["momentum_6m"].dropna())
    combined = (mom_1m + mom_3m + mom_6m).reindex(tech.index)
    return rank_normalize(combined)


def _trend_composite(tech: pd.DataFrame) -> pd.Series:
    """Score based on price position relative to MAs and RSI."""
    above_ma50 = tech["above_ma50"].fillna(0)
    above_ma200 = tech["above_ma200"].fillna(0)
    # Penalise extremely overbought RSI for longs
    rsi_penalty = (tech["rsi14"] > 75).astype(int)
    trend = above_ma50 + above_ma200 - rsi_penalty
    return rank_normalize(trend)


def _pe_score(fund: pd.DataFrame) -> pd.Series:
    """Lower P/E is better for longs (inverse rank)."""
    return 1 - rank_normalize(fund["pe_ratio"])


def _roe_score(fund: pd.DataFrame) -> pd.Series:
    """Higher ROE is better for longs."""
    return rank_normalize(fund["roe"])


def _earnings_growth_score(fund: pd.DataFrame) -> pd.Series:
    """Higher earnings growth is better for longs."""
    return rank_normalize(fund["earnings_growth_yoy"])


def _revenue_growth_score(fund: pd.DataFrame) -> pd.Series:
    """Higher revenue growth is better for longs."""
    return rank_normalize(fund["revenue_growth_yoy"])


def _debt_score(fund: pd.DataFrame) -> pd.Series:
    """Lower debt-to-equity is better for longs (inverse rank)."""
    return 1 - rank_normalize(fund["debt_to_equity"])


def select_long_book(
    tech: pd.DataFrame,
    fund: pd.DataFrame,
    top_n: int = 10,
    rsi_overbought_cap: float = 80.0,
) -> pd.DataFrame:
    """Select the top long candidates.

    Parameters
    ----------
    tech : pd.DataFrame
        Technical signals indexed by ticker (output of ``compute_technical_signals``).
    fund : pd.DataFrame
        Fundamental data with ticker as index or column.
    top_n : int
        Number of stocks to include in the long book.
    rsi_overbought_cap : float
        Stocks with RSI above this threshold are excluded from the long book.

    Returns
    -------
    pd.DataFrame
        Ranked long book with composite scores and individual factor scores.
    """
    if "ticker" in fund.columns:
        fund = fund.set_index("ticker")

    # Align universe to tickers present in both datasets
    tickers = tech.index.intersection(fund.index)
    tech = tech.loc[tickers].copy()
    fund = fund.loc[tickers].copy()

    # Hard filter: exclude stocks that are extremely overbought
    valid_mask = tech["rsi14"].fillna(50) < rsi_overbought_cap
    tech = tech.loc[valid_mask]
    fund = fund.loc[valid_mask]

    if tech.empty:
        return pd.DataFrame()

    # Compute individual factor scores
    scores = pd.DataFrame(index=tech.index)
    scores["momentum_composite"] = _momentum_composite(tech)
    scores["trend_composite"] = _trend_composite(tech)
    scores["pe_score"] = _pe_score(fund)
    scores["roe_score"] = _roe_score(fund)
    scores["earnings_growth_score"] = _earnings_growth_score(fund)
    scores["revenue_growth_score"] = _revenue_growth_score(fund)
    scores["debt_score"] = _debt_score(fund)

    # Weighted composite score
    composite = sum(
        scores[factor] * weight
        for factor, weight in LONG_FACTOR_WEIGHTS.items()
    )
    scores["composite_score"] = composite

    # Merge with fundamental and technical info for transparency
    result = scores.copy()
    result["close"] = tech["close"]
    result["momentum_1m"] = tech["momentum_1m"]
    result["momentum_3m"] = tech["momentum_3m"]
    result["momentum_6m"] = tech["momentum_6m"]
    result["rsi14"] = tech["rsi14"]
    result["above_ma50"] = tech["above_ma50"]
    result["above_ma200"] = tech["above_ma200"]
    result["pe_ratio"] = fund["pe_ratio"]
    result["pb_ratio"] = fund["pb_ratio"]
    result["roe"] = fund["roe"]
    result["earnings_growth_yoy"] = fund["earnings_growth_yoy"]
    result["dividend_yield"] = fund["dividend_yield"]

    result.sort_values("composite_score", ascending=False, inplace=True)
    result["long_rank"] = range(1, len(result) + 1)

    return result.head(top_n)
