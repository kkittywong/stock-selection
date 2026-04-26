"""
Tests for HSI stock selection.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Allow imports from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    compute_moving_average,
    compute_momentum,
    compute_rsi,
    rank_normalize,
    compute_technical_signals,
    load_constituents,
    load_price_data,
    load_fundamental_data,
)
from src.long_selection import select_long_book
from src.short_selection import select_short_book
from src.stock_selector import HSIStockSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """Simple deterministic price series (50 periods)."""
    np.random.seed(0)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = ["AAA.HK", "BBB.HK", "CCC.HK"]
    rows = []
    for t in tickers:
        base = 100.0
        for d in dates:
            base *= 1 + np.random.normal(0.001, 0.01)
            rows.append({"date": d, "ticker": t, "close": round(base, 4),
                          "open": base, "high": base * 1.01,
                          "low": base * 0.99, "volume": 1_000_000})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_fund():
    tickers = ["AAA.HK", "BBB.HK", "CCC.HK"]
    data = {
        "ticker": tickers,
        "pe_ratio": [10.0, 30.0, 60.0],
        "pb_ratio": [1.0, 2.0, 5.0],
        "roe": [0.20, 0.10, -0.05],
        "earnings_growth_yoy": [0.30, 0.05, -0.20],
        "revenue_growth_yoy": [0.20, 0.03, -0.10],
        "debt_to_equity": [0.5, 1.0, 2.5],
        "dividend_yield": [0.04, 0.02, 0.00],
        "market_cap_hkd_bn": [500, 200, 50],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

class TestComputeMovingAverage:
    def test_basic(self):
        s = pd.Series(range(1, 11), dtype=float)
        ma = compute_moving_average(s, 3)
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[1])
        assert ma.iloc[2] == pytest.approx(2.0)
        assert ma.iloc[-1] == pytest.approx(9.0)

    def test_single_value_window(self):
        s = pd.Series([5.0, 10.0, 15.0])
        ma = compute_moving_average(s, 1)
        pd.testing.assert_series_equal(ma, s)


class TestComputeMomentum:
    def test_positive_momentum(self):
        s = pd.Series([100.0, 110.0, 121.0])
        mom = compute_momentum(s, 1)
        assert mom.iloc[-1] == pytest.approx(0.1)

    def test_negative_momentum(self):
        s = pd.Series([100.0, 90.0, 81.0])
        mom = compute_momentum(s, 1)
        assert mom.iloc[-1] == pytest.approx(-0.1)


class TestComputeRSI:
    def test_all_up(self):
        s = pd.Series(list(range(1, 51)), dtype=float)
        rsi = compute_rsi(s, 14)
        # All up: RSI should approach 100
        assert rsi.dropna().iloc[-1] > 90

    def test_all_down(self):
        s = pd.Series(list(range(50, 0, -1)), dtype=float)
        rsi = compute_rsi(s, 14)
        # All down: RSI should approach 0
        assert rsi.dropna().iloc[-1] < 10


class TestRankNormalize:
    def test_range(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = rank_normalize(s)
        assert normed.min() > 0
        assert normed.max() <= 1.0

    def test_monotone(self):
        s = pd.Series([10.0, 20.0, 30.0])
        normed = rank_normalize(s)
        assert normed.iloc[0] < normed.iloc[1] < normed.iloc[2]


class TestComputeTechnicalSignals:
    def test_returns_expected_columns(self, sample_prices):
        tech = compute_technical_signals(sample_prices)
        expected = {
            "close", "ma50", "ma200", "momentum_1m", "momentum_3m",
            "momentum_6m", "rsi14", "above_ma50", "above_ma200",
        }
        assert expected.issubset(set(tech.columns))

    def test_index_is_tickers(self, sample_prices):
        tech = compute_technical_signals(sample_prices)
        assert set(tech.index) == {"AAA.HK", "BBB.HK", "CCC.HK"}

    def test_above_ma_binary(self, sample_prices):
        tech = compute_technical_signals(sample_prices)
        valid = tech["above_ma50"].dropna()
        assert valid.isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Long selection tests
# ---------------------------------------------------------------------------

class TestSelectLongBook:
    def test_returns_dataframe(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_long_book(tech, sample_fund, top_n=2)
        assert isinstance(result, pd.DataFrame)

    def test_top_n_respected(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_long_book(tech, sample_fund, top_n=2)
        assert len(result) <= 2

    def test_has_composite_score(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_long_book(tech, sample_fund, top_n=3)
        assert "composite_score" in result.columns

    def test_sorted_descending(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_long_book(tech, sample_fund, top_n=3)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rsi_filter(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        # Set all RSI to 0 to pass overbought filter trivially
        tech["rsi14"] = 0
        result = select_long_book(tech, sample_fund, top_n=3, rsi_overbought_cap=80)
        assert not result.empty


# ---------------------------------------------------------------------------
# Short selection tests
# ---------------------------------------------------------------------------

class TestSelectShortBook:
    def test_returns_dataframe(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_short_book(tech, sample_fund, top_n=2)
        assert isinstance(result, pd.DataFrame)

    def test_top_n_respected(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_short_book(tech, sample_fund, top_n=2)
        assert len(result) <= 2

    def test_has_composite_score(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_short_book(tech, sample_fund, top_n=3)
        assert "composite_score" in result.columns

    def test_sorted_descending(self, sample_prices, sample_fund):
        tech = compute_technical_signals(sample_prices)
        result = select_short_book(tech, sample_fund, top_n=3)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# HSIStockSelector integration tests
# ---------------------------------------------------------------------------

class TestHSIStockSelector:
    def test_run_returns_both_books(self):
        selector = HSIStockSelector(long_top_n=5, short_top_n=5)
        results = selector.run()
        assert "long_book" in results
        assert "short_book" in results

    def test_no_overlap_between_books(self):
        selector = HSIStockSelector(long_top_n=10, short_top_n=10)
        results = selector.run()
        long_tickers = set(results["long_book"].index)
        short_tickers = set(results["short_book"].index)
        assert len(long_tickers & short_tickers) == 0

    def test_books_are_dataframes(self):
        selector = HSIStockSelector(long_top_n=5, short_top_n=5)
        results = selector.run()
        assert isinstance(results["long_book"], pd.DataFrame)
        assert isinstance(results["short_book"], pd.DataFrame)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_constituents(self):
        df = load_constituents()
        assert not df.empty
        assert "ticker" in df.columns

    def test_load_price_data(self):
        df = load_price_data()
        assert not df.empty
        assert {"date", "ticker", "close"}.issubset(df.columns)

    def test_load_fundamental_data(self):
        df = load_fundamental_data()
        assert not df.empty
        assert {"ticker", "pe_ratio", "roe"}.issubset(df.columns)
