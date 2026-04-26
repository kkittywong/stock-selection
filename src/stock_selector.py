"""
Main stock selector that orchestrates long and short book selection.
"""

import pandas as pd

from .utils import (
    load_constituents,
    load_price_data,
    load_fundamental_data,
    compute_technical_signals,
)
from .long_selection import select_long_book
from .short_selection import select_short_book


class HSIStockSelector:
    """Orchestrate HSI long/short book selection.

    Parameters
    ----------
    price_path : str, optional
        Path to the price data CSV.  Falls back to the bundled sample file.
    fundamental_path : str, optional
        Path to the fundamental data CSV.  Falls back to the bundled sample file.
    constituents_path : str, optional
        Path to the HSI constituents CSV.  Falls back to the bundled sample file.
    long_top_n : int
        Number of stocks to include in the long book (default 10).
    short_top_n : int
        Number of stocks to include in the short book (default 10).
    """

    def __init__(
        self,
        price_path: str = None,
        fundamental_path: str = None,
        constituents_path: str = None,
        long_top_n: int = 10,
        short_top_n: int = 10,
    ):
        self.long_top_n = long_top_n
        self.short_top_n = short_top_n

        self.constituents = load_constituents(constituents_path)
        self.price_data = load_price_data(price_path)
        self.fundamental_data = load_fundamental_data(fundamental_path)

        # Filter price data to HSI universe only
        hsi_tickers = set(self.constituents["ticker"])
        self.price_data = self.price_data[
            self.price_data["ticker"].isin(hsi_tickers)
        ].copy()

    def run(self) -> dict:
        """Run the full selection pipeline.

        Returns
        -------
        dict with keys ``"long_book"`` and ``"short_book"``, each containing
        a :class:`pandas.DataFrame` of selected stocks.
        """
        tech = compute_technical_signals(self.price_data)
        fund = self.fundamental_data.copy()

        long_book = select_long_book(tech, fund, top_n=self.long_top_n)
        short_book = select_short_book(tech, fund, top_n=self.short_top_n)

        # Ensure no overlap between books
        long_tickers = set(long_book.index)
        short_book = short_book[~short_book.index.isin(long_tickers)]

        return {
            "long_book": long_book,
            "short_book": short_book,
        }

    def print_results(self, results: dict = None) -> None:
        """Pretty-print the long and short books."""
        if results is None:
            results = self.run()

        long_book = results["long_book"]
        short_book = results["short_book"]

        display_long_cols = [
            "long_rank", "close", "momentum_1m", "momentum_3m", "momentum_6m",
            "rsi14", "pe_ratio", "roe", "earnings_growth_yoy",
            "composite_score",
        ]
        display_short_cols = [
            "short_rank", "close", "momentum_1m", "momentum_3m", "momentum_6m",
            "rsi14", "pe_ratio", "roe", "earnings_growth_yoy",
            "composite_score",
        ]

        print("=" * 80)
        print("LONG BOOK SELECTION")
        print("=" * 80)
        cols = [c for c in display_long_cols if c in long_book.columns]
        print(long_book[cols].to_string())

        print()
        print("=" * 80)
        print("SHORT BOOK SELECTION")
        print("=" * 80)
        cols = [c for c in display_short_cols if c in short_book.columns]
        print(short_book[cols].to_string())
