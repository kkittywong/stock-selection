"""
HSI Stock Selection package.
"""

from .stock_selector import HSIStockSelector
from .long_selection import select_long_book
from .short_selection import select_short_book
from .utils import (
    load_constituents,
    load_price_data,
    load_fundamental_data,
    compute_technical_signals,
    compute_moving_average,
    compute_momentum,
    compute_rsi,
    rank_normalize,
)

__all__ = [
    "HSIStockSelector",
    "select_long_book",
    "select_short_book",
    "load_constituents",
    "load_price_data",
    "load_fundamental_data",
    "compute_technical_signals",
    "compute_moving_average",
    "compute_momentum",
    "compute_rsi",
    "rank_normalize",
]
