"""
Entry point for HSI Stock Selection.

Usage
-----
    python main.py
    python main.py --long-n 15 --short-n 15
    python main.py --price data/my_prices.csv --fundamental data/my_fund.csv
"""

import argparse
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.stock_selector import HSIStockSelector


def parse_args():
    parser = argparse.ArgumentParser(
        description="HSI long/short book stock selection"
    )
    parser.add_argument(
        "--price",
        default=None,
        help="Path to price data CSV (default: data/price_data.csv)",
    )
    parser.add_argument(
        "--fundamental",
        default=None,
        help="Path to fundamental data CSV (default: data/fundamental_data.csv)",
    )
    parser.add_argument(
        "--constituents",
        default=None,
        help="Path to HSI constituents CSV (default: data/hsi_constituents.csv)",
    )
    parser.add_argument(
        "--long-n",
        type=int,
        default=10,
        help="Number of stocks in the long book (default: 10)",
    )
    parser.add_argument(
        "--short-n",
        type=int,
        default=10,
        help="Number of stocks in the short book (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save CSV output files (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selector = HSIStockSelector(
        price_path=args.price,
        fundamental_path=args.fundamental,
        constituents_path=args.constituents,
        long_top_n=args.long_n,
        short_top_n=args.short_n,
    )

    results = selector.run()
    selector.print_results(results)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        long_path = os.path.join(args.output_dir, "long_book.csv")
        short_path = os.path.join(args.output_dir, "short_book.csv")
        results["long_book"].to_csv(long_path)
        results["short_book"].to_csv(short_path)
        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
