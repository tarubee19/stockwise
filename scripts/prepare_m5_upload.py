#!/usr/bin/env python
"""Trim the real M5 files to the FOODS category and zip them (about 25 MB) for upload.

The M5 download is large. StockWise studies the FOODS category (grocery-like products, the
closest match to a dark store), so this script keeps only those rows and writes ONE small zip.
Nothing is changed or simulated: the rows are exactly the ones in the original files.

Usage (PowerShell, from any folder):
    python scripts/prepare_m5_upload.py "C:\\path\\to\\folder\\with\\the\\M5\\csv\\files"

It writes m5_foods.zip in the current folder.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", help="folder that contains the M5 CSV files")
    ap.add_argument("--out", default="m5_foods.zip", help="output zip (default m5_foods.zip)")
    args = ap.parse_args()

    src = Path(args.folder)
    sales_path = next(
        (src / n for n in ("sales_train_evaluation.csv", "sales_train_validation.csv") if (src / n).exists()),
        None,
    )
    missing = [n for n in ("calendar.csv", "sell_prices.csv") if not (src / n).exists()]
    if sales_path is None:
        missing.append("sales_train_evaluation.csv")
    if missing:
        sys.exit(f"Not found in {src}: {', '.join(missing)}")

    print(f"Reading {sales_path.name} ...")
    sales = pd.read_csv(sales_path)
    foods = sales[sales["cat_id"] == "FOODS"]
    del sales

    print("Reading sell_prices.csv ...")
    prices = pd.read_csv(src / "sell_prices.csv")
    prices = prices[prices["item_id"].str.startswith("FOODS_")]

    calendar = pd.read_csv(src / "calendar.csv")

    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("calendar.csv", calendar.to_csv(index=False))
        zf.writestr("sell_prices.csv", prices.to_csv(index=False))
        zf.writestr("sales_train_foods.csv", foods.to_csv(index=False))

    d_cols = [c for c in foods.columns if c.startswith("d_")]
    print()
    print(f"sales (FOODS) : {len(foods):,} series x {len(d_cols):,} days   (expect ~14,370 x 1,941)")
    print(f"sell prices   : {len(prices):,} rows")
    print(f"calendar      : {len(calendar):,} days                    (expect ~1,969)")
    print(f"Wrote {out.resolve()}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Attach this one zip file in the chat.")


if __name__ == "__main__":
    main()
