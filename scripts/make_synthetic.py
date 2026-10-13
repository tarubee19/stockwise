#!/usr/bin/env python
"""Write simulated M5-shaped CSV files (item ids contain SIM; everything built from them is
labelled SIMULATED). Used for development, tests and CI.

    python scripts/make_synthetic.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockwise.synthetic import write_m5_folder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--items-per-dept", type=int, default=40)
    ap.add_argument("--days", type=int, default=1100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = write_m5_folder(
        args.out, n_items_per_dept=args.items_per_dept, n_days=args.days, seed=args.seed
    )
    print(f"Wrote simulated M5-shaped files to {out}")


if __name__ == "__main__":
    main()
