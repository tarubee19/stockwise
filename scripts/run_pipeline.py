#!/usr/bin/env python
"""Run the whole StockWise study and write results/.

    python scripts/run_pipeline.py --data data/m5_foods.zip --results-dir results
    python scripts/run_pipeline.py --data data/synthetic --results-dir results_sim   # simulated

--data is the m5_foods.zip written by scripts/prepare_m5_upload.py, or a folder with the three
M5 CSV files. Item ids that contain SIM are labelled SIMULATED everywhere.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockwise.config import CostConfig, ExperimentConfig  # noqa: E402
from stockwise.data import load_panel  # noqa: E402
from stockwise.pipeline import run_study  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="m5_foods.zip or a folder of M5 CSV files")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--n-jobs", type=int, default=4, help="threads for LightGBM")
    ap.add_argument("--lead-time", type=int, default=2, help="days from order to arrival")
    ap.add_argument("--val-days", type=int, default=120)
    ap.add_argument("--test-days", type=int, default=270)
    ap.add_argument("--train-window", type=int, default=540)
    ap.add_argument("--train-stride", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--holding-rate", type=float, default=0.25)
    ap.add_argument("--shelf-life", type=int, default=7)
    ap.add_argument("--stockout-mult", type=float, default=2.0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig(
        lead_time=args.lead_time,
        val_days=args.val_days,
        test_days=args.test_days,
        train_window=args.train_window,
        train_stride=args.train_stride,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        cost=CostConfig(
            margin=args.margin,
            holding_rate=args.holding_rate,
            shelf_life_days=args.shelf_life,
            stockout_mult=args.stockout_mult,
        ),
    )
    data = load_panel(args.data, cfg)
    print(f"loaded {data.n_series:,} series ({data.dataset} data): {data.report}")
    run_study(data, replace(cfg), results_dir=args.results_dir, figures=not args.no_figures)


if __name__ == "__main__":
    main()
