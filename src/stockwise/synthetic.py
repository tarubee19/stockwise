"""Simulated M5-shaped data for tests, CI and the Docker demo.

It exists so the code can run without the real M5 download. Item ids contain ``SIM`` so every
output built from it is labelled SIMULATED. Results from this data say nothing about retail.

The generator mixes weekday and yearly seasonality, price promotions with a demand response,
SNAP and event days, gamma-Poisson (negative binomial) noise, and a share of items that launch
late, so the loader's eligibility filter is exercised too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STORES = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX", "WI_1": "WI"}
DEPTS = ("FOODS_1", "FOODS_2", "FOODS_3")
START_DATE = "2011-01-29"  # a Saturday, like d_1 in M5


def make_m5_frames(
    n_items_per_dept: int = 12,
    n_days: int = 900,
    seed: int = 0,
    late_launch_share: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (calendar, sell_prices, sales) shaped exactly like the M5 files."""
    rng = np.random.default_rng(seed)
    total_days = n_days + 28
    dates = pd.date_range(START_DATE, periods=total_days)
    wm_yr_wk = 11101 + np.arange(total_days) // 7
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    event_mask = rng.random(total_days) < 0.05
    snap = {s: ((dates.day.to_numpy() >= 1 + 3 * i) & (dates.day.to_numpy() <= 10 + 3 * i)).astype(int)
            for i, s in enumerate(("CA", "TX", "WI"))}
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm_yr_wk,
            "weekday": dates.day_name(),
            "wday": (dow + 2) % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in range(total_days)],
            "event_name_1": np.where(event_mask, "SimEvent", None),
            "event_type_1": np.where(event_mask, "Cultural", None),
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": snap["CA"],
            "snap_TX": snap["TX"],
            "snap_WI": snap["WI"],
        }
    )

    weekday_factor = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.3, 1.1])  # Mon..Sun
    week_idx = wm_yr_wk - wm_yr_wk[0]
    n_weeks = int(week_idx.max()) + 1

    sales_rows, price_rows = [], []
    for dept in DEPTS:
        for k in range(n_items_per_dept):
            item_id = f"{dept}_SIM{k:03d}"
            base_rate = float(np.exp(rng.normal(0.4, 0.9)))  # popular items sell several a day
            base_price = float(np.exp(rng.normal(1.1, 0.5)))
            dispersion = float(rng.uniform(0.15, 0.8))
            season_amp = float(rng.uniform(0.0, 0.25))
            season_phase = float(rng.uniform(0, 2 * np.pi))
            for store, state in STORES.items():
                store_factor = float(np.exp(rng.normal(0.0, 0.3)))
                launch = int(rng.integers(120, 400)) if rng.random() < late_launch_share else 0
                promo = rng.random(n_weeks) < 0.08
                depth = np.where(promo, rng.uniform(0.7, 0.9, n_weeks), 1.0)
                weekly_price = np.round(base_price * depth * (1 + rng.normal(0, 0.01, n_weeks)), 2)
                weekly_price = np.maximum(weekly_price, 0.05)
                day_price = weekly_price[week_idx]
                ratio = day_price / base_price
                lam = (
                    base_rate
                    * store_factor
                    * weekday_factor[dow]
                    * (1 + season_amp * np.sin(2 * np.pi * doy / 365.25 + season_phase))
                    * ratio**-1.5
                    * (1 + 0.10 * snap[state])
                    * (1 + 0.20 * event_mask)
                )
                gamma = rng.gamma(1 / dispersion, dispersion, total_days)
                demand = rng.poisson(lam * gamma)[:n_days]
                demand[:launch] = 0
                sales_rows.append(
                    {
                        "id": f"{item_id}_{store}_evaluation",
                        "item_id": item_id,
                        "dept_id": dept,
                        "cat_id": "FOODS",
                        "store_id": store,
                        "state_id": state,
                        **{f"d_{i + 1}": int(v) for i, v in enumerate(demand)},
                    }
                )
                first_week = int(week_idx[launch]) if launch else 0
                for w in range(first_week, n_weeks):
                    price_rows.append(
                        {
                            "store_id": store,
                            "item_id": item_id,
                            "wm_yr_wk": int(wm_yr_wk[0] + w),
                            "sell_price": float(weekly_price[w]),
                        }
                    )

    return calendar, pd.DataFrame(price_rows), pd.DataFrame(sales_rows)


def write_m5_folder(out: str | Path, **kwargs) -> Path:
    """Write the three simulated files with the names ``read_m5`` expects."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    calendar, prices, sales = make_m5_frames(**kwargs)
    calendar.to_csv(out / "calendar.csv", index=False)
    prices.to_csv(out / "sell_prices.csv", index=False)
    sales.to_csv(out / "sales_train_foods.csv", index=False)
    return out
