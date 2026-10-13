"""Load, validate and reshape the M5 files into aligned numpy matrices.

M5 (Walmart, public Kaggle competition) comes as three files:

* ``sales_train_*.csv``: one row per store x item, one column per day (``d_1`` ...), unit sales
* ``calendar.csv``: one row per day with the date, week id, events and SNAP flags per state
* ``sell_prices.csv``: the selling price of an item in a store for each week it was listed

StockWise studies the FOODS category. ``read_m5`` accepts the folder of CSV files or the single
``m5_foods.zip`` written by ``scripts/prepare_m5_upload.py``.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CALENDAR_EXTRA_DAYS, ExperimentConfig

DATASET_REAL = "real"
DATASET_SIMULATED = "simulated"

SALES_FILES = ("sales_train_foods.csv", "sales_train_evaluation.csv", "sales_train_validation.csv")
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


class DataValidationError(ValueError):
    """Raised when the input files do not look like M5 data."""


@dataclass
class RawM5:
    calendar: pd.DataFrame
    prices: pd.DataFrame
    sales: pd.DataFrame


@dataclass
class M5Data:
    """Aligned matrices for the series kept in the study (rows = series, columns = days)."""

    dataset: str
    ids: pd.DataFrame  # id, item_id, dept_id, store_id, state_id (one row per series)
    sales: np.ndarray  # (n_series, n_days) float32 units sold
    price: np.ndarray  # (n_series, n_days + extra) float32 price, filled across gaps
    dates: pd.DatetimeIndex  # (n_days + extra,) including the appended future days
    dow: np.ndarray  # (n_days + extra,) day of week, Monday = 0
    month: np.ndarray  # (n_days + extra,)
    event: np.ndarray  # (n_days + extra,) number of calendar events that day
    snap: np.ndarray  # (n_series, n_days + extra) SNAP day flag for the series' state
    n_days: int
    report: dict = field(default_factory=dict)  # how many series were dropped and why

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]


# ----------------------------------------------------------------------------- reading


def _read_csv_from(source: Path, name: str) -> pd.DataFrame:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = {Path(n).name: n for n in zf.namelist()}
            if name not in members:
                raise FileNotFoundError(name)
            with zf.open(members[name]) as fh:
                return pd.read_csv(io.BytesIO(fh.read()))
    path = source / name
    if not path.exists():
        raise FileNotFoundError(name)
    return pd.read_csv(path)


def read_m5(source: str | Path) -> RawM5:
    """Read the three M5 tables from a folder or a zip file."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    sales = None
    for name in SALES_FILES:
        try:
            sales = _read_csv_from(source, name)
            break
        except FileNotFoundError:
            continue
    if sales is None:
        raise DataValidationError(f"no sales file ({' / '.join(SALES_FILES)}) found in {source}")
    try:
        calendar = _read_csv_from(source, "calendar.csv")
        prices = _read_csv_from(source, "sell_prices.csv")
    except FileNotFoundError as exc:
        raise DataValidationError(f"{exc} not found in {source}") from exc
    return _normalise(RawM5(calendar=calendar, prices=prices, sales=sales))


def _normalise(raw: RawM5) -> RawM5:
    """Some public mirrors of M5 drop the calendar's ``d`` column (d_1, d_2, ... in date order)
    and the sales ``id`` column. Both are fully determined by the other columns, so rebuild them."""
    cal, sales = raw.calendar, raw.sales
    if "d" not in cal.columns and "date" in cal.columns:
        cal = cal.sort_values("date").reset_index(drop=True)
        cal["d"] = [f"d_{i + 1}" for i in range(len(cal))]
    if "id" not in sales.columns and {"item_id", "store_id"} <= set(sales.columns):
        sales = sales.copy()
        sales.insert(0, "id", sales["item_id"] + "_" + sales["store_id"] + "_evaluation")
    return RawM5(calendar=cal, prices=raw.prices, sales=sales)


# ----------------------------------------------------------------------------- validation


def _check_columns(df: pd.DataFrame, cols: list[str], table: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"{table} is missing columns: {missing}")


def validate_raw(raw: RawM5) -> int:
    """Check the three tables and return the number of sales days."""
    _check_columns(raw.sales, ["item_id", "dept_id", "store_id", "state_id"], "sales")
    _check_columns(raw.prices, ["store_id", "item_id", "wm_yr_wk", "sell_price"], "sell_prices")
    _check_columns(raw.calendar, ["date", "wm_yr_wk", "d"], "calendar")

    d_cols = [c for c in raw.sales.columns if c.startswith("d_")]
    if not d_cols:
        raise DataValidationError("sales has no d_<n> columns")
    nums = [int(c[2:]) for c in d_cols]
    if nums != list(range(1, len(nums) + 1)):
        raise DataValidationError("sales day columns are not d_1 ... d_N in order")
    if len(raw.calendar) < len(d_cols):
        raise DataValidationError("calendar has fewer days than the sales table")

    values = raw.sales[d_cols].to_numpy()
    if np.isnan(values.astype(float)).any():
        raise DataValidationError("sales contain missing values")
    if (values < 0).any():
        raise DataValidationError("sales contain negative values")
    if (raw.prices["sell_price"] <= 0).any() or raw.prices["sell_price"].isna().any():
        raise DataValidationError("sell_price must be positive and non-missing")
    if raw.sales.duplicated(["store_id", "item_id"]).any():
        raise DataValidationError("duplicate store x item rows in sales")
    return len(d_cols)


# ----------------------------------------------------------------------------- panel


def _calendar_arrays(calendar: pd.DataFrame, n_days: int) -> dict[str, np.ndarray | pd.DatetimeIndex]:
    """Calendar features for the sales days plus ``CALENDAR_EXTRA_DAYS`` future days."""
    cal = calendar.reset_index(drop=True).copy()
    cal["date"] = pd.to_datetime(cal["date"])
    total = n_days + CALENDAR_EXTRA_DAYS
    if len(cal) >= total:
        cal = cal.iloc[:total]
    else:  # extend with plain calendar days and no events / SNAP
        extra = pd.date_range(cal["date"].iloc[-1] + pd.Timedelta(days=1), periods=total - len(cal))
        pad = pd.DataFrame({"date": extra})
        cal = pd.concat([cal, pad], ignore_index=True)
        cal["wm_yr_wk"] = cal["wm_yr_wk"].ffill()
    dates = pd.DatetimeIndex(cal["date"])
    event = np.zeros(total, dtype=np.int8)
    for col in ("event_name_1", "event_name_2"):
        if col in cal.columns:
            event += cal[col].notna().to_numpy().astype(np.int8)
    snap_cols = {c.split("_", 1)[1]: c for c in cal.columns if c.startswith("snap_")}
    snap_by_state = {
        state: cal[col].fillna(0).to_numpy().astype(np.int8) for state, col in snap_cols.items()
    }
    return {
        "dates": dates,
        "dow": dates.dayofweek.to_numpy().astype(np.int8),
        "month": dates.month.to_numpy().astype(np.int8),
        "event": event,
        "snap_by_state": snap_by_state,
        "wm_yr_wk": cal["wm_yr_wk"].to_numpy(),
    }


def _price_matrix(
    prices: pd.DataFrame, ids: pd.DataFrame, week_of_day: np.ndarray
) -> np.ndarray:
    """(n_series, n_days) price with NaN on days an item was not listed in that store."""
    weeks = np.unique(week_of_day)
    week_pos = {w: i for i, w in enumerate(weeks)}
    key = (ids["store_id"] + "|" + ids["item_id"]).to_numpy()
    key_pos = {k: i for i, k in enumerate(key)}
    pkey = (prices["store_id"] + "|" + prices["item_id"]).map(key_pos)
    pweek = prices["wm_yr_wk"].map(week_pos)
    ok = pkey.notna() & pweek.notna()
    weekly = np.full((len(ids), len(weeks)), np.nan, dtype=np.float32)
    weekly[pkey[ok].astype(int).to_numpy(), pweek[ok].astype(int).to_numpy()] = prices.loc[
        ok, "sell_price"
    ].to_numpy(dtype=np.float32)
    day_week = np.array([week_pos[w] for w in week_of_day])
    return weekly[:, day_week]


def _fill_price(price: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill gaps along time; series with no price at all stay NaN."""
    filled = pd.DataFrame(price.T).ffill().bfill().to_numpy().T
    return filled.astype(np.float32)


def build_panel(raw: RawM5, cfg: ExperimentConfig, dataset: str | None = None) -> M5Data:
    """Filter to FOODS, align prices and calendar, and keep the series eligible for the study."""
    n_days = validate_raw(raw)
    sales = raw.sales
    if "cat_id" in sales.columns:
        sales = sales[sales["cat_id"] == "FOODS"]
    sales = sales.reset_index(drop=True)
    if sales.empty:
        raise DataValidationError("no FOODS series found")

    d_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    ids = sales[["item_id", "dept_id", "store_id", "state_id"]].copy()
    ids.insert(0, "series_id", ids["store_id"] + "|" + ids["item_id"])
    matrix = sales[d_cols].to_numpy().astype(np.float32)

    if dataset is None:
        dataset = DATASET_SIMULATED if ids["item_id"].str.contains("SIM").any() else DATASET_REAL

    cal = _calendar_arrays(raw.calendar, n_days)
    week_of_day = np.asarray(cal["wm_yr_wk"])
    price_raw = _price_matrix(raw.prices, ids, week_of_day)  # includes the extra future days

    # eligibility: listed on every day from the start of the history window to the end, and
    # selling at least ``min_mean_demand`` units a day in the history before validation.
    hist_start = n_days - cfg.val_days - cfg.test_days - cfg.min_history
    if hist_start < 0:
        raise DataValidationError(
            f"{n_days} days are not enough for min_history={cfg.min_history}, "
            f"val_days={cfg.val_days}, test_days={cfg.test_days}"
        )
    val_start = n_days - cfg.val_days - cfg.test_days
    listed = ~np.isnan(price_raw[:, hist_start:n_days]).any(axis=1)
    mean_demand = matrix[:, hist_start:val_start].mean(axis=1)
    sells = mean_demand >= cfg.min_mean_demand
    keep = listed & sells
    if keep.sum() < 10:
        raise DataValidationError(f"only {int(keep.sum())} series are eligible for the study")

    price = _fill_price(price_raw[keep])
    state = ids.loc[keep, "state_id"].to_numpy()
    snap = np.zeros((int(keep.sum()), n_days + CALENDAR_EXTRA_DAYS), dtype=np.int8)
    for s in np.unique(state):
        arr = cal["snap_by_state"].get(s)
        if arr is not None:
            snap[state == s] = arr

    report = {
        "n_foods_series": int(len(ids)),
        "n_listed_whole_history": int(listed.sum()),
        "n_dropped_not_listed": int((~listed).sum()),
        "n_dropped_low_demand": int((listed & ~sells).sum()),
        "n_kept": int(keep.sum()),
        "history_start_day": int(hist_start),
        "n_days": int(n_days),
    }
    return M5Data(
        dataset=dataset,
        ids=ids[keep].reset_index(drop=True),
        sales=matrix[keep],
        price=price,
        dates=cal["dates"],
        dow=cal["dow"],
        month=cal["month"],
        event=cal["event"],
        snap=snap,
        n_days=n_days,
        report=report,
    )


def load_panel(source: str | Path, cfg: ExperimentConfig, dataset: str | None = None) -> M5Data:
    return build_panel(read_m5(source), cfg, dataset)
