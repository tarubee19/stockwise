"""Causal features for forecasting the demand of the next ``P`` days.

An *origin* ``t`` is the end of day ``t``, when an order decision is made. Every feature for
origin ``t`` uses only information that is known at that moment:

* sales up to and including day ``t`` (lags, rolling means, intermittency),
* the selling price and calendar (events, SNAP days, weekday) of the coming ``P`` days, which a
  store knows in advance (prices are set weekly, the calendar is fixed).

The target is the total demand of days ``t+1 ... t+P``. ``P = lead time + review period`` is the
protection interval an order-up-to level must cover. Nothing after day ``t`` ever enters the
sales features; ``tests/test_features.py`` proves it by changing the future and checking that
the features do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_PROTECTION
from .data import M5Data

MIN_ORIGIN = 364  # the 365-day window needs this much history

FEATURE_NAMES: list[str] = (
    [f"y_lag{k}" for k in range(7)]
    + ["mean7", "mean14", "mean28", "mean56", "mean112", "mean365"]
    + ["std28", "cv28", "zero_share28", "days_since_sale"]
    + ["sn1", "sn2", "sn3", "sn4", "sn_mean", "sba"]
    + ["pr1", "prP", "pr_rel1", "pr_relP", "log_price"]
    + ["dow_next", "month_next", "snap_next", "event_next", "weekend_next"]
    + ["store", "dept", "state"]
)
CATEGORICAL = ["dow_next", "month_next", "store", "dept", "state"]


def croston_sba(sales: np.ndarray, alpha: float = 0.1, init_days: int = 28) -> np.ndarray:
    """Croston method with the Syntetos-Boylan bias correction, vectorised over series.

    ``out[:, t]`` is the forecast of the per-day demand rate made after observing day ``t``.
    """
    n, n_days = sales.shape
    head = sales[:, :init_days]
    n_pos = (head > 0).sum(axis=1)
    z = np.where(n_pos > 0, head.sum(axis=1) / np.maximum(n_pos, 1), 1.0).astype(np.float64)
    p = np.where(n_pos > 0, init_days / np.maximum(n_pos, 1), float(init_days)).astype(np.float64)
    since = np.zeros(n)
    out = np.empty((n, n_days), dtype=np.float32)
    for t in range(n_days):
        d = sales[:, t].astype(np.float64)
        if t >= init_days:
            since += 1
            hit = d > 0
            z[hit] += alpha * (d[hit] - z[hit])
            p[hit] += alpha * (since[hit] - p[hit])
            since[hit] = 0
        out[:, t] = (1 - alpha / 2) * z / p
    return out


class FeatureBuilder:
    """Precomputes cumulative sums once, then builds feature matrices for any set of origins."""

    def __init__(self, data: M5Data, protection: int):
        if not 1 <= protection <= MAX_PROTECTION:
            raise ValueError(f"protection must be between 1 and {MAX_PROTECTION}")
        self.data = data
        self.P = protection
        n, n_days = data.sales.shape
        self.n, self.N = n, n_days
        sales = data.sales

        zero_col = np.zeros((n, 1))
        self.cs = np.concatenate([zero_col, np.cumsum(sales, axis=1, dtype=np.float64)], axis=1)
        self.cs2 = np.concatenate(
            [zero_col, np.cumsum(np.square(sales, dtype=np.float64), axis=1)], axis=1
        )
        self.cz = np.concatenate(
            [zero_col.astype(np.int32), np.cumsum(sales == 0, axis=1, dtype=np.int32)], axis=1
        )
        idx = np.where(sales > 0, np.arange(n_days, dtype=np.int32)[None, :], np.int32(-1))
        self.last_sale = np.maximum.accumulate(idx, axis=1)
        self.sba = croston_sba(sales)

        total_cols = data.price.shape[1]
        self.price = data.price.astype(np.float32)
        self.pc = np.concatenate(
            [np.zeros((n, 1)), np.cumsum(self.price, axis=1, dtype=np.float64)], axis=1
        )
        self.cs_snap = np.concatenate(
            [np.zeros((n, 1), dtype=np.int32), np.cumsum(data.snap, axis=1, dtype=np.int32)], axis=1
        )
        self.cs_event = np.concatenate([[0], np.cumsum(data.event, dtype=np.int32)])
        weekend = (data.dow >= 5).astype(np.int32)
        self.cs_weekend = np.concatenate([[0], np.cumsum(weekend, dtype=np.int32)])
        self.total_cols = total_cols

        ids = data.ids
        self.store = pd.Categorical(ids["store_id"]).codes.astype(np.float32)
        self.dept = pd.Categorical(ids["dept_id"]).codes.astype(np.float32)
        self.state = pd.Categorical(ids["state_id"]).codes.astype(np.float32)

    # ---------------------------------------------------------------- helpers
    def _check_origins(self, origins: np.ndarray, with_target: bool) -> np.ndarray:
        o = np.asarray(origins, dtype=np.int64)
        if o.size == 0:
            raise ValueError("no origins given")
        if o.min() < MIN_ORIGIN:
            raise ValueError(f"origins must be >= {MIN_ORIGIN} (got {int(o.min())})")
        last_allowed = self.N - 1 - (self.P if with_target else 0)
        if o.max() > last_allowed:
            raise ValueError(f"origin {int(o.max())} leaves no room for the target (max {last_allowed})")
        if self.N + self.P > self.total_cols:
            raise ValueError("price/calendar arrays are too short for the protection interval")
        return o

    def rolling_mean(self, o: np.ndarray, w: int) -> np.ndarray:
        return (self.cs[:, o + 1] - self.cs[:, o + 1 - w]) / w

    def rolling_std(self, o: np.ndarray, w: int) -> np.ndarray:
        mean = self.rolling_mean(o, w)
        mean_sq = (self.cs2[:, o + 1] - self.cs2[:, o + 1 - w]) / w
        return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))

    def target(self, origins: np.ndarray) -> np.ndarray:
        """(n_series, n_origins) demand of days t+1 ... t+P."""
        o = self._check_origins(origins, with_target=True)
        return (self.cs[:, o + self.P + 1] - self.cs[:, o + 1]).astype(np.float32)

    # ---------------------------------------------------------------- features
    def build(self, origins: np.ndarray, with_target: bool = True) -> np.ndarray:
        """Feature array of shape (n_origins, n_series, n_features), float32."""
        o = self._check_origins(origins, with_target)
        n, P = self.n, self.P
        cols: dict[str, np.ndarray] = {}
        sales = self.data.sales

        for k in range(7):
            cols[f"y_lag{k}"] = sales[:, o - k]
        for w in (7, 14, 28, 56, 112, 365):
            cols[f"mean{w}"] = self.rolling_mean(o, w)
        mean28 = cols["mean28"]
        std28 = self.rolling_std(o, 28)
        cols["std28"] = std28
        cols["cv28"] = std28 / np.maximum(mean28, 1e-3)
        cols["zero_share28"] = (self.cz[:, o + 1] - self.cz[:, o + 1 - 28]) / 28
        cols["days_since_sale"] = np.minimum(o[None, :] - self.last_sale[:, o], 60)

        sn = []
        for j in range(1, 5):  # same weekdays j weeks earlier
            s = self.cs[:, o + P - 7 * j + 1] - self.cs[:, o + 1 - 7 * j]
            cols[f"sn{j}"] = s
            sn.append(s)
        cols["sn_mean"] = np.mean(sn, axis=0)
        cols["sba"] = self.sba[:, o] * P

        price = self.price
        base = (self.pc[:, o + 1] - self.pc[:, o + 1 - 56]) / 56
        pr1, prP = price[:, o + 1], price[:, o + P]
        cols["pr1"], cols["prP"] = pr1, prP
        cols["pr_rel1"] = pr1 / base
        cols["pr_relP"] = prP / base
        cols["log_price"] = np.log(price[:, o])

        cols["dow_next"] = np.broadcast_to(self.data.dow[o + 1].astype(np.float32), (n, o.size))
        cols["month_next"] = np.broadcast_to(self.data.month[o + 1].astype(np.float32), (n, o.size))
        cols["snap_next"] = self.cs_snap[:, o + P + 1] - self.cs_snap[:, o + 1]
        ev = self.cs_event[o + P + 1] - self.cs_event[o + 1]
        cols["event_next"] = np.broadcast_to(ev.astype(np.float32), (n, o.size))
        wk = self.cs_weekend[o + P + 1] - self.cs_weekend[o + 1]
        cols["weekend_next"] = np.broadcast_to(wk.astype(np.float32), (n, o.size))
        for name, vec in (("store", self.store), ("dept", self.dept), ("state", self.state)):
            cols[name] = np.broadcast_to(vec[:, None], (n, o.size))

        out = np.empty((o.size, n, len(FEATURE_NAMES)), dtype=np.float32)
        for j, name in enumerate(FEATURE_NAMES):
            out[:, :, j] = np.asarray(cols[name]).T
        return out


def flatten(x: np.ndarray) -> np.ndarray:
    """(n_origins, n_series, F) -> (n_origins * n_series, F), origin-major."""
    return x.reshape(-1, x.shape[-1])


def column(x: np.ndarray, name: str) -> np.ndarray:
    """One feature as an (n_series, n_origins) array."""
    return x[:, :, FEATURE_NAMES.index(name)].T
