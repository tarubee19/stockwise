"""Order-up-to levels for each replenishment policy.

An order-up-to level ``S`` is the stock position (on hand + on order) the store tries to reach
at the end of each day so that the next ``P`` days of demand are covered. Each policy turns a
service level ``q`` (the probability that demand over the protection interval is covered) into
an ``S``:

* ``ma_z``: 28-day moving average plus ``z * sigma`` safety stock, the textbook rule;
* ``lgbm_nb``: LightGBM mean forecast with a negative binomial safety stock;
* ``lgbm_qr``: the LightGBM quantile forecast itself;
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .forecasting import PassOutput

ALPHA_MIN, ALPHA_MAX = 0.01, 5.0


def fit_dispersion(
    mu: np.ndarray, y: np.ndarray, groups: np.ndarray, default: float = 0.5
) -> dict[int, float]:
    """Negative binomial dispersion alpha (var = mu + alpha * mu^2) per group, by moments.

    ``mu`` and ``y`` are (n_series, n_origins); ``groups`` has one integer per series.
    """
    out: dict[int, float] = {}
    for g in np.unique(groups):
        m = mu[groups == g].astype(np.float64)
        d = y[groups == g].astype(np.float64)
        denom = float(np.sum(m**2))
        if denom <= 0:
            out[int(g)] = default
            continue
        alpha = float(np.sum((d - m) ** 2 - m) / denom)
        out[int(g)] = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    return out


def dispersion_matrix(alpha_by_group: dict[int, float], groups: np.ndarray, n_cols: int) -> np.ndarray:
    a = np.array([alpha_by_group.get(int(g), 0.5) for g in groups], dtype=np.float64)
    return np.repeat(a[:, None], n_cols, axis=1)


def nb_quantile(mu: np.ndarray, alpha: np.ndarray, q: float) -> np.ndarray:
    """Quantile of a negative binomial with mean ``mu`` and dispersion ``alpha``."""
    mu = np.maximum(mu.astype(np.float64), 1e-6)
    n = 1.0 / alpha
    p = n / (n + mu)
    return stats.nbinom.ppf(q, n, p)


def levels(
    out: PassOutput,
    quantiles: tuple[float, ...],
    alpha: np.ndarray,
) -> dict[str, dict[float, np.ndarray]]:
    """Order-up-to levels (n_series, n_origins) for every policy and service level."""
    mu_ma = out.mean["ma28"]
    res: dict[str, dict[float, np.ndarray]] = {"ma_z": {}, "lgbm_nb": {}, "lgbm_qr": {}}
    for j, q in enumerate(quantiles):
        z = float(stats.norm.ppf(q))
        res["ma_z"][q] = np.ceil(np.maximum(mu_ma + z * out.ma_sigma, 0.0)).astype(np.float32)
        res["lgbm_nb"][q] = nb_quantile(out.mean["lgbm"], alpha, q).astype(np.float32)
        res["lgbm_qr"][q] = np.ceil(out.qraw[:, :, j]).astype(np.float32)
    return res


# ----------------------------------------------------------------------------- segments


def abc_classes(revenue: np.ndarray) -> np.ndarray:
    """ABC classes by cumulative revenue share: A = top 80%, B = next 15%, C = last 5%."""
    order = np.argsort(-revenue, kind="stable")
    share = np.cumsum(revenue[order]) / max(float(revenue.sum()), 1e-9)
    cls = np.empty(len(revenue), dtype="<U1")
    labels = np.where(share <= 0.80, "A", np.where(share <= 0.95, "B", "C"))
    cls[order] = labels
    return cls


def demand_pattern(history: np.ndarray) -> np.ndarray:
    """Syntetos-Boylan classes from the average demand interval (ADI) and CV^2 of sizes."""
    n_days = history.shape[1]
    nonzero = history > 0
    n_nz = nonzero.sum(axis=1)
    adi = np.where(n_nz > 0, n_days / np.maximum(n_nz, 1), np.inf)
    pos = np.where(nonzero, history, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(pos, axis=1)
        cv2 = (np.nanstd(pos, axis=1) / mean) ** 2
    cv2 = np.where(np.isfinite(cv2), cv2, 0.0)
    smooth_adi = adi < 1.32
    low_cv = cv2 < 0.49
    cls = np.where(
        smooth_adi,
        np.where(low_cv, "smooth", "erratic"),
        np.where(low_cv, "intermittent", "lumpy"),
    )
    return cls.astype("<U12")


def segment_frame(ids: pd.DataFrame, abc: np.ndarray, pattern: np.ndarray) -> pd.DataFrame:
    seg = ids[["series_id", "item_id", "dept_id", "store_id", "state_id"]].copy()
    seg["abc"] = abc
    seg["pattern"] = pattern
    return seg
