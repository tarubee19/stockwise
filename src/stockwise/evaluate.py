"""Forecast accuracy, policy curves, service-level selection, frontier comparison, bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig
from .simulate import SimTotals, aggregate, cost_components, metrics

# ----------------------------------------------------------------------------- accuracy


def naive_scale(cs: np.ndarray, protection: int, start: int, stop: int) -> np.ndarray:
    """Per-series MSE of the same-weekdays-last-week forecast of P-day demand on days
    [start, stop). It scales squared errors so series of different size can be averaged."""
    u = np.arange(start, stop)
    r_now = cs[:, u + 1] - cs[:, u + 1 - protection]
    r_prev = cs[:, u + 1 - 7] - cs[:, u + 1 - 7 - protection]
    return np.mean((r_now - r_prev) ** 2, axis=1)


def forecast_accuracy(
    y: np.ndarray,
    forecasts: dict[str, np.ndarray],
    scale: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """WAPE, bias and scaled RMSE of mean forecasts of the P-day demand.

    * WAPE = sum |error| / sum demand (lower is better);
    * bias = sum error / sum demand (positive = over-forecasting);
    * RMSSE = sqrt(MSE / MSE of the same-weekdays-last-week forecast in the history), averaged
      over series, plain and weighted by revenue (below 1 beats that naive forecast).
    """
    rows = []
    total = float(y.sum())
    ok = scale > 0
    for name, f in forecasts.items():
        err = f.astype(np.float64) - y
        mse_i = np.mean(err**2, axis=1)
        rmsse_i = np.sqrt(mse_i[ok] / scale[ok])
        w = weights[ok]
        rows.append(
            {
                "method": name,
                "wape": float(np.abs(err).sum() / total),
                "bias": float(err.sum() / total),
                "rmsse": float(rmsse_i.mean()),
                "wrmsse": float((w * rmsse_i).sum() / w.sum()),
                "n_forecasts": int(y.size),
            }
        )
    return pd.DataFrame(rows)


def pinball(y: np.ndarray, f: np.ndarray, q: float) -> np.ndarray:
    d = y - f
    return np.maximum(q * d, (q - 1) * d)


def quantile_accuracy(
    y: np.ndarray, levels: dict[str, dict[float, np.ndarray]]
) -> pd.DataFrame:
    """Pinball loss (as a share of total demand) and coverage of each policy's quantile."""
    rows = []
    total = float(y.sum())
    for method, by_q in levels.items():
        for q, s in sorted(by_q.items()):
            rows.append(
                {
                    "method": method,
                    "q": q,
                    "pinball_pct_demand": float(pinball(y, s, q).sum() / total),
                    "coverage": float(np.mean(y <= s)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- curves


def curve_frame(
    sims: dict[tuple[str, float], SimTotals], cost: CostConfig, period: str
) -> pd.DataFrame:
    rows = []
    for (policy, q), totals in sims.items():
        rows.append({"period": period, "policy": policy, "q": q, **metrics(aggregate(totals), cost)})
    return pd.DataFrame(rows)


def per_series_cost(totals: SimTotals, cost: CostConfig) -> np.ndarray:
    return cost_components(totals.fields, cost)["total"]


def choose_levels(curve: pd.DataFrame) -> dict[str, float]:
    """Service level with the lowest total cost, per policy (policies with one level skipped)."""
    out: dict[str, float] = {}
    for policy, g in curve.groupby("policy"):
        if not g.empty:
            out[policy] = float(g.loc[g["total_cost"].idxmin(), "q"])
    return out


def choose_levels_by_class(
    val_sims: dict[float, SimTotals], classes: np.ndarray, cost: CostConfig
) -> dict[str, float]:
    """Best service level for each class of series, by total validation cost in that class."""
    costs = {q: per_series_cost(t, cost) for q, t in val_sims.items()}
    out: dict[str, float] = {}
    for c in np.unique(classes):
        mask = classes == c
        out[str(c)] = min(costs, key=lambda q: float(costs[q][mask].sum()))
    return out


# ----------------------------------------------------------------------------- frontier


def value_at_fill(curve: pd.DataFrame, target: float, column: str) -> float:
    """Linear interpolation of ``column`` along a policy's fill-rate curve; NaN if out of range."""
    c = curve.sort_values("fill_rate")
    x, v = c["fill_rate"].to_numpy(), c[column].to_numpy()
    if len(x) < 2 or target < x.min() or target > x.max():
        return float("nan")
    return float(np.interp(target, x, v))


def frontier_table(
    curve: pd.DataFrame, baseline: str, targets: tuple[float, ...] = (0.90, 0.95, 0.98)
) -> pd.DataFrame:
    """Inventory and cost each policy needs to reach the same fill rate, versus the baseline."""
    rows = []
    for t in targets:
        base_inv = value_at_fill(curve[curve.policy == baseline], t, "avg_inventory")
        base_cost = value_at_fill(curve[curve.policy == baseline], t, "total_cost")
        for policy, g in curve.groupby("policy"):
            inv = value_at_fill(g, t, "avg_inventory")
            tc = value_at_fill(g, t, "total_cost")
            rows.append(
                {
                    "fill_target": t,
                    "policy": policy,
                    "avg_inventory": inv,
                    "total_cost": tc,
                    "inventory_vs_baseline_pct": (inv / base_inv - 1) * 100 if base_inv else np.nan,
                    "cost_vs_baseline_pct": (tc / base_cost - 1) * 100 if base_cost else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- bootstrap


def bootstrap_vs_baseline(
    chosen: dict[str, SimTotals],
    baseline: str,
    cost: CostConfig,
    draws: int,
    seed: int,
    chunk: int = 100,
) -> pd.DataFrame:
    """95% intervals, resampling series, for each policy against the baseline policy.

    Metrics: cost reduction (%), fill-rate difference (percentage points) and inventory change (%).
    """
    rng = np.random.default_rng(seed)
    base = chosen[baseline]
    n = base.n_series
    cost_arr = {p: per_series_cost(t, cost) for p, t in chosen.items()}
    results: dict[str, list[np.ndarray]] = {p: [] for p in chosen if p != baseline}
    for start in range(0, draws, chunk):
        k = min(chunk, draws - start)
        idx = rng.integers(0, n, size=(k, n))
        b_cost = cost_arr[baseline][idx].sum(axis=1)
        b_lost = base["lost"][idx].sum(axis=1)
        b_inv = base["onhand_days"][idx].sum(axis=1)
        dem = base["demand"][idx].sum(axis=1)
        for p in results:
            t = chosen[p]
            cr = (1 - cost_arr[p][idx].sum(axis=1) / b_cost) * 100
            fd = (b_lost - t["lost"][idx].sum(axis=1)) / dem * 100
            ic = (t["onhand_days"][idx].sum(axis=1) / b_inv - 1) * 100
            results[p].append(np.column_stack([cr, fd, ic]))

    rows = []
    b_tot = float(cost_arr[baseline].sum())
    for p, parts in results.items():
        arr = np.vstack(parts)
        t = chosen[p]
        point = (
            (1 - float(cost_arr[p].sum()) / b_tot) * 100,
            (float(base["lost"].sum()) - float(t["lost"].sum())) / float(base["demand"].sum()) * 100,
            (float(t["onhand_days"].sum()) / float(base["onhand_days"].sum()) - 1) * 100,
        )
        for j, name in enumerate(("cost_reduction_pct", "fill_rate_diff_pp", "inventory_change_pct")):
            rows.append(
                {
                    "policy": p,
                    "metric": name,
                    "estimate": point[j],
                    "ci_low": float(np.percentile(arr[:, j], 2.5)),
                    "ci_high": float(np.percentile(arr[:, j], 97.5)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- segments


def segment_table(
    chosen: dict[str, SimTotals],
    baseline: str,
    seg: pd.DataFrame,
    cost: CostConfig,
) -> pd.DataFrame:
    """Per ABC class x demand pattern: size, fill rate, inventory and cost for each policy."""
    rows = []
    keys = seg[["abc", "pattern"]].drop_duplicates().sort_values(["abc", "pattern"])
    for _, k in keys.iterrows():
        mask = ((seg["abc"] == k["abc"]) & (seg["pattern"] == k["pattern"])).to_numpy()
        base_cost = float(per_series_cost(chosen[baseline], cost)[mask].sum())
        for policy, t in chosen.items():
            m = metrics(aggregate(t, mask), cost)
            rows.append(
                {
                    "abc": k["abc"],
                    "pattern": k["pattern"],
                    "policy": policy,
                    "n_series": int(mask.sum()),
                    "fill_rate": m["fill_rate"],
                    "avg_inventory": m["avg_inventory"],
                    "total_cost": m["total_cost"],
                    "cost_vs_baseline_pct": (m["total_cost"] / base_cost - 1) * 100
                    if base_cost
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
