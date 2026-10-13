"""Day-by-day inventory simulation, vectorised over series.

Event order on simulation day ``d`` (all series at once):

1. orders placed ``P`` days ago arrive (``P = lead time + review period``) and enter stock as new;
2. the day's demand is served first-in-first-out, oldest units first; unmet demand is a lost sale;
3. units that reach the end of their shelf life are discarded as waste;
4. stock held overnight is charged holding cost;
5. an order is placed: ``max(0, S_d - (on hand + on order))`` where ``S_d`` is the policy's
   order-up-to level computed from information up to the end of day ``d``.

Series are independent, so every output is a per-series total that can be regrouped freely
(by ABC class, by demand pattern) and bootstrapped over series.

Cost parameters are not used here. The simulation records physical quantities and prices, and
``cost_components`` turns them into money, so cost assumptions can be changed without re-running
the simulation (except the shelf life, which changes the physical flows).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CostConfig

PHYSICAL_FIELDS = (
    "demand",
    "sold",
    "lost",
    "waste",
    "onhand_days",
    "inv_value_days",
    "waste_value",
    "lost_pmed",
    "lost_revenue",
    "revenue",
    "stockout_days",
    "ordered",
)


@dataclass
class SimTotals:
    """Per-series totals over the evaluated days (after warm-up)."""

    fields: dict[str, np.ndarray]
    n_days: int

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]

    @property
    def n_series(self) -> int:
        return len(next(iter(self.fields.values())))

    def subset(self, mask: np.ndarray) -> SimTotals:
        return SimTotals({k: v[mask] for k, v in self.fields.items()}, self.n_days)


def combine(by_level: dict[float, SimTotals], level_of_series: np.ndarray) -> SimTotals:
    """Per series, take the totals of the service level chosen for that series."""
    levels = sorted(by_level)
    first = by_level[levels[0]]
    out = {k: np.empty_like(v) for k, v in first.fields.items()}
    for q in levels:
        mask = level_of_series == q
        for k in out:
            out[k][mask] = by_level[q].fields[k][mask]
    return SimTotals(out, first.n_days)


def simulate(
    order_up_to: np.ndarray,
    demand: np.ndarray,
    price: np.ndarray,
    price_ref: np.ndarray,
    init_stock: np.ndarray,
    lead_time: int,
    shelf_life: int,
    warmup: int,
) -> SimTotals:
    """Run the replenishment simulation.

    ``order_up_to``, ``demand`` and ``price`` are (n_series, n_days); ``price_ref`` and
    ``init_stock`` are (n_series,). The review period is one day.
    """
    n, n_days = demand.shape
    if order_up_to.shape != demand.shape:
        raise ValueError("order_up_to and demand must have the same shape")
    if shelf_life < 1 or lead_time < 0:
        raise ValueError("shelf_life must be >= 1 and lead_time >= 0")
    if warmup >= n_days:
        raise ValueError("warm-up is longer than the simulation")
    p = lead_time + 1

    inv = np.zeros((n, shelf_life))  # inv[:, a] = units that are a days old at the start of a day
    inv[:, 0] = init_stock
    pipe = np.zeros((n, p))
    acc = {k: np.zeros(n) for k in PHYSICAL_FIELDS}

    for d in range(n_days):
        slot = d % p
        inv[:, 0] += pipe[:, slot]
        pipe[:, slot] = 0.0

        dem = demand[:, d].astype(np.float64)
        rem = dem.copy()
        for a in range(shelf_life - 1, -1, -1):
            take = np.minimum(inv[:, a], rem)
            inv[:, a] -= take
            rem -= take
        lost = rem
        sold = dem - lost

        waste = inv[:, -1].copy()
        if shelf_life > 1:
            inv[:, 1:] = inv[:, :-1].copy()
        inv[:, 0] = 0.0
        onhand = inv.sum(axis=1)

        position = onhand + pipe.sum(axis=1)
        order = np.maximum(order_up_to[:, d].astype(np.float64) - position, 0.0)
        pipe[:, slot] = order

        if d >= warmup:
            pr = price[:, d].astype(np.float64)
            acc["demand"] += dem
            acc["sold"] += sold
            acc["lost"] += lost
            acc["waste"] += waste
            acc["onhand_days"] += onhand
            acc["inv_value_days"] += onhand * price_ref
            acc["waste_value"] += waste * price_ref
            acc["lost_pmed"] += lost * price_ref
            acc["lost_revenue"] += lost * pr
            acc["revenue"] += sold * pr
            acc["stockout_days"] += (lost > 0).astype(np.float64)
            acc["ordered"] += order
    return SimTotals(acc, n_days - warmup)


# ----------------------------------------------------------------------------- money


def cost_components(phys: dict[str, np.ndarray | float], cost: CostConfig) -> dict:
    """Turn physical totals into costs. Works on per-series arrays or aggregated scalars.

    unit cost = (1 - margin) * reference price, so
      holding = holding_rate / 365 * unit cost * unit-days in stock
      waste   = unit cost * units discarded
      lost    = stockout_mult * (selling price - unit cost) * units not sold
    """
    c = 1.0 - cost.margin
    holding = cost.holding_rate / 365.0 * c * phys["inv_value_days"]
    waste = c * phys["waste_value"]
    lost = cost.stockout_mult * (phys["lost_revenue"] - c * phys["lost_pmed"])
    return {"holding": holding, "waste": waste, "lost": lost, "total": holding + waste + lost}


def aggregate(totals: SimTotals, mask: np.ndarray | None = None) -> dict[str, float]:
    """Sum the per-series totals (optionally over a subset of series)."""
    out: dict[str, float] = {}
    for k, v in totals.fields.items():
        out[k] = float((v if mask is None else v[mask]).sum())
    n = totals.n_series if mask is None else int(mask.sum())
    out["series_days"] = float(n * totals.n_days)
    out["n_series"] = float(n)
    return out


def metrics(phys: dict[str, float], cost: CostConfig) -> dict[str, float]:
    """Managerial KPIs from aggregated physical totals."""
    comp = cost_components(phys, cost)
    demand = max(phys["demand"], 1e-9)
    return {
        "fill_rate": 1.0 - phys["lost"] / demand,
        "stockout_day_rate": phys["stockout_days"] / max(phys["series_days"], 1e-9),
        "waste_rate": phys["waste"] / demand,
        "avg_inventory": phys["onhand_days"] / max(phys["series_days"], 1e-9),
        "days_of_cover": phys["onhand_days"] / demand,
        "holding_cost": comp["holding"],
        "waste_cost": comp["waste"],
        "lost_margin_cost": comp["lost"],
        "total_cost": comp["total"],
        "cost_pct_revenue": comp["total"] / max(phys["revenue"], 1e-9),
        "revenue": phys["revenue"],
    }
