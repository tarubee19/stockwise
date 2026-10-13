"""Re-price the simulated physical flows under different cost assumptions.

The simulator records physical quantities (units lost, wasted, held; revenue). Money is applied
afterwards, so a manager can ask "what if holding costs are twice as high?" without re-running
the simulation. The same function produces the sensitivity table in ``results/`` and the sliders
in the dashboard, so the two always agree.

Input is the ``physical_totals`` table: one row per (period, policy, service level q, ABC class)
with the summed physical fields of ``simulate.PHYSICAL_FIELDS`` plus ``series_days``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_POLICY, CostConfig
from .simulate import PHYSICAL_FIELDS, metrics

SUM_COLUMNS = (*PHYSICAL_FIELDS, "series_days", "n_series")
POLICIES_TUNED = ("ma_z", "lgbm_nb", "lgbm_qr")
ABC_POLICY = "lgbm_qr_abc"


def _sum_rows(df: pd.DataFrame) -> dict[str, float]:
    return {c: float(df[c].sum()) for c in SUM_COLUMNS}


def _cost(df: pd.DataFrame, cost: CostConfig) -> float:
    return metrics(_sum_rows(df), cost)["total_cost"]


def evaluate_scenario(
    phys: pd.DataFrame, cost: CostConfig, baseline: str = BASELINE_POLICY
) -> pd.DataFrame:
    """Choose service levels on the validation period, report the test period.

    Returns one row per policy with the chosen level (``"per class"`` for the ABC policy) and
    the test KPIs; ``cost_vs_baseline_pct`` is relative to the baseline policy.
    """
    val = phys[phys["period"] == "val"]
    test = phys[phys["period"] == "test"]
    rows: list[dict] = []

    for policy in POLICIES_TUNED:
        v = val[val["policy"] == policy]
        if v.empty:
            continue
        by_q = {q: _cost(g, cost) for q, g in v.groupby("q")}
        q_star = min(by_q, key=by_q.get)
        t = test[(test["policy"] == policy) & (np.isclose(test["q"], q_star))]
        rows.append({"policy": policy, "chosen_q": f"{q_star:.2f}", **metrics(_sum_rows(t), cost)})

    v = val[val["policy"] == "lgbm_qr"]
    if not v.empty:
        parts, labels = [], []
        for cls in sorted(v["abc"].unique()):
            vc = v[v["abc"] == cls]
            by_q = {q: _cost(g, cost) for q, g in vc.groupby("q")}
            q_star = min(by_q, key=by_q.get)
            labels.append(f"{cls}:{q_star:.2f}")
            parts.append(
                test[
                    (test["policy"] == "lgbm_qr")
                    & (test["abc"] == cls)
                    & np.isclose(test["q"], q_star)
                ]
            )
        rows.append(
            {
                "policy": ABC_POLICY,
                "chosen_q": " ".join(labels),
                **metrics(_sum_rows(pd.concat(parts)), cost),
            }
        )

    out = pd.DataFrame(rows)
    base_cost = float(out.loc[out["policy"] == baseline, "total_cost"].iloc[0])
    out["cost_vs_baseline_pct"] = (out["total_cost"] / base_cost - 1) * 100
    return out
