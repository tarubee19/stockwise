"""End-to-end study: forecast demand, simulate replenishment policies, evaluate, write results.

Timeline (day indices; ``P`` = lead time + review period):

    ... history ... | validation origins | test origins | last P days
                      (choose service     (report)       (only used as targets)
                       levels here)

* Pass A trains on origins whose targets were known at the first validation origin and predicts
  the validation period. Service levels are chosen on it.
* Pass B trains on origins known at the first test origin (so validation data is included) and
  predicts the test period. All reported numbers come from the test period.
* A final fit uses every day to produce tomorrow's order-up-to levels.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import scipy

from .config import BASELINE_POLICY, POLICY_LABELS, ExperimentConfig
from .data import M5Data
from .evaluate import (
    bootstrap_vs_baseline,
    choose_levels,
    choose_levels_by_class,
    curve_frame,
    forecast_accuracy,
    frontier_table,
    naive_scale,
    quantile_accuracy,
    segment_table,
)
from .features import FeatureBuilder
from .forecasting import PassOutput, fit_pass, predict_origins
from .policies import (
    abc_classes,
    demand_pattern,
    dispersion_matrix,
    fit_dispersion,
    levels,
    segment_frame,
)
from .simulate import SimTotals, aggregate, combine, metrics, simulate
from .whatif import evaluate_scenario

Log = Callable[[str], None]


def timeline(n_days: int, cfg: ExperimentConfig) -> tuple[int, int, int, int]:
    """(val_first, val_last, test_first, test_last) origin days."""
    p = cfg.protection
    test_last = n_days - 1 - p
    test_first = test_last - cfg.test_days + 1
    val_last = test_first - 1
    val_first = val_last - cfg.val_days + 1
    return val_first, val_last, test_first, test_last


@dataclass
class Period:
    """Simulation inputs of one period."""

    name: str
    out: PassOutput
    demand_day: np.ndarray  # (n, D) units demanded on each origin day
    price_day: np.ndarray  # (n, D)
    init_stock: np.ndarray  # (n,)
    dates: pd.DatetimeIndex


def make_period(name: str, data: M5Data, out: PassOutput, protection: int) -> Period:
    o = out.origins
    return Period(
        name=name,
        out=out,
        demand_day=data.sales[:, o],
        price_day=data.price[:, o],
        init_stock=np.ceil(out.cv_inputs["mean28"][:, 0] * protection),
        dates=data.dates[o],
    )


def run_sims(
    period: Period,
    levels_by_policy: dict[str, dict[float, np.ndarray]],
    price_ref: np.ndarray,
    cfg: ExperimentConfig,
    shelf_life: int | None = None,
) -> dict[tuple[str, float], SimTotals]:
    """Simulate every (policy, service level)."""
    w = shelf_life or cfg.cost.shelf_life_days
    sims: dict[tuple[str, float], SimTotals] = {}

    def one(s: np.ndarray) -> SimTotals:
        return simulate(
            s, period.demand_day, period.price_day, price_ref, period.init_stock,
            cfg.lead_time, w, cfg.cost.warmup_days,
        )

    for policy, by_q in levels_by_policy.items():
        for q, s in by_q.items():
            sims[(policy, q)] = one(s)
    return sims


def physical_frame(
    sims_by_period: dict[str, dict[tuple[str, float], SimTotals]], abc: np.ndarray
) -> pd.DataFrame:
    """Physical totals per (period, policy, q, ABC class); the input of ``whatif``."""
    rows = []
    for period, sims in sims_by_period.items():
        for (policy, q), totals in sims.items():
            for cls in ("A", "B", "C"):
                mask = abc == cls
                if mask.any():
                    rows.append(
                        {"period": period, "policy": policy, "q": q, "abc": cls,
                         **aggregate(totals, mask)}
                    )
    return pd.DataFrame(rows)


def _chosen_sims(
    sims: dict[tuple[str, float], SimTotals],
    q_by_policy: dict[str, float],
    q_by_class: dict[str, float],
    abc: np.ndarray,
) -> dict[str, SimTotals]:
    chosen = {p: sims[(p, q)] for p, q in q_by_policy.items()}
    q_series = np.array([q_by_class[c] for c in abc])
    by_q = {q: sims[("lgbm_qr", q)] for q in sorted(set(q_by_class.values()))}
    chosen["lgbm_qr_abc"] = combine(by_q, q_series)
    return chosen


def _selected_table(
    chosen_test: dict[str, SimTotals],
    chosen_val: dict[str, SimTotals],
    q_text: dict[str, str],
    boot: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for policy, t in chosen_test.items():
        row = {"policy": policy, "chosen_q": q_text.get(policy, "-")}
        row.update(metrics(aggregate(t), cfg.cost))
        v = metrics(aggregate(chosen_val[policy]), cfg.cost)
        row["val_total_cost"] = v["total_cost"]
        row["val_fill_rate"] = v["fill_rate"]
        for metric in ("cost_reduction_pct", "fill_rate_diff_pp", "inventory_change_pct"):
            b = boot[(boot.policy == policy) & (boot.metric == metric)]
            if not b.empty:
                row[metric] = float(b["estimate"].iloc[0])
                row[f"{metric}_lo"] = float(b["ci_low"].iloc[0])
                row[f"{metric}_hi"] = float(b["ci_high"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


SENSITIVITY = (
    ("base case", {}),
    ("holding cost 10%/yr", {"holding_rate": 0.10}),
    ("holding cost 40%/yr", {"holding_rate": 0.40}),
    ("margin 15%", {"margin": 0.15}),
    ("margin 40%", {"margin": 0.40}),
    ("stock-out penalty = lost margin only", {"stockout_mult": 1.0}),
    ("stock-out penalty x4", {"stockout_mult": 4.0}),
    ("shelf life 3 days", {"shelf_life_days": 3}),
    ("shelf life 14 days", {"shelf_life_days": 14}),
    ("shelf life 30 days", {"shelf_life_days": 30}),
)


def run_study(
    data: M5Data,
    cfg: ExperimentConfig,
    results_dir: str | Path | None = None,
    log: Log = print,
    figures: bool = True,
) -> dict:
    """Run everything. Returns the result tables and summary; writes them if ``results_dir``."""
    t_start = time.time()

    def say(msg: str) -> None:
        log(f"[{time.time() - t_start:7.1f}s] {msg}")

    p = cfg.protection
    n, n_days = data.sales.shape
    val_first, val_last, test_first, test_last = timeline(n_days, cfg)
    hist_start = data.report["history_start_day"]
    say(
        f"{n:,} series x {n_days:,} days ({data.dataset} data); protection interval P={p}; "
        f"validation origins {data.dates[val_first].date()}..{data.dates[val_last].date()}, "
        f"test origins {data.dates[test_first].date()}..{data.dates[test_last].date()}"
    )

    fb = FeatureBuilder(data, p)
    say("features prepared")

    # ---- segments and reference prices (history only, nothing from validation or test)
    hist = data.sales[:, hist_start:val_first]
    price_hist = data.price[:, hist_start:val_first]
    revenue_hist = (hist * price_hist).sum(axis=1)
    abc = abc_classes(revenue_hist)
    pattern = demand_pattern(hist)
    seg = segment_frame(data.ids, abc, pattern)
    price_ref = np.median(price_hist, axis=1)

    # ---- pass A: validation
    say("pass A: fitting models for the validation period")
    models_a = fit_pass(fb, val_first - p, cfg, lambda m: say(m))
    val_origins = np.arange(val_first, val_last + 1)
    out_a = predict_origins(fb, models_a, val_origins)
    del models_a
    say("pass A: predictions done")

    groups = (fb.dept * 10 + fb.store).astype(int)
    assert out_a.y is not None
    alpha_by_group = fit_dispersion(out_a.mean["lgbm"], out_a.y, groups)

    # ---- pass B: test
    say("pass B: fitting models for the test period")
    models_b = fit_pass(fb, test_first - p, cfg, lambda m: say(m))
    test_origins = np.arange(test_first, test_last + 1)
    out_b = predict_origins(fb, models_b, test_origins)
    del models_b
    say("pass B: predictions done")

    val = make_period("val", data, out_a, p)
    test = make_period("test", data, out_b, p)
    lv_val = levels(out_a, cfg.quantiles, dispersion_matrix(alpha_by_group, groups, len(val_origins)))
    lv_test = levels(out_b, cfg.quantiles, dispersion_matrix(alpha_by_group, groups, len(test_origins)))

    # ---- simulation
    sims_val = run_sims(val, lv_val, price_ref, cfg)
    sims_test = run_sims(test, lv_test, price_ref, cfg)
    say(f"simulated {len(sims_val) + len(sims_test)} policy runs")

    curve_val = curve_frame(sims_val, cfg.cost, "val")
    curve_test = curve_frame(sims_test, cfg.cost, "test")
    q_by_policy = choose_levels(curve_val)
    q_by_class = choose_levels_by_class(
        {q: t for (pol, q), t in sims_val.items() if pol == "lgbm_qr"}, abc, cfg.cost
    )
    chosen_val = _chosen_sims(sims_val, q_by_policy, q_by_class, abc)
    chosen_test = _chosen_sims(sims_test, q_by_policy, q_by_class, abc)
    q_text = {k: f"{v:.2f}" for k, v in q_by_policy.items()}
    q_text["lgbm_qr_abc"] = " ".join(f"{c}:{q_by_class[c]:.2f}" for c in sorted(q_by_class))

    boot = bootstrap_vs_baseline(
        chosen_test, BASELINE_POLICY, cfg.cost, cfg.bootstrap_draws, cfg.seed
    )
    selected = _selected_table(chosen_test, chosen_val, q_text, boot, cfg)
    frontier = frontier_table(curve_test, BASELINE_POLICY)
    segments = segment_table(chosen_test, BASELINE_POLICY, seg, cfg.cost)
    say("evaluation done")

    # ---- forecast accuracy on the test period
    assert out_b.y is not None
    scale = naive_scale(fb.cs, p, hist_start + 8 + p, val_first)
    weights = (data.sales[:, val_first - 56 : val_first] * data.price[:, val_first - 56 : val_first]).sum(axis=1)
    accuracy = forecast_accuracy(out_b.y, out_b.mean, scale, np.maximum(weights, 1e-9))
    quant_acc = quantile_accuracy(out_b.y, lv_test)

    # ---- what-if and sensitivity (re-priced physical flows; shelf life re-simulated)
    phys = physical_frame({"val": sims_val, "test": sims_test}, abc)
    sens_rows = []
    cache: dict[int, pd.DataFrame] = {cfg.cost.shelf_life_days: phys}
    for name, change in SENSITIVITY:
        cost_s = replace(cfg.cost, **change)
        w = cost_s.shelf_life_days
        if w not in cache:
            cache[w] = physical_frame(
                {
                    "val": run_sims(val, lv_val, price_ref, cfg, w),
                    "test": run_sims(test, lv_test, price_ref, cfg, w),
                },
                abc,
            )
        res = evaluate_scenario(cache[w], cost_s)
        res.insert(0, "scenario", name)
        sens_rows.append(res)
    sensitivity = pd.concat(sens_rows, ignore_index=True)
    say("sensitivity done")

    # ---- final fit: tomorrow's order-up-to levels
    say("final fit for the recommendations")
    models_c = fit_pass(fb, n_days - 1 - p, cfg, lambda m: say(m))
    out_c = predict_origins(fb, models_c, np.array([n_days - 1]), with_target=False)
    del models_c
    recs = _recommendations(data, cfg, out_c, abc, pattern)

    sample = _sample_predictions(data, cfg, test, out_b, revenue_hist)

    curves = pd.concat([curve_val, curve_test], ignore_index=True)
    summary = _summary(
        data, cfg, selected, frontier, accuracy, boot, q_by_policy, q_by_class,
        (val_first, val_last, test_first, test_last), time.time() - t_start,
    )
    tables = {
        "forecast_accuracy": accuracy,
        "quantile_accuracy": quant_acc,
        "policy_curves": curves,
        "policy_selected": selected,
        "frontier": frontier,
        "bootstrap": boot,
        "segment_results": segments,
        "sensitivity": sensitivity,
        "physical_totals": phys,
        "segments": seg,
        "recommendations": recs,
        "sample_predictions": sample,
    }
    meta = _meta(data, cfg)
    result = {"tables": tables, "summary": summary, "meta": meta}
    if results_dir is not None:
        write_results(results_dir, result)
        say(f"results written to {results_dir}")
        if figures:
            from .reporting import make_figures

            make_figures(Path(results_dir))
            say("figures written")
    say("pipeline complete")
    return result


# ----------------------------------------------------------------------------- outputs


def _recommendations(
    data: M5Data, cfg: ExperimentConfig, out: PassOutput, abc: np.ndarray, pattern: np.ndarray
) -> pd.DataFrame:
    n_days = data.n_days
    rec = data.ids[["series_id", "item_id", "dept_id", "store_id", "state_id"]].copy()
    rec["abc"] = abc
    rec["pattern"] = pattern
    rec["as_of"] = str(data.dates[n_days - 1].date())
    rec["covers_from"] = str(data.dates[n_days].date())
    rec["covers_days"] = cfg.protection
    rec["price"] = data.price[:, n_days - 1].round(2)
    rec["forecast_mean"] = out.mean["lgbm"][:, 0].round(3)
    rec["forecast_ma28"] = out.mean["ma28"][:, 0].round(3)
    for j, q in enumerate(cfg.quantiles):
        rec[f"order_up_to_q{int(round(q * 100))}"] = np.ceil(out.qraw[:, 0, j]).astype(int)
    return rec


def _sample_predictions(
    data: M5Data, cfg: ExperimentConfig, test: Period, out: PassOutput, revenue: np.ndarray
) -> pd.DataFrame:
    """A small slice of the test predictions for the dashboard: top sellers plus a random draw."""
    rng = np.random.default_rng(cfg.seed)
    n = data.n_series
    top = np.argsort(-revenue)[: min(30, n)]
    rest = np.setdiff1d(np.arange(n), top)
    pick = np.sort(np.concatenate([top, rng.choice(rest, size=min(30, rest.size), replace=False)]))
    frames = []
    for i in pick:
        d = {
            "series_id": data.ids["series_id"].iloc[i],
            "date": test.dates.strftime("%Y-%m-%d"),
            "sales_day": test.demand_day[i],
            "demand_next": out.y[i] if out.y is not None else np.nan,
            "forecast_mean": out.mean["lgbm"][i],
            "forecast_ma28": out.mean["ma28"][i],
        }
        for j, q in enumerate(cfg.quantiles):
            d[f"q{int(round(q * 100))}"] = out.qraw[i, :, j]
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True).round(2)


def _summary(
    data: M5Data,
    cfg: ExperimentConfig,
    selected: pd.DataFrame,
    frontier: pd.DataFrame,
    accuracy: pd.DataFrame,
    boot: pd.DataFrame,
    q_by_policy: dict[str, float],
    q_by_class: dict[str, float],
    tl: tuple[int, int, int, int],
    seconds: float,
) -> dict:
    val_first, val_last, test_first, test_last = tl
    sel = selected.set_index("policy")
    best_acc = accuracy.sort_values("wape").iloc[0]
    f95 = frontier[(frontier.fill_target == 0.95)].set_index("policy")
    return {
        "dataset": data.dataset,
        "n_series": data.n_series,
        "n_days": data.n_days,
        "protection_days": cfg.protection,
        "lead_time_days": cfg.lead_time,
        "validation_period": [str(data.dates[val_first].date()), str(data.dates[val_last].date())],
        "test_period": [str(data.dates[test_first].date()), str(data.dates[test_last].date())],
        "chosen_service_level": q_by_policy,
        "chosen_service_level_by_abc": q_by_class,
        "best_forecast_by_wape": str(best_acc["method"]),
        "policies": {
            p: {
                "fill_rate": float(sel.loc[p, "fill_rate"]),
                "avg_inventory": float(sel.loc[p, "avg_inventory"]),
                "total_cost": float(sel.loc[p, "total_cost"]),
                "cost_pct_revenue": float(sel.loc[p, "cost_pct_revenue"]),
                **{
                    k: float(sel.loc[p, k])
                    for k in ("cost_reduction_pct", "cost_reduction_pct_lo", "cost_reduction_pct_hi")
                    if k in sel.columns and pd.notna(sel.loc[p, k])
                },
            }
            for p in sel.index
        },
        "inventory_vs_baseline_at_95pct_fill": {
            p: (None if pd.isna(f95.loc[p, "inventory_vs_baseline_pct"])
                else float(f95.loc[p, "inventory_vs_baseline_pct"]))
            for p in f95.index
        },
        "runtime_seconds": round(seconds, 1),
    }


def _meta(data: M5Data, cfg: ExperimentConfig) -> dict:
    return {
        "config": cfg.to_dict(),
        "dataset": data.dataset,
        "series_filter": data.report,
        "policy_labels": POLICY_LABELS,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }


def write_results(results_dir: str | Path, result: dict) -> None:
    out = Path(results_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for name, df in result["tables"].items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    (out / "meta.json").write_text(json.dumps(result["meta"], indent=2))
