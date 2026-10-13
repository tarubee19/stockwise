import numpy as np
import pandas as pd
import pytest

from stockwise.config import CostConfig
from stockwise.evaluate import (
    bootstrap_vs_baseline,
    forecast_accuracy,
    pinball,
    value_at_fill,
)
from stockwise.policies import abc_classes, demand_pattern, fit_dispersion, nb_quantile
from stockwise.simulate import SimTotals
from stockwise.whatif import evaluate_scenario


def test_nb_quantile_monotone_and_poisson_limit():
    mu = np.array([1.0, 5.0, 20.0])
    a = np.full(3, 0.5)
    assert (nb_quantile(mu, a, 0.95) >= nb_quantile(mu, a, 0.8)).all()
    small = nb_quantile(mu, np.full(3, 0.01), 0.5)
    assert np.allclose(small, np.round(mu), atol=1)


def test_dispersion_recovers_overdispersion():
    rng = np.random.default_rng(0)
    mu = np.full((200, 200), 10.0)
    alpha = 0.5
    y = rng.negative_binomial(1 / alpha, (1 / alpha) / (1 / alpha + mu))
    est = fit_dispersion(mu, y, np.zeros(200, dtype=int))[0]
    assert est == pytest.approx(alpha, abs=0.05)


def test_abc_classes_shares():
    rev = np.array([100.0] * 2 + [10.0] * 10 + [1.0] * 100)
    c = abc_classes(rev)
    assert c[0] == "A" and c[-1] == "C"


def test_demand_pattern_labels():
    smooth = np.full((1, 200), 5.0) + np.tile([0, 1], 100)
    inter = np.zeros((1, 200))
    inter[0, ::10] = 3.0
    assert demand_pattern(smooth)[0] == "smooth"
    assert demand_pattern(inter)[0] in {"intermittent", "lumpy"}


def test_pinball_zero_when_exact_and_asymmetric():
    y = np.array([5.0])
    assert pinball(y, np.array([5.0]), 0.9)[0] == 0
    assert pinball(y, np.array([3.0]), 0.9)[0] > pinball(y, np.array([7.0]), 0.9)[0]


def test_forecast_accuracy_perfect_forecast():
    y = np.random.default_rng(0).poisson(4, (5, 30)).astype(float)
    out = forecast_accuracy(y, {"perfect": y}, np.ones(5), np.ones(5))
    assert out.loc[0, "wape"] == 0 and out.loc[0, "bias"] == 0


def test_value_at_fill_interpolation_and_range():
    c = pd.DataFrame({"fill_rate": [0.8, 0.9, 1.0], "avg_inventory": [1.0, 2.0, 4.0]})
    assert value_at_fill(c, 0.95, "avg_inventory") == pytest.approx(3.0)
    assert np.isnan(value_at_fill(c, 0.5, "avg_inventory"))


def _totals(scale, n=40):
    rng = np.random.default_rng(3)
    base = rng.uniform(50, 100, n)
    f = {k: np.zeros(n) for k in ("sold", "waste", "onhand_days", "inv_value_days", "waste_value",
                                  "lost_pmed", "lost_revenue", "revenue", "stockout_days", "ordered")}
    f["demand"] = base
    f["lost"] = base * 0.1 * scale
    f["lost_pmed"] = f["lost"]
    f["lost_revenue"] = f["lost"] * 1.5
    f["onhand_days"] = base * 2
    f["inv_value_days"] = base * 2
    f["sold"] = base - f["lost"]
    f["revenue"] = f["sold"] * 1.5
    return SimTotals(f, 30)


def test_bootstrap_identical_policy_is_zero_and_better_is_positive():
    chosen = {"ma_z": _totals(1.0), "same": _totals(1.0), "better": _totals(0.2)}
    out = bootstrap_vs_baseline(chosen, "ma_z", CostConfig(), draws=100, seed=0)
    same = out[(out.policy == "same") & (out.metric == "cost_reduction_pct")].iloc[0]
    better = out[(out.policy == "better") & (out.metric == "cost_reduction_pct")].iloc[0]
    assert same.estimate == pytest.approx(0) and better.ci_low > 0


def test_whatif_matches_pipeline_outputs(tmp_path):
    from dataclasses import replace

    from stockwise.config import ExperimentConfig
    from stockwise.data import load_panel
    from stockwise.pipeline import run_study
    from stockwise.synthetic import write_m5_folder

    cfg = replace(ExperimentConfig(), val_days=30, test_days=60, train_window=300,
                  n_estimators=10, n_jobs=2, bootstrap_draws=20)
    write_m5_folder(tmp_path / "d", n_items_per_dept=6, n_days=900, seed=2)
    data = load_panel(tmp_path / "d", cfg)
    run_study(data, cfg, results_dir=tmp_path / "r", figures=False)
    phys = pd.read_csv(tmp_path / "r" / "physical_totals.csv")
    sel = pd.read_csv(tmp_path / "r" / "policy_selected.csv").set_index("policy")
    again = evaluate_scenario(phys, cfg.cost).set_index("policy")
    for p in again.index:
        assert again.loc[p, "total_cost"] == pytest.approx(sel.loc[p, "total_cost"], rel=1e-9)
    # an expensive-holding scenario must raise holding cost
    hi = evaluate_scenario(phys, replace(cfg.cost, holding_rate=1.0)).set_index("policy")
    assert hi.loc["ma_z", "holding_cost"] > again.loc["ma_z", "holding_cost"]
    # API on these results
    from fastapi.testclient import TestClient

    from stockwise.api import create_app
    c = TestClient(create_app(tmp_path / "r"))
    assert c.get("/summary").headers["x-data-label"] == "SIMULATED"
    assert c.get("/policies").status_code == 200
    # pipeline is deterministic: same input -> same selected table
    run_study(data, cfg, results_dir=tmp_path / "r2", figures=False)
    a = (tmp_path / "r" / "policy_selected.csv").read_bytes()
    b = (tmp_path / "r2" / "policy_selected.csv").read_bytes()
    assert a == b
