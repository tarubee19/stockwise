import numpy as np
import pytest

from stockwise.config import CostConfig
from stockwise.simulate import aggregate, cost_components, metrics, simulate


def run(S, D, lead=0, shelf=7, init=0.0, warmup=0, price=1.0):
    S, D = np.atleast_2d(S).astype(float), np.atleast_2d(D).astype(float)
    n = D.shape[0]
    return simulate(S, D, np.full_like(D, price), np.full(n, price), np.full(n, init),
                    lead, shelf, warmup)


def test_hand_checked_lead_zero():
    # S=5 each day, lead 0: order placed end of day d arrives start of day d+1 (pipeline P=1).
    t = run([[5, 5, 5]], [[2, 7, 1]], lead=0, init=5)
    # day0: stock 5, demand 2 -> 3 left, order 2 (to 5). day1: +2 =5, demand 7 -> lost 2.
    # order 5. day2: +5 =5, demand 1.
    assert t["lost"][0] == 2
    assert t["sold"][0] == 8
    assert t["demand"][0] == 10


def test_unit_conservation():
    rng = np.random.default_rng(0)
    D = rng.poisson(3, (20, 200)).astype(float)
    S = np.full_like(D, 12.0)
    t = run(S, D, lead=2, shelf=5, init=6)
    assert np.allclose(t["demand"], t["sold"] + t["lost"])
    # received = initial + arrived; arrived <= ordered; all stock is sold, wasted or left
    assert (t["sold"] + t["waste"] <= 6 + t["ordered"] + 1e-9).all()


def test_fifo_and_waste():
    # 4 units on day 0, shelf life 2, no demand day 0 -> aged; demand 1 day1 consumes oldest;
    # remaining 3 expire at end of day 1 (age 1 -> waste at end of day when reaching shelf life).
    t = run([[0, 0, 0]], [[0, 1, 0]], lead=0, shelf=2, init=4)
    assert t["waste"][0] == 3 and t["sold"][0] == 1


def test_lead_time_delays_arrival():
    S = np.array([[10, 0, 0, 0, 0, 0]])
    D = np.array([[0, 0, 0, 5, 0, 0]])
    short = run(S, D, lead=1)  # arrives day 2
    long = run(S, D, lead=3)   # arrives day 4: too late for day 3 demand
    assert short["lost"][0] == 0
    assert long["lost"][0] == 5


def test_warmup_excluded():
    t = run([[5] * 6], [[3] * 6], lead=0, init=5, warmup=4)
    assert t["demand"][0] == 6 and t.n_days == 2


def test_validation_errors():
    with pytest.raises(ValueError):
        simulate(np.zeros((1, 3)), np.zeros((1, 4)), np.ones((1, 4)), np.ones(1), np.zeros(1), 1, 3, 0)
    with pytest.raises(ValueError):
        run([[1, 1]], [[1, 1]], warmup=2)


def test_cost_components_formula():
    cfg = CostConfig(margin=0.25, holding_rate=0.365, stockout_mult=2.0)
    phys = {"inv_value_days": 1000.0, "waste_value": 10.0, "lost_revenue": 40.0, "lost_pmed": 40.0}
    c = cost_components(phys, cfg)
    assert c["holding"] == pytest.approx(0.365 / 365 * 0.75 * 1000)
    assert c["waste"] == pytest.approx(7.5)
    assert c["lost"] == pytest.approx(2.0 * (40 - 0.75 * 40))
    assert c["total"] == pytest.approx(c["holding"] + c["waste"] + c["lost"])


def test_metrics_and_aggregate_consistent():
    rng = np.random.default_rng(1)
    D = rng.poisson(2, (5, 100)).astype(float)
    t = run(np.full_like(D, 6.0), D, lead=1, init=3)
    agg = aggregate(t)
    m = metrics(agg, CostConfig())
    assert m["fill_rate"] == pytest.approx(t["sold"].sum() / t["demand"].sum())
    mask = np.array([True, False, True, False, False])
    assert aggregate(t, mask)["n_series"] == 2
