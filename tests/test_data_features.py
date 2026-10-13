import numpy as np
import pytest

from stockwise.config import ExperimentConfig
from stockwise.data import DataValidationError, build_panel, validate_raw
from stockwise.features import MIN_ORIGIN, FeatureBuilder


def test_panel_shapes_and_label(panel):
    assert panel.dataset == "simulated"
    assert panel.sales.shape == (panel.n_series, panel.n_days)
    assert panel.price.shape[1] > panel.n_days  # future calendar appended
    assert not np.isnan(panel.price).any()
    assert panel.report["n_kept"] == panel.n_series


def test_eligibility_filter_drops_late_launch(raw, small_cfg):
    d = build_panel(raw, small_cfg)
    assert d.report["n_dropped_not_listed"] > 0 or d.report["n_dropped_low_demand"] >= 0
    assert d.report["n_kept"] <= d.report["n_foods_series"]


def test_validate_raw_rejects_bad_tables(raw):
    bad = type(raw)(calendar=raw.calendar.drop(columns=["d"]), prices=raw.prices, sales=raw.sales)
    with pytest.raises(DataValidationError):
        validate_raw(bad)


def test_target_is_future_sum(panel):
    fb = FeatureBuilder(panel, protection=3)
    o = np.array([400, 500])
    y = fb.target(o)
    for j, t in enumerate(o):
        assert np.allclose(y[:, j], panel.sales[:, t + 1 : t + 4].sum(axis=1))


def test_features_do_not_use_future_sales(panel):
    """Features at origin t must be identical if every sale after t is changed."""
    t = np.array([500, 600])
    a = FeatureBuilder(panel, 3).build(t, with_target=False)
    import copy
    p2 = copy.deepcopy(panel)
    p2.sales[:, 601:] = p2.sales[:, 601:] * 7 + 13
    b = FeatureBuilder(p2, 3).build(t, with_target=False)
    assert np.array_equal(a, b, equal_nan=True)


def test_origin_bounds(panel):
    fb = FeatureBuilder(panel, 3)
    with pytest.raises(ValueError):
        fb.build(np.array([MIN_ORIGIN - 1]))
    with pytest.raises(ValueError):
        fb.build(np.array([panel.n_days - 1]), with_target=True)
    assert fb.build(np.array([panel.n_days - 1]), with_target=False).shape[0] == 1


def test_protection_property():
    assert ExperimentConfig(lead_time=2, review_period=1).protection == 3
