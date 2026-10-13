import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockwise.config import ExperimentConfig  # noqa: E402
from stockwise.data import (
    RawM5,  # noqa: E402
    build_panel,  # noqa: E402
)
from stockwise.synthetic import make_m5_frames  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg():
    return replace(
        ExperimentConfig(), val_days=30, test_days=60, train_window=300, n_estimators=15,
        n_jobs=2, bootstrap_draws=50,
    )


@pytest.fixture(scope="session")
def raw():
    cal, prices, sales = make_m5_frames(n_items_per_dept=6, n_days=900, seed=1)
    return RawM5(calendar=cal, prices=prices, sales=sales)


@pytest.fixture(scope="session")
def panel(raw, small_cfg):
    return build_panel(raw, small_cfg)
