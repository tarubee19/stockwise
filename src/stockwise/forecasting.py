"""Demand models and walk-forward passes.

For every origin day the question is: how many units will be demanded over the next ``P`` days?
Two kinds of answer are produced:

* a *mean* forecast (baselines and a LightGBM Tweedie model), scored for accuracy;
* *quantile* forecasts (LightGBM quantile regression), the input of the inventory decision.

A *pass* trains on origins whose targets were already known at the first prediction origin and
then predicts every origin of a period. The study uses three passes: validation (used to pick
service levels), test (used to report) and a final fit for tomorrow's recommendations.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from .config import ExperimentConfig
from .features import (
    CATEGORICAL,
    FEATURE_NAMES,
    MIN_ORIGIN,
    FeatureBuilder,
    column,
    flatten,
)

Log = Callable[[str], None]


def _noop(_: str) -> None:
    return None


@dataclass
class PassOutput:
    """Everything a period needs, as (n_series, n_origins) arrays unless stated."""

    origins: np.ndarray
    y: np.ndarray | None  # demand over the next P days (None for the final, target-less pass)
    mean: dict[str, np.ndarray]  # method -> mean forecast of that demand
    qraw: np.ndarray  # (n_series, n_origins, n_quantiles) LightGBM quantile forecasts
    ma_sigma: np.ndarray  # std of the P-day demand implied by the 28-day daily std
    cv_inputs: dict[str, np.ndarray]  # a few features kept for the simulator


class DemandModels:
    """LightGBM Tweedie mean model plus one quantile model per service level."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.mean: lgb.Booster | None = None
        self.quantile: dict[float, lgb.Booster] = {}

    def _params(self, objective: str, **extra) -> dict:
        c = self.cfg
        return {
            "objective": objective,
            "learning_rate": c.learning_rate,
            "num_leaves": c.num_leaves,
            "min_data_in_leaf": c.min_data_in_leaf,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "max_bin": 127,
            "num_threads": c.n_jobs,
            "seed": c.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
            **extra,
        }

    def fit(self, x: np.ndarray, y: np.ndarray, log: Log = _noop) -> DemandModels:
        ds = lgb.Dataset(
            x,
            label=y,
            feature_name=FEATURE_NAMES,
            categorical_feature=CATEGORICAL,
            params={"max_bin": 127, "verbosity": -1},
        )
        ds.construct()
        t0 = time.time()
        self.mean = lgb.train(
            self._params("tweedie", tweedie_variance_power=1.2),
            ds,
            num_boost_round=self.cfg.n_estimators,
        )
        log(f"    mean model fitted ({time.time() - t0:.0f}s)")
        for q in self.cfg.quantiles:
            t0 = time.time()
            self.quantile[q] = lgb.train(
                self._params("quantile", alpha=q), ds, num_boost_round=self.cfg.n_estimators
            )
            log(f"    quantile {q:.2f} fitted ({time.time() - t0:.0f}s)")
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, quantiles) with quantiles sorted so they never cross."""
        if self.mean is None:
            raise RuntimeError("models are not fitted")
        mean = np.maximum(self.mean.predict(x), 0.0)
        q = np.column_stack([self.quantile[k].predict(x) for k in sorted(self.quantile)])
        q = np.maximum.accumulate(np.maximum(q, 0.0), axis=1)
        return mean.astype(np.float32), q.astype(np.float32)


def training_origins(last_origin: int, cfg: ExperimentConfig) -> np.ndarray:
    """Origins used to fit a pass, counting back from ``last_origin`` with the stride."""
    start = max(last_origin - cfg.train_window + 1, MIN_ORIGIN)
    origins = np.arange(last_origin, start - 1, -cfg.train_stride)[::-1]
    if origins.size == 0:
        raise ValueError("training window is empty")
    return origins


def fit_pass(
    fb: FeatureBuilder, last_origin: int, cfg: ExperimentConfig, log: Log = _noop
) -> DemandModels:
    """Fit models on origins up to ``last_origin`` (their targets must already be known)."""
    origins = training_origins(last_origin, cfg)
    x = flatten(fb.build(origins))
    y = fb.target(origins).T.reshape(-1)
    log(f"  training rows: {len(y):,} ({origins.size} origin days x {fb.n:,} series)")
    return DemandModels(cfg).fit(x, y, log)


def predict_origins(
    fb: FeatureBuilder,
    models: DemandModels,
    origins: np.ndarray,
    with_target: bool = True,
    chunk: int = 20,
) -> PassOutput:
    """Predict every origin in chunks to keep memory bounded."""
    n, p = fb.n, fb.P
    n_o = len(origins)
    n_q = len(models.quantile)
    mean_lgbm = np.empty((n, n_o), dtype=np.float32)
    qraw = np.empty((n, n_o, n_q), dtype=np.float32)
    names = ("snaive", "sn4", "ma28", "sba")
    base = {k: np.empty((n, n_o), dtype=np.float32) for k in names}
    sigma = np.empty((n, n_o), dtype=np.float32)
    mean28 = np.empty((n, n_o), dtype=np.float32)
    mean365 = np.empty((n, n_o), dtype=np.float32)

    for s in range(0, n_o, chunk):
        sl = slice(s, min(s + chunk, n_o))
        x = fb.build(origins[sl], with_target)
        m, q = models.predict(flatten(x))
        k = x.shape[0]
        mean_lgbm[:, sl] = m.reshape(k, n).T
        qraw[:, sl, :] = q.reshape(k, n, n_q).transpose(1, 0, 2)
        base["snaive"][:, sl] = column(x, "sn1")
        base["sn4"][:, sl] = column(x, "sn_mean")
        base["ma28"][:, sl] = column(x, "mean28") * p
        base["sba"][:, sl] = column(x, "sba")
        sigma[:, sl] = column(x, "std28") * np.sqrt(p)
        mean28[:, sl] = column(x, "mean28")
        mean365[:, sl] = column(x, "mean365")

    mean = {**base, "lgbm": mean_lgbm}
    y = fb.target(origins) if with_target else None
    return PassOutput(
        origins=np.asarray(origins),
        y=y,
        mean=mean,
        qraw=qraw,
        ma_sigma=sigma,
        cv_inputs={"mean28": mean28, "mean365": mean365},
    )
