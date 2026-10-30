#!/usr/bin/env python
"""Is LightGBM the right forecaster, and are its settings justified?  (M5 FOODS, real data)

Three experiments, all on the same features, walk-forward origins and P-day target as the main study:

1. TUNING ON VALIDATION ONLY. A small grid over the LightGBM settings that the main study fixes
   (num_leaves, learning_rate x rounds, min_data_in_leaf, Tweedie variance power, objective),
   scored on the *validation* period. The test period is never used to choose settings.

2. GLOBAL ML MODELS ON THE TEST PERIOD. LightGBM (Tweedie, Poisson, L2), XGBoost (Tweedie),
   CatBoost (Tweedie) and a ridge regression on the same 40 features, against the study's
   baselines (seasonal naive, moving average, Croston-SBA). Series-bootstrap 95% intervals on
   the WAPE difference vs LightGBM.

3. LOCAL STATISTICAL MODELS AND A TIME-SERIES FOUNDATION MODEL on a series subsample with weekly
   origins (they are fitted per series, so the full panel is too slow): AutoETS, Croston
   (optimised), TSB, IMAPA from statsforecast, and Amazon Chronos-Bolt (zero-shot, 2024-25).

Usage:
    python scripts/model_comparison.py --data data/m5_foods.zip --out results/model_comparison
        [--series 3000] [--local-series 500] [--skip-local] [--skip-chronos]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockwise.config import ExperimentConfig  # noqa: E402
from stockwise.data import load_panel  # noqa: E402
from stockwise.evaluate import forecast_accuracy, naive_scale  # noqa: E402
from stockwise.features import CATEGORICAL, FEATURE_NAMES, FeatureBuilder, column, flatten  # noqa: E402
from stockwise.forecasting import training_origins  # noqa: E402
from stockwise.pipeline import timeline  # noqa: E402


def subsample(data, n: int, seed: int):
    """Stratified (store x dept) random subset of series, same seed for every experiment."""
    if n >= data.n_series:
        return data
    rng = np.random.default_rng(seed)
    ids = data.ids.reset_index(drop=True)
    strata = ids["store_id"].astype(str) + "|" + ids["dept_id"].astype(str)
    keep = []
    for _, idx in strata.groupby(strata).groups.items():
        k = max(1, round(n * len(idx) / len(ids)))
        keep += list(rng.choice(np.asarray(idx), size=min(k, len(idx)), replace=False))
    keep = np.sort(np.asarray(keep))
    fields = {}
    for name in data.__dataclass_fields__:
        v = getattr(data, name)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == data.n_series:
            fields[name] = v[keep]
        elif isinstance(v, pd.DataFrame) and len(v) == data.n_series:
            fields[name] = v.iloc[keep].reset_index(drop=True)
        else:
            fields[name] = v
    return type(data)(**fields)


# ----------------------------------------------------------------------------- models
def lgb_model(params: dict, rounds: int):
    import lightgbm as lgb

    def fit(x, y):
        ds = lgb.Dataset(x, label=y, feature_name=FEATURE_NAMES, categorical_feature=CATEGORICAL,
                         params={"max_bin": 127, "verbosity": -1})
        base = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127, "seed": 42,
                "deterministic": True, "force_row_wise": True, "verbosity": -1, "num_threads": 2}
        booster = lgb.train({**base, **params}, ds, num_boost_round=rounds)
        return lambda xx: np.maximum(booster.predict(xx), 0)
    return fit


def xgb_model():
    import xgboost as xgb

    def fit(x, y):
        m = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.2, n_estimators=300,
                             learning_rate=0.05, max_depth=8, min_child_weight=100, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", max_bin=127, n_jobs=2, random_state=42)
        m.fit(x, y)
        return lambda xx: np.maximum(m.predict(xx), 0)
    return fit


def cat_model():
    from catboost import CatBoostRegressor
    cat_idx = [FEATURE_NAMES.index(c) for c in CATEGORICAL]

    def fit(x, y):
        X = pd.DataFrame(x, columns=FEATURE_NAMES)
        for c in CATEGORICAL:
            X[c] = X[c].astype(int)
        m = CatBoostRegressor(loss_function="Tweedie:variance_power=1.2", iterations=300, learning_rate=0.08,
                              depth=8, random_seed=42, verbose=0, thread_count=2, cat_features=cat_idx)
        m.fit(X, y)

        def pred(xx):
            XX = pd.DataFrame(xx, columns=FEATURE_NAMES)
            for c in CATEGORICAL:
                XX[c] = XX[c].astype(int)
            return np.maximum(m.predict(XX), 0)
        return pred
    return fit


def ridge_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cat_idx = [FEATURE_NAMES.index(c) for c in CATEGORICAL]
    num_idx = [i for i in range(len(FEATURE_NAMES)) if i not in cat_idx]

    def fit(x, y):
        pipe = make_pipeline(ColumnTransformer([("n", StandardScaler(), num_idx),
                                                ("c", OneHotEncoder(handle_unknown="ignore"), cat_idx)]),
                             Ridge(alpha=10.0))
        pipe.fit(np.nan_to_num(x), y)
        return lambda xx: np.maximum(pipe.predict(np.nan_to_num(xx)), 0)
    return fit


# ----------------------------------------------------------------------------- passes
def make_xy(fb, origins):
    x = flatten(fb.build(origins))
    y = fb.target(origins).T.reshape(-1)
    return x, y


def predict_panel(fb, predict, origins, chunk=30):
    out = np.empty((fb.n, len(origins)), dtype=np.float32)
    base = {k: np.empty((fb.n, len(origins)), dtype=np.float32) for k in ("snaive", "ma28", "sba")}
    for s in range(0, len(origins), chunk):
        sl = slice(s, min(s + chunk, len(origins)))
        x = fb.build(origins[sl])
        out[:, sl] = predict(flatten(x)).reshape(x.shape[0], fb.n).T
        base["snaive"][:, sl] = column(x, "sn1")
        base["ma28"][:, sl] = column(x, "mean28") * fb.P
        base["sba"][:, sl] = column(x, "sba")
    return out, base


def wape(y, f):
    return float(np.abs(f - y).sum() / y.sum())


def boot_diff(y, f_a, f_b, draws=1000, seed=0):
    """Series-resampling bootstrap of WAPE(b) - WAPE(a)."""
    rng = np.random.default_rng(seed)
    ea, eb, tot = np.abs(f_a - y).sum(1), np.abs(f_b - y).sum(1), y.sum(1)
    idx = rng.integers(0, len(y), (draws, len(y)))
    d = (eb[idx].sum(1) - ea[idx].sum(1)) / tot[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ----------------------------------------------------------------------------- experiments
def tuning(fb, cfg, val_first, val_last, log):
    p = cfg.protection
    tr = training_origins(val_first - p, cfg)
    x, y = make_xy(fb, tr)
    origins = np.arange(val_first, val_last + 1)
    yv = fb.target(origins)
    grid = [("current: leaves 63, lr 0.05 x 300, min_leaf 100, tweedie 1.2", {}, 300)]
    for leaves in (15, 255):
        grid.append((f"num_leaves {leaves}", {"num_leaves": leaves}, 300))
    for lr, r in ((0.1, 150), (0.025, 600)):
        grid.append((f"lr {lr} x {r} rounds", {"learning_rate": lr}, r))
    grid.append(("lr 0.05 x 600 rounds", {}, 600))
    for mdl in (20, 500):
        grid.append((f"min_data_in_leaf {mdl}", {"min_data_in_leaf": mdl}, 300))
    for vp in (1.1, 1.5):
        grid.append((f"tweedie_variance_power {vp}", {"tweedie_variance_power": vp}, 300))
    grid.append(("objective poisson", {"objective": "poisson"}, 300))
    grid.append(("objective L2 (MSE)", {"objective": "regression"}, 300))
    rows = []
    for name, params, rounds in grid:
        params = {"objective": "tweedie", "tweedie_variance_power": 1.2, **params}
        t0 = time.time()
        pred = lgb_model(params, rounds)(x, y)
        f, _ = predict_panel(fb, pred, origins)
        rows.append({"setting": name, "val_wape": wape(yv, f), "val_bias": float((f - yv).sum() / yv.sum()),
                     "fit_s": time.time() - t0})
        log(f"  tune {name:58s} WAPE {rows[-1]['val_wape']:.4f}  ({rows[-1]['fit_s']:.0f}s)")
    return pd.DataFrame(rows)


def global_models(fb, cfg, test_first, test_last, data, log):
    p = cfg.protection
    tr = training_origins(test_first - p, cfg)
    x, y = make_xy(fb, tr)
    origins = np.arange(test_first, test_last + 1)
    yt = fb.target(origins)
    models = {
        "LightGBM Tweedie (current)": lgb_model({"objective": "tweedie", "tweedie_variance_power": 1.2}, 300),
        "LightGBM Poisson": lgb_model({"objective": "poisson"}, 300),
        "LightGBM L2": lgb_model({"objective": "regression"}, 300),
        "XGBoost Tweedie": xgb_model(),
        "CatBoost Tweedie": cat_model(),
        "Ridge (same features)": ridge_model(),
    }
    fcs, times, base = {}, {}, None
    for name, fitter in models.items():
        t0 = time.time()
        pred = fitter(x, y)
        fcs[name], base = predict_panel(fb, pred, origins)
        times[name] = time.time() - t0
        log(f"  {name:30s} test WAPE {wape(yt, fcs[name]):.4f} ({times[name]:.0f}s)")
    fcs.update({"Seasonal naive": base["snaive"], "Moving avg 28d": base["ma28"], "Croston-SBA": base["sba"]})
    # same scale and revenue weights as the main study (pipeline.run_study)
    hist_start = data.report["history_start_day"]
    val_first = test_first - cfg.val_days
    scale = naive_scale(fb.cs, p, hist_start + 8 + p, val_first)
    weights = np.maximum((data.sales[:, val_first - 56: val_first] * data.price[:, val_first - 56: val_first]).sum(1), 1e-9)
    acc = forecast_accuracy(yt, fcs, scale, weights)
    ref = fcs["LightGBM Tweedie (current)"]
    acc["wape_diff_vs_lgbm_ci95"] = [None if k == "LightGBM Tweedie (current)" else boot_diff(yt, ref, f)
                                     for k, f in fcs.items()]
    acc["fit_predict_s"] = [times.get(k) for k in fcs]
    return acc, fcs, yt, origins


def local_models(data, cfg, fcs_global, yt, origins, n_local, seed, use_chronos, log):
    """statsforecast + Chronos on a series subsample and weekly origins; compared on exactly those cells."""
    rng = np.random.default_rng(seed)
    rows_idx = np.sort(rng.choice(data.n_series, size=min(n_local, data.n_series), replace=False))
    o_idx = np.arange(0, len(origins), 7)
    sel_origins = origins[o_idx]
    p = cfg.protection
    y = yt[np.ix_(rows_idx, o_idx)]
    out = {k: v[np.ix_(rows_idx, o_idx)] for k, v in fcs_global.items()
           if k in ("LightGBM Tweedie (current)", "Croston-SBA", "Moving avg 28d", "Seasonal naive")}
    sales = data.sales[rows_idx].astype(float)

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import IMAPA, TSB, AutoETS, CrostonOptimized
        models = [AutoETS(season_length=7), CrostonOptimized(), TSB(alpha_d=0.2, alpha_p=0.2), IMAPA()]
        names = ["AutoETS", "CrostonOptimized", "TSB", "IMAPA"]
        res = {n: np.zeros_like(y) for n in names}
        t0 = time.time()
        for j, o in enumerate(sel_origins):
            hist = sales[:, max(0, o - 364): o + 1]
            df = pd.DataFrame({"unique_id": np.repeat(np.arange(len(rows_idx)), hist.shape[1]),
                               "ds": np.tile(np.arange(hist.shape[1]), len(rows_idx)),
                               "y": hist.reshape(-1)})
            fc = StatsForecast(models=models, freq=1, n_jobs=2).forecast(df=df, h=p)
            fc = fc.reset_index() if "unique_id" not in fc.columns else fc
            agg = fc.groupby("unique_id")[[m.__class__.__name__ if not hasattr(m, "alias") else m.alias
                                           for m in models]].sum()
            for n, col in zip(names, agg.columns):
                res[n][:, j] = np.maximum(agg[col].to_numpy(), 0)
        out.update({f"{n} (statsforecast)": v for n, v in res.items()})
        log(f"  statsforecast done ({time.time() - t0:.0f}s)")
    except Exception as e:  # pragma: no cover
        log(f"  statsforecast skipped: {e!r}")

    if use_chronos:
        try:
            import torch
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                                       device_map="cuda" if torch.cuda.is_available() else "cpu",
                                                       torch_dtype=torch.float32)
            t0 = time.time()
            f = np.zeros_like(y)
            for j, o in enumerate(sel_origins):
                ctx = torch.tensor(sales[:, max(0, o - 511): o + 1], dtype=torch.float32)
                q, mean = pipe.predict_quantiles(ctx, prediction_length=p, quantile_levels=[0.5])
                f[:, j] = np.maximum(mean.numpy().sum(1), 0)
            out["Chronos-Bolt-small (zero-shot)"] = f
            log(f"  chronos done ({time.time() - t0:.0f}s)")
        except Exception as e:  # pragma: no cover
            log(f"  chronos skipped: {e!r}")

    ref = out["LightGBM Tweedie (current)"]
    rows = [{"method": k, "wape": wape(y, v), "bias": float((v - y).sum() / y.sum()),
             "wape_diff_vs_lgbm_ci95": None if k.startswith("LightGBM") else boot_diff(y, ref, v)}
            for k, v in out.items()]
    return pd.DataFrame(rows).sort_values("wape"), {"series": int(len(rows_idx)), "origins": int(len(sel_origins))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "model_comparison"))
    ap.add_argument("--series", type=int, default=3000)
    ap.add_argument("--local-series", type=int, default=500)
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--skip-local", action="store_true")
    ap.add_argument("--skip-chronos", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(m):
        print(f"[{time.time() - t_start:7.0f}s] {m}", flush=True)

    cfg = ExperimentConfig(n_jobs=2)
    data = subsample(load_panel(a.data, cfg), a.series, a.seed)
    log(f"{data.n_series} series ({data.dataset})")
    fb = FeatureBuilder(data, cfg.protection)
    val_first, val_last, test_first, test_last = timeline(data.sales.shape[1], cfg)
    meta = {"series": data.n_series, "dataset": data.dataset, "val": [str(data.dates[val_first].date()), str(data.dates[val_last].date())],
            "test": [str(data.dates[test_first].date()), str(data.dates[test_last].date())]}
    if not a.skip_tuning:
        tune = tuning(fb, cfg, val_first, val_last, log)
        tune.to_csv(out / "tuning_validation.csv", index=False)
    acc, fcs, yt, origins = global_models(fb, cfg, test_first, test_last, data, log)
    acc.to_csv(out / "global_models_test.csv", index=False)
    if not a.skip_local:
        loc, info = local_models(data, cfg, fcs, yt, origins, a.local_series, a.seed, not a.skip_chronos, log)
        loc.to_csv(out / "local_models_test.csv", index=False)
        meta["local"] = info
    meta["runtime_s"] = time.time() - t_start
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log("done")


if __name__ == "__main__":
    main()
