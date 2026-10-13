"""Read-only REST API over a finished study (``results/``).

    uvicorn stockwise.api:app --port 8000          # RESULTS_DIR env var, default ./results

Every response carries ``X-Data-Label`` (REAL-M5 or SIMULATED) so a screenshot can never be
mistaken for the other kind of run.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse


def _label(meta: dict) -> str:
    return "SIMULATED" if meta.get("dataset") == "simulated" else "REAL-M5"


class Store:
    """Lazy, cached access to the result files."""

    def __init__(self, results_dir: str | Path):
        self.dir = Path(results_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def ready(self) -> bool:
        return (self.dir / "meta.json").exists() and (self.dir / "summary.json").exists()

    def json(self, name: str) -> dict:
        return json.loads((self.dir / name).read_text())

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.dir / f"{name}.csv"
            if not path.exists():
                raise HTTPException(503, f"results/{name}.csv is missing; run the pipeline first")
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def create_app(results_dir: str | Path | None = None) -> FastAPI:
    store = Store(results_dir or os.environ.get("RESULTS_DIR", "results"))
    app = FastAPI(title="StockWise API", version="1.0")

    @app.middleware("http")
    async def label_header(request: Request, call_next):
        response = await call_next(request)
        label = _label(store.json("meta.json")) if store.ready() else "NO-RESULTS"
        response.headers["X-Data-Label"] = label
        return response

    def need() -> None:
        if not store.ready():
            raise HTTPException(503, "no results found; run scripts/run_pipeline.py first")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "results_ready": store.ready()}

    @app.get("/meta")
    def meta() -> dict:
        need()
        m = store.json("meta.json")
        return {"label": _label(m), **m}

    @app.get("/summary")
    def summary() -> dict:
        need()
        return {"label": _label(store.json("meta.json")), **store.json("summary.json")}

    @app.get("/policies")
    def policies() -> list[dict]:
        need()
        return _records(store.table("policy_selected"))

    @app.get("/accuracy")
    def accuracy() -> dict:
        need()
        return {
            "forecast": _records(store.table("forecast_accuracy")),
            "quantile": _records(store.table("quantile_accuracy")),
        }

    @app.get("/segments")
    def segments() -> list[dict]:
        need()
        return _records(store.table("segment_results"))

    @app.get("/sensitivity")
    def sensitivity() -> list[dict]:
        need()
        return _records(store.table("sensitivity"))

    @app.get("/recommend-order")
    def recommend_order(
        store_id: str = Query(..., alias="store"),
        item_id: str = Query(..., alias="item"),
        service_level: float = Query(0.90),
        on_hand: float = Query(0.0, ge=0),
        on_order: float = Query(0.0, ge=0),
    ) -> JSONResponse:
        need()
        rec = store.table("recommendations")
        levels = sorted(
            float(c.removeprefix("order_up_to_q")) / 100
            for c in rec.columns
            if c.startswith("order_up_to_q")
        )
        match = next((q for q in levels if abs(q - service_level) < 1e-9), None)
        if match is None:
            raise HTTPException(422, f"service_level must be one of {levels}")
        row = rec[(rec["store_id"] == store_id) & (rec["item_id"] == item_id)]
        if row.empty:
            raise HTTPException(404, f"no series for store={store_id} item={item_id}")
        r = row.iloc[0]
        target = float(r[f"order_up_to_q{round(match * 100)}"])
        order = max(0, math.ceil(target - on_hand - on_order))
        return JSONResponse(
            {
                "store": store_id,
                "item": item_id,
                "as_of": r["as_of"],
                "covers_from": r["covers_from"],
                "covers_days": int(r["covers_days"]),
                "service_level": match,
                "forecast_mean_units": float(r["forecast_mean"]),
                "order_up_to_level": target,
                "on_hand": on_hand,
                "on_order": on_order,
                "recommended_order_units": order,
                "abc": r["abc"],
                "pattern": r["pattern"],
                "note": "Target is the "
                f"{match:.0%} quantile of demand over the next {int(r['covers_days'])} days.",
            }
        )

    return app


app = create_app()
