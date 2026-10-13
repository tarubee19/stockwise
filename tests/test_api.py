import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockwise.api import create_app


@pytest.fixture()
def results(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"dataset": "m5"}))
    (tmp_path / "summary.json").write_text(json.dumps({"x": 1}))
    pd.DataFrame([{
        "series_id": "CA_1|I1", "item_id": "I1", "store_id": "CA_1", "abc": "A", "pattern": "smooth",
        "as_of": "2016-01-01", "covers_from": "2016-01-02", "covers_days": 3, "forecast_mean": 10.0,
        "order_up_to_q90": 20, "order_up_to_q95": 25,
    }]).to_csv(tmp_path / "recommendations.csv", index=False)
    return tmp_path


def test_recommend_order(results):
    c = TestClient(create_app(results))
    r = c.get("/recommend-order", params=dict(store="CA_1", item="I1", service_level=0.9,
                                              on_hand=5, on_order=3))
    assert r.status_code == 200 and r.json()["recommended_order_units"] == 12
    assert r.headers["x-data-label"] == "REAL-M5"
    over = c.get("/recommend-order", params=dict(store="CA_1", item="I1", on_hand=100))
    assert over.json()["recommended_order_units"] == 0


def test_errors(results):
    c = TestClient(create_app(results))
    assert c.get("/recommend-order", params=dict(store="CA_1", item="I1", service_level=0.5)).status_code == 422
    assert c.get("/recommend-order", params=dict(store="CA_1", item="NOPE")).status_code == 404
    assert c.get("/policies").status_code == 503  # table missing


def test_no_results(tmp_path):
    c = TestClient(create_app(tmp_path))
    assert c.get("/health").json()["results_ready"] is False
    assert c.get("/summary").status_code == 503
