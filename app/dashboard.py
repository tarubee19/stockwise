"""StockWise dashboard.  streamlit run app/dashboard.py   (RESULTS_DIR env var, default ./results)"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockwise.config import POLICY_COLORS, POLICY_LABELS, CostConfig  # noqa: E402
from stockwise.whatif import evaluate_scenario  # noqa: E402

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))

st.set_page_config(page_title="StockWise", layout="wide")


@st.cache_data
def load(path: str):
    p = Path(path)
    if not (p / "meta.json").exists():
        return None
    csv = {f.stem: pd.read_csv(f) for f in p.glob("*.csv")}
    return {
        "meta": json.loads((p / "meta.json").read_text()),
        "summary": json.loads((p / "summary.json").read_text()),
        **csv,
    }


res = load(str(RESULTS))
st.title("StockWise: demand forecasting and ordering policy")
if res is None:
    st.error(f"No results in {RESULTS}. Run `python scripts/run_pipeline.py --data ... ` first.")
    st.stop()

meta, summary = res["meta"], res["summary"]
simulated = meta.get("dataset") == "simulated"
if simulated:
    st.error("SIMULATED DATA. These numbers come from generated demo data, not from real sales.")
else:
    st.caption(
        "Data: M5 Forecasting (Walmart US, FOODS category). US supermarket data, not Indian "
        "dark stores. Costs are stated assumptions; the reliable findings are the relative ones."
    )

def pct(x: float) -> str:
    return f"{x:+.1f}%"

tabs = st.tabs(["Business impact", "Forecast accuracy", "Order recommender", "Segments", "What-if"])

with tabs[0]:
    sel = res["policy_selected"].set_index("policy")
    base = "ma_z"
    best = sel["total_cost"].drop(base).idxmin()
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline fill rate", f"{sel.loc[base, 'fill_rate']:.1%}")
    c2.metric(
        f"Best policy: {POLICY_LABELS[best]}",
        f"{sel.loc[best, 'fill_rate']:.1%} fill",
        pct((sel.loc[best, 'total_cost'] / sel.loc[base, 'total_cost'] - 1) * 100) + " cost",
        delta_color="inverse",
    )
    c3.metric(
        "Inventory vs baseline",
        pct((sel.loc[best, "avg_inventory"] / sel.loc[base, "avg_inventory"] - 1) * 100),
        delta_color="inverse",
    )
    curves = res["policy_curves"]
    curves = curves[curves["period"] == "test"].copy()
    curves["Policy"] = curves["policy"].map(POLICY_LABELS)
    fig = px.line(
        curves.sort_values("fill_rate"), x="fill_rate", y="avg_inventory", color="Policy",
        markers=True, color_discrete_map={POLICY_LABELS[k]: v for k, v in POLICY_COLORS.items()},
        labels={"fill_rate": "Fill rate", "avg_inventory": "Average units on hand per series"},
        title="Test period: inventory needed for each fill rate (lower-right is better)",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(res["bootstrap"].round(2), width="stretch", hide_index=True)

with tabs[1]:
    acc = res["forecast_accuracy"].copy()
    st.dataframe(acc.round(3), width="stretch", hide_index=True)
    st.dataframe(res["quantile_accuracy"].round(3), width="stretch", hide_index=True)

with tabs[2]:
    rec = res["recommendations"]
    levels = sorted(int(c.removeprefix("order_up_to_q")) for c in rec.columns if c.startswith("order_up_to_q"))
    a, b = st.columns(2)
    store = a.selectbox("Store", sorted(rec["store_id"].unique()))
    items = sorted(rec.loc[rec["store_id"] == store, "item_id"])
    item = b.selectbox("Item", items)
    c, d, e = st.columns(3)
    q = c.selectbox("Service level", levels, index=min(2, len(levels) - 1), format_func=lambda v: f"{v}%")
    on_hand = d.number_input("On hand", min_value=0.0, value=0.0)
    on_order = e.number_input("On order", min_value=0.0, value=0.0)
    r = rec[(rec["store_id"] == store) & (rec["item_id"] == item)].iloc[0]
    target = float(r[f"order_up_to_q{q}"])
    st.metric("Recommended order (units)", max(0, math.ceil(target - on_hand - on_order)))
    st.write(
        f"Order up to {target:.0f} units to cover the {int(r['covers_days'])} days from "
        f"{r['covers_from']}. Expected demand {r['forecast_mean']:.1f}. Class {r['abc']}, {r['pattern']}."
    )
    sp = res["sample_predictions"]
    sp = sp[sp["series_id"] == r["series_id"]]
    if not sp.empty:
        st.plotly_chart(
            px.line(sp, x="date", y=["demand_next", "forecast_mean", "q90"],
                    title="Held-out test period: 3-day demand vs forecast"),
            width="stretch",
        )

with tabs[3]:
    seg = res["segment_results"]
    seg = seg[seg["policy"].isin(["ma_z", "lgbm_qr"])]
    st.write("Cost of each policy relative to the baseline, by ABC class and demand pattern (%).")
    st.dataframe(seg.round(3), width="stretch", hide_index=True)
    q = seg[seg["policy"] == "lgbm_qr"].assign(
        segment=lambda d: d["abc"] + " / " + d["pattern"]
    )
    if not q.empty:
        st.plotly_chart(
            px.bar(q, x="segment", y="cost_vs_baseline_pct", title="LightGBM quantile: cost vs baseline (%)"),
            width="stretch",
        )

with tabs[4]:
    st.write("Change the cost assumptions; service levels are re-chosen on validation data.")
    cfg = CostConfig(**meta["config"]["cost"])
    a, b, c = st.columns(3)
    holding = a.slider("Holding cost, % per year", 5, 60, int(cfg.holding_rate * 100)) / 100
    margin = b.slider("Gross margin, %", 10, 50, int(cfg.margin * 100)) / 100
    mult = c.slider("Stock-out penalty multiple", 0.5, 5.0, float(cfg.stockout_mult), 0.5)
    out = evaluate_scenario(
        res["physical_totals"],
        replace(cfg, holding_rate=holding, margin=margin, stockout_mult=mult),
    )
    out["policy"] = out["policy"].map(POLICY_LABELS)
    st.dataframe(out.round(3), width="stretch", hide_index=True)
    st.caption("Shelf life changes the physics and needs a re-run; see the sensitivity table.")
    st.dataframe(res["sensitivity"].round(2), width="stretch", hide_index=True)
