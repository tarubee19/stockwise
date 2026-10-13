"""Static figures for the README, drawn from the CSV files in ``results/``.

One hue per entity across figures (a policy keeps its colour everywhere), thin marks, a recessive
grid, and a red banner on every figure when the data is simulated. Every figure has a CSV twin.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import FORECAST_LABELS, POLICY_COLORS, POLICY_LABELS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
COST_COLORS = {"holding_cost": "#2a78d6", "waste_cost": "#eb6834", "lost_margin_cost": "#1baf7a"}
COST_LABELS = {
    "holding_cost": "Holding cost",
    "waste_cost": "Waste (expired stock)",
    "lost_margin_cost": "Lost margin (stock-outs)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def _title(ax, title: str, subtitle: str | None, label: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", fontsize=9, color=INK_2, va="bottom")
    if label == "simulated":
        ax.annotate("SIMULATED DATA - pipeline validation only, not retail performance",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 46 if subtitle else 30),
                    textcoords="offset points", fontsize=8, color=CRITICAL, va="bottom")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_frontier(res: Path, label: str) -> None:
    curves = pd.read_csv(res / "policy_curves.csv")
    test = curves[curves["period"] == "test"]
    sel = pd.read_csv(res / "policy_selected.csv").set_index("policy")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for policy in ("ma_z", "lgbm_nb", "lgbm_qr"):
        g = test[test.policy == policy].sort_values("fill_rate")
        ax.plot(g.fill_rate * 100, g.avg_inventory, color=POLICY_COLORS[policy], linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, label=POLICY_LABELS[policy])
        if policy in sel.index:
            ax.scatter([sel.loc[policy, "fill_rate"] * 100], [sel.loc[policy, "avg_inventory"]],
                       s=110, facecolor="none", edgecolor=POLICY_COLORS[policy], linewidth=1.8, zorder=5)
    if "lgbm_qr_abc" in sel.index:
        ax.scatter([sel.loc["lgbm_qr_abc", "fill_rate"] * 100], [sel.loc["lgbm_qr_abc", "avg_inventory"]],
                   marker="D", s=46, color=POLICY_COLORS["lgbm_qr_abc"], edgecolor=SURFACE,
                   zorder=6, label=POLICY_LABELS["lgbm_qr_abc"])
    ax.set_xlabel("Fill rate: share of demand served (%)  -  further right is better")
    ax.set_ylabel("Average stock per store-item (units)  -  lower is better")
    ax.legend(loc="upper left", fontsize=8.5)
    _title(ax, "Same service with less stock: the inventory-service frontier",
           "Test period; each dot is one service level; rings mark the level chosen on the validation period", label)
    _save(fig, res / "figures" / "fig_frontier.png")


def fig_cost_breakdown(res: Path, label: str) -> None:
    sel = pd.read_csv(res / "policy_selected.csv").set_index("policy")
    order = [p for p in ("ma_z", "lgbm_nb", "lgbm_qr", "lgbm_qr_abc") if p in sel.index]
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    revenue = sel.loc[order, "revenue"].to_numpy()
    for comp in ("holding_cost", "waste_cost", "lost_margin_cost"):
        v = sel.loc[order, comp].to_numpy() / revenue * 100
        ax.barh(y, v, left=left, height=0.55, color=COST_COLORS[comp], edgecolor=SURFACE,
                linewidth=2, label=COST_LABELS[comp])
        left += v
    for yi, tot in zip(y, left, strict=True):
        ax.text(tot + 0.02 * max(left), yi, f"{tot:.2f}%", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, [POLICY_LABELS[p] for p in order])
    ax.set_xlabel("Total inventory cost, % of revenue  -  lower is better")
    ax.set_xlim(0, max(left) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    _title(ax, "Where the cost goes, at each policy's chosen service level",
           "Test period; costs use the assumptions listed in the README", label)
    _save(fig, res / "figures" / "fig_cost_breakdown.png")


def fig_forecast_accuracy(res: Path, label: str) -> None:
    acc = pd.read_csv(res / "forecast_accuracy.csv").sort_values("wape", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = [POLICY_COLORS["lgbm_qr"] if m == "lgbm" else "#a8a29e" for m in acc["method"]]
    bars = ax.barh([FORECAST_LABELS.get(m, m) for m in acc["method"]], acc["wape"] * 100,
                   color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, acc["wape"] * 100, strict=True):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("WAPE of the demand forecast over the lead-time window, % of demand  -  lower is better")
    ax.set_xlim(0, float(acc["wape"].max() * 100) * 1.15)
    ax.grid(axis="y", visible=False)
    _title(ax, "Demand forecast accuracy", "Test period, every store-item and origin day", label)
    _save(fig, res / "figures" / "fig_forecast_accuracy.png")


def fig_segments(res: Path, label: str) -> None:
    seg = pd.read_csv(res / "segment_results.csv")
    s = seg[seg.policy == "lgbm_qr_abc"]
    if s.empty:
        return
    abc = ["A", "B", "C"]
    patterns = [p for p in ("smooth", "erratic", "intermittent", "lumpy") if p in set(s.pattern)]
    cost = s.pivot(index="abc", columns="pattern", values="cost_vs_baseline_pct").reindex(index=abc, columns=patterns)
    count = s.pivot(index="abc", columns="pattern", values="n_series").reindex(index=abc, columns=patterns)
    span = max(float(np.nanmax(np.abs(cost.to_numpy()))), 1.0)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#e8e7e1", "#d03b3b"])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.imshow(cost.to_numpy(), cmap=cmap, vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(range(len(patterns)), [p.capitalize() for p in patterns])
    ax.set_yticks(range(len(abc)), [f"Class {c}" for c in abc])
    ax.grid(False)
    for i in range(len(abc)):
        for j in range(len(patterns)):
            v, n = cost.iloc[i, j], count.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}%\nn={int(n)}", ha="center", va="center", fontsize=8.5, color=INK)
    _title(ax, "Cost change vs the moving-average baseline, by ABC class and demand pattern",
           "Blue = cheaper than the baseline; n = store-items in the cell", label)
    _save(fig, res / "figures" / "fig_segments.png")


def fig_sample_forecast(res: Path, label: str) -> None:
    path = res / "sample_predictions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sid = df.groupby("series_id")["demand_next"].sum().idxmax()
    g = df[df.series_id == sid].copy()
    g["date"] = pd.to_datetime(g["date"])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.fill_between(g["date"], g["q70"], g["q95"], color=POLICY_COLORS["lgbm_qr"], alpha=0.16,
                    linewidth=0, label="70%-95% order-up-to band")
    ax.plot(g["date"], g["demand_next"], color=INK_2, linewidth=1.2, label="Actual demand")
    ax.plot(g["date"], g["forecast_mean"], color=POLICY_COLORS["lgbm_qr"], linewidth=1.8, label="LightGBM forecast")
    ax.set_ylabel("Units over the lead-time window")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    _title(ax, f"One high-volume store-item: {sid.replace('|', ' / ')}",
           "Demand over the next lead-time window versus the forecast and the stock level that covers it", label)
    _save(fig, res / "figures" / "fig_sample_forecast.png")


def make_figures(res: Path) -> None:
    _style()
    label = json.loads((res / "summary.json").read_text()).get("dataset", "real")
    for fn in (fig_frontier, fig_cost_breakdown, fig_forecast_accuracy, fig_segments, fig_sample_forecast):
        fn(res, label)

