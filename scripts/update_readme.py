#!/usr/bin/env python
"""Fill the results block of README.md from results/ (no number is typed by hand).

    python scripts/update_readme.py --results-dir results --print    # preview
    python scripts/update_readme.py --results-dir results            # write README.md

The block between <!-- RESULTS:START --> and <!-- RESULTS:END --> is replaced. Simulated runs
get a loud SIMULATED heading and are refused unless --allow-simulated is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LABELS = {
    "ma_z": "Moving average + safety stock (baseline)",
    "lgbm_nb": "LightGBM mean + NegBin safety stock",
    "lgbm_qr": "LightGBM quantile",
    "lgbm_qr_abc": "LightGBM quantile, level per ABC class",
}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build(res: Path) -> str:
    meta = json.loads((res / "meta.json").read_text())
    summ = json.loads((res / "summary.json").read_text())
    sel = pd.read_csv(res / "policy_selected.csv").set_index("policy")
    fa = pd.read_csv(res / "forecast_accuracy.csv")
    fr = pd.read_csv(res / "frontier.csv")
    sens = pd.read_csv(res / "sensitivity.csv")
    sim = summ["dataset"] == "simulated"

    out = []
    if sim:
        out.append("## Results: SIMULATED DATA (demo only, not real sales)\n")
    else:
        out.append("## Results: M5 Walmart FOODS (real data, US supermarkets)\n")
    f = meta["series_filter"]
    out.append(
        f"{summ['n_series']:,} store-item series, {summ['n_days']:,} days. Validation "
        f"{summ['validation_period'][0]} to {summ['validation_period'][1]}; test (reported) "
        f"{summ['test_period'][0]} to {summ['test_period'][1]}. {f['n_dropped_not_listed']:,} series "
        f"not listed throughout and {f['n_dropped_low_demand']:,} with demand below "
        f"{meta['config']['min_mean_demand']} units/day were excluded. Lead time "
        f"{summ['lead_time_days']} days, order every day. Service levels chosen on validation only.\n"
    )

    t = sel.reset_index()
    t["Policy"] = t["policy"].map(LABELS)
    ci = lambda r, m: f"{r[m]:+.1f} [{r[m + '_lo']:+.1f}, {r[m + '_hi']:+.1f}]"  # noqa: E731
    rows = []
    for _, r in t.iterrows():
        base = r["policy"] == "ma_z"
        rows.append(
            {
                "Policy": r["Policy"],
                "Level": r["chosen_q"],
                "Fill rate": f"{r['fill_rate']:.1%}",
                "Avg units on hand": f"{r['avg_inventory']:.2f}",
                "Waste": f"{r['waste_rate']:.1%}",
                "Cost % revenue": f"{r['cost_pct_revenue']:.1%}",
                "Cost reduction % (95% CI)": "-" if base else ci(r, "cost_reduction_pct"),
                "Fill rate vs baseline, pp": "-" if base else ci(r, "fill_rate_diff_pp"),
            }
        )
    out.append("### Policies on the test period\n")
    out.append(md(pd.DataFrame(rows)) + "\n")
    out.append(
        "Cost reduction is positive when the policy is cheaper than the baseline. Intervals "
        "resample series (1000 draws).\n"
    )

    out.append("### Inventory and cost needed for the same fill rate\n")
    fr = fr.copy()
    fr["Policy"] = fr["policy"].map(LABELS)
    fr["Fill"] = (fr["fill_target"] * 100).round().astype(int).astype(str) + "%"
    fr = fr[fr["policy"] != "ma_z"][
        ["Fill", "Policy", "inventory_vs_baseline_pct", "cost_vs_baseline_pct"]
    ].round(1)
    fr.columns = ["Fill rate", "Policy", "Inventory vs baseline %", "Cost vs baseline %"]
    out.append(md(fr.fillna("n/a")) + "\n\nn/a: that fill rate is outside the policy's tested range.\n")

    out.append("### Forecast accuracy (3-day demand, test period)\n")
    fa = fa.copy()
    fa["method"] = fa["method"].map(
        {"snaive": "Seasonal naive", "sn4": "Seasonal avg 4 wk", "ma28": "Moving avg 28d",
         "sba": "Croston-SBA", "lgbm": "LightGBM"}
    )
    out.append(md(fa[["method", "wape", "bias", "rmsse"]].round(3)) + "\n")

    out.append("### Sensitivity to the cost assumptions (cost change vs baseline, %)\n")
    s = sens[sens["policy"] != "ma_z"].copy()
    s["policy"] = s["policy"].map(LABELS)
    out.append(md(s.pivot_table(index="scenario", columns="policy", values="cost_vs_baseline_pct",
                                sort=False).round(1).reset_index()) + "\n")
    out.append(
        f"Runtime {summ['runtime_seconds']:.0f}s. Costs are assumptions "
        f"(margin {meta['config']['cost']['margin']:.0%}, holding {meta['config']['cost']['holding_rate']:.0%}/yr, "
        f"shelf life {meta['config']['cost']['shelf_life_days']} d, stock-out penalty "
        f"{meta['config']['cost']['stockout_mult']}x lost margin); read the relative results."
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--allow-simulated", action="store_true")
    a = ap.parse_args()
    res = Path(a.results_dir)
    block = build(res)
    if "SIMULATED" in block.splitlines()[0] and not a.allow_simulated and not a.print:
        sys.exit("refusing to write simulated results into the README (--allow-simulated overrides)")
    if a.print:
        print(block)
        return
    readme = Path(a.readme)
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit("README.md has no RESULTS markers")
    new = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}",
                 text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    print(f"README updated from {res}")


if __name__ == "__main__":
    main()
