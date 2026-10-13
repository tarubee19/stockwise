# StockWise: demand forecasting and ordering policy for perishable grocery

[![CI](https://github.com/tarubee19/stockwise/actions/workflows/ci.yml/badge.svg)](https://github.com/tarubee19/stockwise/actions)

How much should a store order each day so that shelves are full but little is thrown away?
StockWise forecasts demand over the order-to-arrival window, turns the forecast into an
order-up-to level, and **tests the ordering policy in a day-by-day inventory simulation on real
sales**. It reports fill rate, stock held, waste and cost, with confidence intervals.

**Data:** the M5 Forecasting competition (Walmart US, FOODS category: store-level daily sales,
weekly prices, calendar and SNAP days). This is US supermarket data, **not Indian quick-commerce
data**. M5 has no stock levels or costs, so costs are stated assumptions and sales are censored
by stock-outs. The reliable findings are the relative ones: how policies rank and how much stock
each needs for the same service.

## Method in short

1. **Series:** store-item pairs listed for the whole history and selling at least 0.05 units/day.
   Exclusions are counted and disclosed in the results.
2. **Target:** demand over the protection interval P = lead time + review period (default 2 + 1 = 3
   days), because an order placed today must cover that window.
3. **Models (global LightGBM, one model for all series):** a Tweedie mean model and quantile models
   (70/80/90/95/98/99%). Features use only information up to the order day, plus known future price
   and calendar. Walk-forward: models are refit with data only up to each evaluation period.
4. **Policies:** baseline = 28-day moving average + normal safety stock; LightGBM mean + negative
   binomial safety stock; LightGBM quantile; quantile with a service level per ABC class.
5. **Simulation:** daily arrivals, FIFO sales, shelf-life waste, holding cost, daily orders to the
   order-up-to level. Service levels are chosen on a validation period; **all results are on a
   later test period**. Series-resampling bootstrap gives 95% intervals.
6. **Checks:** forecast accuracy (WAPE, bias, RMSSE, pinball, coverage), ABC x demand-pattern
   segments, sensitivity to cost assumptions and shelf life.

Details and limitations: [docs/methodology.md](docs/methodology.md).

<!-- RESULTS:START -->
## Results: M5 Walmart FOODS (real data, US supermarkets)

12,635 store-item series, 1,941 days. Validation 2015-04-26 to 2015-08-23; test (reported) 2015-08-24 to 2016-05-19. 1,482 series not listed throughout and 253 with demand below 0.05 units/day were excluded. Lead time 2 days, order every day. Service levels chosen on validation only.

### Policies on the test period

| Policy | Level | Fill rate | Avg units on hand | Waste | Cost % revenue | Cost reduction % (95% CI) | Fill rate vs baseline, pp |
|---|---|---|---|---|---|---|---|
| LightGBM mean + NegBin safety stock | 0.90 | 92.6% | 4.36 | 6.9% | 10.2% | +14.8 [+14.0, +15.5] | +2.1 [+2.0, +2.2] |
| LightGBM quantile | 0.80 | 88.8% | 3.32 | 5.8% | 11.9% | +5.3 [+4.7, +5.9] | -1.7 [-1.8, -1.6] |
| Moving average + safety stock (baseline) | 0.90 | 90.5% | 4.13 | 7.6% | 12.3% | - | - |
| LightGBM quantile, level per ABC class | A:0.90 B:0.80 C:0.70 | 91.2% | 3.96 | 6.6% | 10.8% | +10.8 [+10.1, +11.4] | +0.7 [+0.6, +0.8] |

Cost reduction is positive when the policy is cheaper than the baseline. Intervals resample series (1000 draws).

### Inventory and cost needed for the same fill rate

| Fill rate | Policy | Inventory vs baseline % | Cost vs baseline % |
|---|---|---|---|
| 90% | LightGBM mean + NegBin safety stock | -8.5 | -12.8 |
| 90% | LightGBM quantile | -9.1 | -5.0 |
| 95% | LightGBM mean + NegBin safety stock | -7.1 | -17.6 |
| 95% | LightGBM quantile | -10.5 | -7.6 |
| 98% | LightGBM mean + NegBin safety stock | n/a | n/a |
| 98% | LightGBM quantile | n/a | n/a |

n/a: that fill rate is outside the policy's tested range.

### Forecast accuracy (3-day demand, test period)

| method | wape | bias | rmsse |
|---|---|---|---|
| Seasonal naive | 0.537 | -0.005 | 1.077 |
| Seasonal avg 4 wk | 0.481 | -0.009 | 0.904 |
| Moving avg 28d | 0.467 | -0.008 | 0.864 |
| Croston-SBA | 0.554 | 0.027 | 0.936 |
| LightGBM | 0.413 | -0.02 | 0.804 |

### Sensitivity to the cost assumptions (cost change vs baseline, %)

| scenario | LightGBM mean + NegBin safety stock | LightGBM quantile | LightGBM quantile, level per ABC class |
|---|---|---|---|
| base case | -14.8 | -5.3 | -10.8 |
| holding cost 10%/yr | -14.9 | -5.2 | -10.8 |
| holding cost 40%/yr | -14.6 | -5.3 | -10.7 |
| margin 15% | -16.1 | -5.3 | -9.7 |
| margin 40% | -12.6 | -7.4 | -8.6 |
| stock-out penalty = lost margin only | -15.5 | -5.1 | -8.8 |
| stock-out penalty x4 | -12.6 | -7.5 | -8.6 |
| shelf life 3 days | -9.4 | -2.7 | -2.7 |
| shelf life 14 days | -16.2 | -0.8 | -8.5 |
| shelf life 30 days | -28.8 | -8.0 | -18.3 |

Runtime 1479s. Costs are assumptions (margin 25%, holding 25%/yr, shelf life 7 d, stock-out penalty 2.0x lost margin); read the relative results.
<!-- RESULTS:END -->

## What the results tell a manager

Numbers below are read from `results/` (M5 FOODS, test period, assumed costs).

1. **Where the money goes.** With a 7-day shelf life, waste (about 55-60% of cost) and lost sales
   (about 40-45%) dominate; holding cost is about 1%. Cutting inventory per se is the wrong goal.
   The lever is ordering the right amount, and shelf life matters far more than the cost of capital
   (sensitivity: holding 10% to 40% per year barely moves the ranking; shelf life 3 to 30 days does).
2. **The gain is better allocation, not more stock.** The negative-binomial LightGBM policy has the
   same holding cost as the baseline, cuts waste cost by about 94k and lost-margin cost by about
   180k (price units), and serves 2.1 points more demand. It does this by sending stock where demand
   will be and away from where it will not.
3. **Calibration explains the ranking.** Targets of 90% and 95% give actual coverage of 91.6% and
   95.1% for the negative-binomial policy, but 94.2% and 97.0% for the direct quantile model (too
   conservative) and 91.1% and 93.4% for the moving average (too thin in the tail). A model is only
   as useful as its stated service level is true. The quantile model would need recalibration
   before its labels could be trusted.
4. **Policy cost is U-shaped in service level, and flat near the best.** For the negative-binomial
   policy, cost is lowest at the 90% target. Below it lost sales grow faster than waste falls; above it waste
   grows faster than lost sales fall. Pushing to 98-99% costs more than even the moving-average
   baseline at its best level.
5. **Focus on class A.** Class A items (top 80% of revenue) are about 55% of total cost, and the
   per-class policy cuts their cost by about 13%, against 6% for class B and 10% for class C.
   Smooth-demand items gain most (about 29% lower cost for the negative-binomial policy); the
   method adds least where demand is lumpy.
6. **What this does not show.** It is a simulation on US Walmart sales with assumed costs, not a
   pilot. Real savings depend on lead-time variability, supplier constraints, and censored sales
   (empty shelves hide demand).

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dashboard,dev]"

# 1. get the data (see docs/methodology.md): download M5 from Kaggle, then trim to FOODS
python scripts/prepare_m5_upload.py path/to/m5-forecasting-accuracy --out data/m5_foods.zip

# 2. run the study (writes results/ and figures)
python scripts/run_pipeline.py --data data/m5_foods.zip --results-dir results --n-jobs 8

# 3. put the numbers into this README, open the dashboard, call the API
python scripts/update_readme.py --results-dir results
streamlit run app/dashboard.py
uvicorn stockwise.api:app --port 8000
```

No data yet? `python scripts/make_synthetic.py --out data/synthetic` then run the pipeline on it.
Every output of such a run is labelled **SIMULATED** (figures, dashboard banner, API header
`X-Data-Label`); it exists to test the code, not to show performance.

## API

`GET /recommend-order?store=CA_1&item=FOODS_3_090&service_level=0.95&on_hand=10&on_order=5` returns
the forecast, the order-up-to level and `max(0, ceil(S - on_hand - on_order))`. Other endpoints:
`/health /meta /summary /policies /accuracy /segments /sensitivity`.

## Docker

```bash
docker compose up --build      # api :8000, dashboard :8501; mounts ./results
```
With an empty `results/` the container generates a SIMULATED demo so it still starts.

## Tests

`pytest -q` (28 tests): simulator invariants and hand-checked cases, feature leakage (changing
future sales leaves features unchanged), costs, policies, bootstrap, pipeline determinism, API,
dashboard. `ruff check .` is clean.

## Honest limitations

- M5 is US Walmart; findings do not transfer to Indian dark stores without re-running on local data.
- No stock data: sales understate demand when shelves were empty.
- Costs, shelf life and the stock-out penalty are assumptions (all configurable and shown in a
  sensitivity table).
- One lead time and one review period; no supplier minimums or multi-echelon effects.
