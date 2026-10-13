# Methodology

## Data
M5 Forecasting - Accuracy (Kaggle, Walmart US). Files: `calendar.csv`, `sell_prices.csv`,
`sales_train_evaluation.csv` (or `_validation`). `scripts/prepare_m5_upload.py` keeps the FOODS
category (about 14k store-item series) and zips three CSVs (about 25 MB).
Series must be listed on every day of the history window and average at least 0.05 units/day
before the validation period; counts are in `results/meta.json`.

## Timeline
Last `test_days` (270) feasible order days = test; the `val_days` (120) before = validation.
Pass A trains on origins that end before validation and predicts validation (used to choose service
levels). Pass B trains on origins that end before test and predicts test (reported). A final fit on
all data produces next-day recommendations.

## Forecast target
For order day t, y = demand on days t+1..t+P. P = lead time + review period. One direct model per
quantile, so there are no recursive multi-step forecasts.

## Features
Lags, rolling means (7-365 days), rolling std, coefficient of variation, share of zero days, days
since last sale, same-weekday averages, Croston-SBA level, price and relative price over the
window (price is known in advance), calendar (weekday, month, events, SNAP), store/department.
A unit test checks that altering sales after t cannot change features at t.

## Policies
`ma_z`: S = P x mean28 + z sigma sqrt(P). `lgbm_nb`: LightGBM mean, negative-binomial quantile with
dispersion fitted by moments per department x store. `lgbm_qr`: S = predicted quantile.
`lgbm_qr_abc`: quantile policy with the service level chosen per ABC class (80/15/5% revenue).

## Simulation
Order of events per day: arrivals, FIFO sales, waste of units at shelf life, overnight holding,
order of max(0, S - on hand - on order). Costs: unit cost = (1 - margin) x reference price; holding
= rate/365 x unit cost per unit-day; waste = unit cost per unit; stock-out = multiple x lost margin.
Physical flows are stored, money is applied afterwards, so cost sensitivity needs no re-simulation
(shelf life does).

## Statistics
Service level per policy = lowest total cost on validation. Intervals: 1000 bootstrap resamples of
series. Frontier: interpolation of each policy's cost/inventory curve at 90/95/98% fill.

## Limitations
See README. Costs are assumptions; sales are censored; US data.
