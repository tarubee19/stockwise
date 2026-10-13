"""Experiment configuration: every assumption of the study lives here and is saved in results/."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Policy keys, display names and the order they are reported in.
POLICY_LABELS: dict[str, str] = {
    "ma_z": "Moving average + safety stock",
    "lgbm_nb": "LightGBM mean + NegBin safety stock",
    "lgbm_qr": "LightGBM quantile",
    "lgbm_qr_abc": "LightGBM quantile, service level per ABC class",
}
BASELINE_POLICY = "ma_z"

# Mean-forecast methods scored for accuracy, in report order.
FORECAST_LABELS: dict[str, str] = {
    "snaive": "Seasonal naive (same weekdays last week)",
    "sn4": "Seasonal average (same weekdays, last 4 weeks)",
    "ma28": "Moving average (28 days)",
    "sba": "Croston-SBA (intermittent demand)",
    "lgbm": "LightGBM (Tweedie)",
}

QUANTILE_LABELS: dict[str, str] = {
    "ma_z": "Moving average + normal safety stock",
    "lgbm_nb": "LightGBM mean + negative binomial",
    "lgbm_qr": "LightGBM quantile regression",
}

# Colours for the three policy families (validated categorical palette, light surface).
POLICY_COLORS = {
    "ma_z": "#6b7a90",
    "lgbm_nb": "#d98a1c",
    "lgbm_qr": "#1f6fb2",
    "lgbm_qr_abc": "#2b9d7a",
}


@dataclass(frozen=True)
class CostConfig:
    """Cost assumptions. M5 has prices but no costs, so these are explicit assumptions."""

    margin: float = 0.25  # gross margin: unit cost = (1 - margin) * typical selling price
    holding_rate: float = 0.25  # annual cost of holding stock, as a share of unit cost
    shelf_life_days: int = 7  # units still unsold after this many days are discarded
    stockout_mult: float = 2.0  # penalty per lost unit = mult * (price - unit cost): the lost
    # margin plus an equal amount for customer goodwill (lost basket, substitution)
    warmup_days: int = 14  # simulation days excluded from every metric


@dataclass(frozen=True)
class ExperimentConfig:
    lead_time: int = 2  # days from order to arrival
    review_period: int = 1  # an order decision every day
    val_days: int = 120  # period used to choose service levels
    test_days: int = 270  # period used to report results
    min_history: int = 365  # days of history required before the validation period
    min_mean_demand: float = 0.05  # units/day in the history; sparser series are excluded
    train_window: int = 540  # days of history used to fit each model
    train_stride: int = 3  # use every 3rd origin day for training rows
    quantiles: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    n_jobs: int = 4
    seed: int = 42
    bootstrap_draws: int = 1000
    cost: CostConfig = field(default_factory=CostConfig)

    @property
    def protection(self) -> int:
        """Days an order-up-to level must cover: lead time + review period."""
        return self.lead_time + self.review_period

    def to_dict(self) -> dict:
        return asdict(self)


# Calendar days appended after the last sales day so future features (price, events, SNAP) exist.
CALENDAR_EXTRA_DAYS = 14
MAX_PROTECTION = 7
