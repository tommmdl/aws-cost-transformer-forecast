"""Synthetic AWS cost time series generator.

Uses no real customer or company data. The goal is to produce a daily
series that resembles, in shape, real billing from a multi-account
enterprise organization, combining the patterns that show up in FinOps
practice:

- growth trend (onboarding of new services/accounts);
- weekly seasonality (batch/dev workloads turned off on weekends);
- monthly seasonality (spikes at month-end closing);
- Savings Plans / RI effect (step reductions after purchase points);
- point anomalies (spikes, e.g. a large instance left running by mistake);
- noise with heavier tails than Gaussian (Student's t distribution), since
  real cloud cost has more extreme events than pure Gaussian noise would
  suggest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_WEEKEND_FACTOR = 0.75
_MONTH_END_FACTOR = 1.15
_MONTH_END_WINDOW_DAYS = 3
_TREND_DAILY_RATE = 0.0006
_RI_STEP_POINT_FRACTIONS = (0.3, 0.65)
_RI_STEP_REDUCTION_FACTOR = 0.93
_NOISE_T_DEGREES_OF_FREEDOM = 4
_NOISE_SCALE = 0.04
_ANOMALY_PROBABILITY = 0.004
_ANOMALY_MAGNITUDE_RANGE = (3.0, 8.0)
_MIN_COST_FLOOR_FRACTION = 0.05


def generate_synthetic_aws_cost(
    n_days: int = 730,
    start_date: str = "2023-01-01",
    base_daily_cost: float = 5000.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generates a synthetic daily AWS cost series.

    Args:
        n_days: number of days to generate.
        start_date: series start date (ISO format ``YYYY-MM-DD``).
        base_daily_cost: reference daily cost at the start of the series.
        seed: random generator seed, for reproducibility.

    Returns:
        DataFrame with columns ``date`` (Timestamp), ``cost`` (float32) and
        ``is_anomaly`` (bool, flags days with an injected spike).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = base_daily_cost * (1.0 + _TREND_DAILY_RATE) ** t

    day_of_week = dates.dayofweek.to_numpy()
    weekly_factor = np.where(day_of_week >= 5, _WEEKEND_FACTOR, 1.0)

    day_of_month = dates.day.to_numpy()
    days_in_month = dates.days_in_month.to_numpy()
    is_month_end = day_of_month > (days_in_month - _MONTH_END_WINDOW_DAYS)
    monthly_factor = np.where(is_month_end, _MONTH_END_FACTOR, 1.0)

    cost = trend * weekly_factor * monthly_factor

    for fraction in _RI_STEP_POINT_FRACTIONS:
        step_index = int(n_days * fraction)
        cost[step_index:] *= _RI_STEP_REDUCTION_FACTOR

    noise = rng.standard_t(_NOISE_T_DEGREES_OF_FREEDOM, size=n_days) * _NOISE_SCALE
    cost = cost * (1.0 + noise)

    is_anomaly = rng.random(n_days) < _ANOMALY_PROBABILITY
    anomaly_magnitude = rng.uniform(*_ANOMALY_MAGNITUDE_RANGE, size=n_days)
    cost = np.where(is_anomaly, cost + base_daily_cost * anomaly_magnitude, cost)

    cost = np.clip(cost, base_daily_cost * _MIN_COST_FLOOR_FRACTION, None)

    return pd.DataFrame(
        {
            "date": dates,
            "cost": cost.astype(np.float32),
            "is_anomaly": is_anomaly,
        }
    )
