"""Gerador de série sintética de custo AWS.

Não usa nenhum dado real de cliente ou de empresa. O objetivo é produzir uma
série diária que se pareça, em forma, com billing real de uma organização
enterprise multi-conta, combinando os padrões que aparecem na prática de
FinOps:

- tendência de crescimento (onboarding de novos serviços/contas);
- sazonalidade semanal (cargas batch/dev desligadas nos fins de semana);
- sazonalidade mensal (picos no fechamento de mês);
- efeito de Savings Plans / RI (degraus de redução após pontos de compra);
- anomalias pontuais (spikes, ex.: instância grande esquecida ligada);
- ruído com caudas mais pesadas que o gaussiano (distribuição t de Student),
  já que custo de nuvem real tem mais eventos extremos do que um ruído
  gaussiano puro sugeriria.
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
    """Gera uma série diária sintética de custo AWS.

    Args:
        n_days: quantidade de dias a gerar.
        start_date: data inicial da série (formato ISO ``YYYY-MM-DD``).
        base_daily_cost: custo diário de referência no início da série.
        seed: semente do gerador aleatório, para reprodutibilidade.

    Returns:
        DataFrame com colunas ``date`` (Timestamp), ``cost`` (float32) e
        ``is_anomaly`` (bool, marca os dias com spike injetado).
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
