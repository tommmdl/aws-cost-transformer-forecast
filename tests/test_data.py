import numpy as np
import pandas as pd

from aws_cost_forecast.data.synthetic_aws_cost import generate_synthetic_aws_cost


def test_returns_dataframe_with_expected_columns_and_length():
    df = generate_synthetic_aws_cost(n_days=100, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["date", "cost", "is_anomaly"]
    assert len(df) == 100


def test_dates_are_sequential_starting_at_start_date():
    df = generate_synthetic_aws_cost(n_days=30, start_date="2023-01-01", seed=42)

    expected_dates = pd.date_range("2023-01-01", periods=30, freq="D")
    pd.testing.assert_series_equal(
        df["date"], pd.Series(expected_dates, name="date")
    )


def test_same_seed_is_deterministic():
    df1 = generate_synthetic_aws_cost(n_days=200, seed=7)
    df2 = generate_synthetic_aws_cost(n_days=200, seed=7)

    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_series():
    df1 = generate_synthetic_aws_cost(n_days=200, seed=1)
    df2 = generate_synthetic_aws_cost(n_days=200, seed=2)

    assert not df1["cost"].equals(df2["cost"])


def test_cost_is_always_positive():
    df = generate_synthetic_aws_cost(n_days=1000, seed=123)

    assert (df["cost"] > 0).all()


def test_weekend_cost_is_lower_on_average_than_weekday():
    df = generate_synthetic_aws_cost(n_days=2000, seed=42)

    is_weekend = df["date"].dt.dayofweek >= 5
    weekend_mean = df.loc[is_weekend, "cost"].mean()
    weekday_mean = df.loc[~is_weekend, "cost"].mean()

    assert weekend_mean < weekday_mean


def test_trend_makes_later_period_costlier_than_earlier_period():
    df = generate_synthetic_aws_cost(n_days=2000, seed=42)

    first_month_median = df["cost"].iloc[:30].median()
    last_month_median = df["cost"].iloc[-30:].median()

    assert last_month_median > first_month_median


def test_anomalies_are_flagged_and_are_upward_spikes():
    df = generate_synthetic_aws_cost(n_days=2000, seed=42)

    assert df["is_anomaly"].dtype == bool
    assert df["is_anomaly"].sum() > 0

    anomaly_cost = df.loc[df["is_anomaly"], "cost"].mean()
    normal_cost = df.loc[~df["is_anomaly"], "cost"].mean()
    assert anomaly_cost > normal_cost


def test_cost_dtype_is_float32():
    df = generate_synthetic_aws_cost(n_days=50, seed=42)

    assert df["cost"].dtype == np.float32
