"""Tests for aggregation and forecasting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medication_app.config import MEDICATIONS
from medication_app.data import load_sample_data
from medication_app.forecasting import (
    comorbidity_summary,
    daily_totals,
    forecast_dates,
    moving_average,
    placeholder_forecast,
)


@pytest.fixture(scope="module")
def sample():
    return load_sample_data()


def test_sample_data_is_deterministic():
    first_patients, first_consumption = load_sample_data(seed=7)
    second_patients, second_consumption = load_sample_data(seed=7)
    pd.testing.assert_frame_equal(first_patients, second_patients)
    pd.testing.assert_frame_equal(first_consumption, second_consumption)


def test_consumption_dates_are_midnight(sample):
    _, consumption = sample
    assert (consumption["Date"].dt.normalize() == consumption["Date"]).all()


def test_daily_totals_are_unique_and_sorted(sample):
    _, consumption = sample
    daily = daily_totals(consumption, MEDICATIONS[0])
    assert daily["Date"].is_unique
    assert daily["Date"].is_monotonic_increasing


def test_moving_average_matches_window():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert moving_average(series, 2).tolist()[1:] == [1.5, 2.5, 3.5]


def test_moving_average_is_all_nan_when_window_exceeds_history():
    """A 30-day window over a 30-day history is selectable in the UI."""
    series = pd.Series([1.0, 2.0, 3.0])
    assert moving_average(series, 30).isna().all()


def test_forecast_dates_follow_last_observation(sample):
    _, consumption = sample
    daily = daily_totals(consumption, MEDICATIONS[0])
    dates = forecast_dates(daily, 5)
    assert len(dates) == 5
    assert dates[0] > daily["Date"].max()
    assert (pd.Series(dates).diff().dropna() == pd.Timedelta(days=1)).all()


def test_comorbidity_summary_metrics(sample):
    patients, _ = sample
    scores = patients["Charlson_Comorbidity_Score"]
    assert comorbidity_summary(patients, "mean") == pytest.approx(scores.mean())
    assert comorbidity_summary(patients, "median") == pytest.approx(scores.median())


def test_placeholder_forecast_shape_and_scaling(sample):
    _, consumption = sample
    daily = daily_totals(consumption, MEDICATIONS[0])
    at_five = placeholder_forecast(daily, 7, 5.0, rng=np.random.default_rng(0))
    doubled = placeholder_forecast(daily, 7, 10.0, rng=np.random.default_rng(0))

    assert at_five.shape == (7,)
    assert (at_five >= 0).all()
    np.testing.assert_allclose(doubled, at_five * 2)


def test_placeholder_forecast_is_not_reproducible_without_a_seed(sample):
    """Documents the defect: the 'forecast' re-rolls on every callback."""
    _, consumption = sample
    daily = daily_totals(consumption, MEDICATIONS[0])
    assert not np.allclose(
        placeholder_forecast(daily, 7, 5.0), placeholder_forecast(daily, 7, 5.0)
    )
