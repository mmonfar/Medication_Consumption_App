"""Tests for aggregation helpers and the structure of the generated data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medication_app.config import MEDICATIONS, PROFILES
from medication_app.data import load_sample_data
from medication_app.forecasting import (
    comorbidity_summary,
    daily_counts,
    daily_dose_mg,
    forecast_dates,
    is_intermittent,
    moving_average,
    zero_share,
)


@pytest.fixture(scope="module")
def sample():
    return load_sample_data()


def test_sample_data_is_deterministic():
    a_patients, a_consumption, a_cohort = load_sample_data(seed=7, history_days=120)
    b_patients, b_consumption, b_cohort = load_sample_data(seed=7, history_days=120)
    pd.testing.assert_frame_equal(a_patients, b_patients)
    pd.testing.assert_frame_equal(a_consumption, b_consumption)
    pd.testing.assert_frame_equal(a_cohort, b_cohort)


def test_consumption_dates_are_midnight(sample):
    _, consumption, _ = sample
    assert (consumption["Date"].dt.normalize() == consumption["Date"]).all()


def test_cohort_comorbidity_varies_over_time(sample):
    """Must vary, or the elasticity is unidentifiable and the slider is a lie."""
    _, _, cohort = sample
    assert cohort["Mean_Charlson"].std() > 0.05


def test_daily_counts_cover_every_day_including_zeros(sample):
    _, consumption, _ = sample
    counts = daily_counts(consumption, "Meropenem 1g")
    expected = (consumption["Date"].max() - consumption["Date"].min()).days + 1
    assert len(counts) == expected
    assert counts["ds"].is_monotonic_increasing
    assert (counts["y"] >= 0).all()
    assert (counts["y"] == 0).any(), "restricted antibiotic should have zero-days"


def test_daily_counts_are_integers(sample):
    _, consumption, _ = sample
    counts = daily_counts(consumption, MEDICATIONS[0])
    assert np.allclose(counts["y"], counts["y"].round())


def test_weekend_effect_is_present(sample):
    """The generator encodes a weekday/weekend split; aggregation must keep it."""
    _, consumption, _ = sample
    counts = daily_counts(consumption, "Omeprazole 20 mg").set_index("ds")["y"]
    weekend = counts[counts.index.dayofweek >= 5].mean()
    weekday = counts[counts.index.dayofweek < 5].mean()
    assert weekend < weekday


def test_intermittency_classification_matches_profiles(sample):
    _, consumption, _ = sample
    for medication in MEDICATIONS:
        counts = daily_counts(consumption, medication)
        assert is_intermittent(counts["y"]) == PROFILES[medication].intermittent


def test_zero_share_bounds(sample):
    _, consumption, _ = sample
    share = zero_share(daily_counts(consumption, "Meropenem 1g")["y"])
    assert 0.0 <= share <= 1.0


def test_dose_mg_is_separate_from_counts(sample):
    """Milligram totals are display-only; forecasting uses administration counts."""
    _, consumption, _ = sample
    counts = daily_counts(consumption, "Ciprofloxacin 500 mg")
    mg = daily_dose_mg(consumption, "Ciprofloxacin 500 mg")
    assert np.allclose(mg["mg"], counts["y"] * 500)


def test_moving_average_matches_window():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert moving_average(series, 2).tolist()[1:] == [1.5, 2.5, 3.5]


def test_moving_average_no_longer_returns_all_nan():
    """A window longer than the history used to blank the trace entirely."""
    assert not moving_average(pd.Series([1.0, 2.0, 3.0]), 30).isna().any()


def test_forecast_dates_follow_last_observation(sample):
    _, consumption, _ = sample
    counts = daily_counts(consumption, MEDICATIONS[0])
    dates = forecast_dates(counts, 5)
    assert len(dates) == 5
    assert dates[0] > counts["ds"].max()
    assert (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_comorbidity_summary_metrics(sample):
    patients, _, _ = sample
    scores = patients["Charlson_Comorbidity_Score"]
    assert comorbidity_summary(patients, "mean") == pytest.approx(scores.mean())
    assert comorbidity_summary(patients, "median") == pytest.approx(scores.median())
