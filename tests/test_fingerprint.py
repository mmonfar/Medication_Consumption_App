"""Tests for artifact freshness: a cached model must not silently outlive its data."""

from __future__ import annotations

import pandas as pd
import pytest

from medication_app.fingerprint import (
    STALE_AFTER_DAYS,
    assess,
    config_fingerprint,
    data_extent,
)


@pytest.fixture
def consumption() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({"Date": dates, "Medication": "X", "Dose": 1.0})


def _artifact(consumption: pd.DataFrame, **overrides) -> dict:
    payload = {
        "config_fingerprint": config_fingerprint(),
        "data_extent": data_extent(consumption),
    }
    payload.update(overrides)
    return payload


def test_fingerprint_is_stable_across_calls():
    assert config_fingerprint() == config_fingerprint()


def test_matching_artifact_is_usable_and_current(consumption):
    freshness = assess(_artifact(consumption), consumption)
    assert freshness.is_usable
    assert not freshness.is_stale
    assert freshness.days_behind == 0


def test_configuration_mismatch_makes_the_artifact_unusable(consumption):
    """Different generator config means the models describe different data."""
    freshness = assess(_artifact(consumption, config_fingerprint="0" * 16), consumption)
    assert not freshness.is_usable
    assert "no longer describe" in freshness.describe()


def test_data_moving_on_is_measured_in_days_not_a_boolean(consumption):
    stale_extent = dict(data_extent(consumption), last_date="2026-01-10")
    freshness = assess(_artifact(consumption, data_extent=stale_extent), consumption)
    assert freshness.is_usable, "old is not the same as wrong"
    assert freshness.days_behind == 20
    assert freshness.is_stale


def test_a_day_behind_is_not_yet_stale(consumption):
    """The synthetic history is anchored to today, so a one-day lag is routine."""
    recent = dict(data_extent(consumption), last_date="2026-01-29")
    freshness = assess(_artifact(consumption, data_extent=recent), consumption)
    assert freshness.days_behind == 1
    assert not freshness.is_stale
    assert "1 day(s) behind" in freshness.describe()


def test_stale_threshold_is_a_positive_number_of_days():
    assert STALE_AFTER_DAYS > 0


def test_missing_provenance_is_treated_as_a_mismatch(consumption):
    freshness = assess({}, consumption)
    assert not freshness.is_usable
    assert freshness.artifact_last_date == "unknown"
