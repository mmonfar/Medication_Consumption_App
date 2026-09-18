"""Tests for conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pytest

from medication_app.intervals import ConformalBands


def _bands(**overrides) -> ConformalBands:
    defaults = dict(
        levels=(80, 95),
        half_widths={80: [1.0, 1.5, 2.0], 95: [2.0, 3.0, 4.0]},
        cumulative_half_widths={80: [1.0, 1.8, 2.4], 95: [2.0, 3.4, 4.6]},
        coverage={80: 0.81, 95: 0.96},
        cumulative_coverage={80: 0.80, 95: 0.95},
        calibration_windows=90,
        coverage_windows=30,
    )
    defaults.update(overrides)
    return ConformalBands(**defaults)


def test_bands_are_clipped_at_zero():
    """Counts cannot be negative, so the lower band floors rather than going below zero."""
    lower, upper = _bands().apply(np.array([0.3, 0.3, 0.3]), 95)
    assert (lower >= 0).all()
    assert (upper > 0).all()


def test_wider_level_gives_wider_band():
    forecast = np.array([5.0, 5.0, 5.0])
    low80, high80 = _bands().apply(forecast, 80)
    low95, high95 = _bands().apply(forecast, 95)
    assert (high95 - low95 >= high80 - low80).all()


def test_half_widths_do_not_shrink_with_horizon():
    bands = _bands()
    for level in bands.levels:
        widths = bands.half_widths[level]
        assert widths == sorted(widths), "uncertainty must not decrease with horizon"


def test_horizon_beyond_calibration_holds_the_last_width():
    forecast = np.array([1.0] * 6)
    lower, upper = _bands().apply(forecast, 95)
    assert upper[-1] == pytest.approx(upper[2])


def test_cumulative_band_is_narrower_than_summing_daily_bands():
    """Summing daily bounds assumes perfectly correlated errors and over-widens.

    This is the defect the cumulative calibration exists to fix: on real data
    it turned a central estimate of 2.5 Meropenem administrations into an
    interval of 0 to 23.
    """
    bands = _bands()
    forecast = np.array([2.0, 2.0, 2.0])
    lower, upper = bands.apply(forecast, 95)
    summed_width = float(upper.sum() - lower.sum())
    cumulative_width = 2 * bands.cumulative_half_widths[95][2]
    assert cumulative_width < summed_width
