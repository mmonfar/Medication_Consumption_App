"""Tests for metrics, model selection safety, and the comorbidity elasticity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medication_app.backtest import (
    MAX_UNDER_SUPPLY,
    cumulative_bias,
    eligible,
    mase,
    rmsse,
    select_model,
)
from medication_app.config import PROFILES
from medication_app.data import load_sample_data
from medication_app.forecasting import daily_counts
from medication_app.models import Elasticity, fit_elasticity


@pytest.fixture(scope="module")
def sample():
    return load_sample_data()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_mase_of_perfect_forecast_is_zero():
    actual = np.array([1.0, 2.0, 3.0])
    insample = np.array([1.0, 3.0, 2.0, 4.0])
    assert mase(actual, actual, insample) == 0.0


def test_rmsse_penalises_large_misses_more_than_mase():
    """Squared error is the reason intermittent series are scored on RMSSE."""
    actual = np.array([0.0, 0.0, 0.0, 10.0])
    flat_zero = np.zeros(4)
    insample = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    assert rmsse(actual, flat_zero, insample) > mase(actual, flat_zero, insample)


def test_cumulative_bias_sign_distinguishes_over_and_under_supply():
    actual = np.array([2.0, 2.0])
    assert cumulative_bias(actual, np.array([3.0, 3.0])) > 0  # overstock
    assert cumulative_bias(actual, np.array([1.0, 1.0])) < 0  # stockout risk
    assert np.isnan(cumulative_bias(np.zeros(2), np.zeros(2)))


# --------------------------------------------------------------------------- #
# Selection safety
# --------------------------------------------------------------------------- #


def _scores() -> pd.DataFrame:
    """A leaderboard shaped like the real Meropenem result."""
    return pd.DataFrame(
        [
            {"model": "Naive", "primary": 0.30, "cum_bias": -1.0, "is_benchmark": True},
            {"model": "CrostonSBA", "primary": 0.44, "cum_bias": 0.32, "is_benchmark": False},
            {"model": "SeasonalNaive", "primary": 0.66, "cum_bias": -0.09, "is_benchmark": True},
        ]
    )


def test_systematically_under_supplying_model_is_disqualified():
    """Forecasting zero doses for a whole horizon must not be selectable."""
    assert "Naive" not in set(eligible(_scores())["model"])


def test_selection_prefers_croston_over_a_stockout_forecast():
    name, beats = select_model(_scores())
    assert name == "CrostonSBA"
    assert beats


def test_selection_falls_back_to_benchmark_when_nothing_beats_it():
    scores = _scores()
    scores.loc[scores["model"] == "CrostonSBA", "primary"] = 0.99
    name, beats = select_model(scores)
    assert name == "SeasonalNaive"
    assert not beats


def test_guard_threshold_is_a_real_bound():
    assert 0.0 < MAX_UNDER_SUPPLY < 1.0


# --------------------------------------------------------------------------- #
# Comorbidity elasticity
# --------------------------------------------------------------------------- #


def test_elasticity_recovers_the_generator_parameter(sample):
    """Seasonal controls must de-confound the winter antibiotic surge."""
    _, consumption, cohort = sample
    series = daily_counts(consumption, "Ciprofloxacin 500 mg")
    fitted = fit_elasticity(series, cohort)
    true_beta = PROFILES["Ciprofloxacin 500 mg"].comorbidity_beta
    assert fitted.beta == pytest.approx(true_beta, abs=3 * fitted.std_error)


def test_unusable_elasticity_leaves_the_forecast_untouched():
    unusable = Elasticity(beta=2.0, std_error=5.0, r_squared=0.0, n=50)
    forecast = np.array([1.0, 2.0])
    np.testing.assert_allclose(unusable.adjust(forecast, 8.0, 4.0), forecast)


def test_usable_elasticity_scales_the_forecast():
    usable = Elasticity(beta=1.0, std_error=0.05, r_squared=0.5, n=50)
    np.testing.assert_allclose(usable.adjust(np.array([10.0]), 8.0, 4.0), [20.0])


def test_autocorrelated_residuals_mark_an_elasticity_suspect():
    spurious = Elasticity(
        beta=1.0, std_error=0.05, r_squared=0.95, n=50, residual_autocorrelation=0.9
    )
    assert spurious.is_suspect
    assert not spurious.is_usable


# --------------------------------------------------------------------------- #
# Forecast combination
# --------------------------------------------------------------------------- #


def test_combination_competes_as_its_own_candidate(sample):
    """Combining is a claim to be tested, not an assumption (Bates & Granger, 1969)."""
    from medication_app.backtest import cross_validate
    from medication_app.models import COMBINATION

    _, consumption, _ = sample
    scores = cross_validate(daily_counts(consumption, "Ciprofloxacin 500 mg"))
    assert COMBINATION in set(scores["model"])


def test_combination_is_not_treated_as_a_benchmark(sample):
    from medication_app.backtest import cross_validate
    from medication_app.models import COMBINATION

    _, consumption, _ = sample
    scores = cross_validate(daily_counts(consumption, "Ciprofloxacin 500 mg"))
    row = scores[scores["model"] == COMBINATION].iloc[0]
    assert not row["is_benchmark"]


def test_component_models_exclude_benchmarks(sample):
    from medication_app.models import BENCHMARKS, component_models

    _, consumption, _ = sample
    series = daily_counts(consumption, "Omeprazole 20 mg")
    names = {type(m).__name__ for m in component_models(series)}
    assert names and not (names & set(BENCHMARKS))
