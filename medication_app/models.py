"""Forecasting models and selection.

Method choice follows the vault's application guide (refs_books,
forecasting-principles-practice/APPLICATION-GUIDE.md sections 2 and 4):

* a naive/seasonal-naive/mean benchmark is always fitted first and nothing
  ships unless it beats the right benchmark on held-out data;
* intermittent series (many zero days) go to the Croston family, with SBA
  preferred for its bias correction;
* ETS/ARIMA are used only where history is ample -- two years of daily data
  covers the weekly cycle many times over and the annual cycle twice;
* the comorbidity slider is scenario-based forecasting, not a fitted
  coefficient applied blindly: the elasticity is estimated from history and
  reported with its own uncertainty.

Croston has no underlying stochastic model, so it yields no prediction
intervals. Callers must surface that rather than implying precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    CrostonClassic,
    CrostonSBA,
    HistoricAverage,
    Naive,
    SeasonalNaive,
)

from .config import COMORBIDITY_REFERENCE, WEEKLY_SEASONALITY
from .forecasting import is_intermittent

# Models that cannot produce prediction intervals (no stochastic model behind them).
NO_INTERVAL_MODELS = {"CrostonClassic", "CrostonSBA"}

BENCHMARKS = ("Naive", "SeasonalNaive", "HistoricAverage")


def benchmark_models() -> list:
    """The honest floor. Always fitted; never skipped."""
    return [
        Naive(),
        SeasonalNaive(season_length=WEEKLY_SEASONALITY),
        HistoricAverage(),
    ]


def candidate_models(series: pd.DataFrame) -> list:
    """Benchmarks plus the methods appropriate to this series' shape."""
    models = benchmark_models()
    if is_intermittent(series["y"]):
        models += [CrostonClassic(), CrostonSBA()]
    else:
        models += [
            AutoETS(season_length=WEEKLY_SEASONALITY),
            AutoARIMA(season_length=WEEKLY_SEASONALITY),
        ]
    return models


def _prepare(series: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    return series.assign(unique_id=unique_id)[["unique_id", "ds", "y"]]


def fit_predict(
    series: pd.DataFrame,
    horizon: int,
    models: list | None = None,
    unique_id: str = "series",
    level: list[int] | None = None,
) -> pd.DataFrame:
    """Fit ``models`` on ``series`` and forecast ``horizon`` days ahead."""
    models = models if models is not None else candidate_models(series)
    sf = StatsForecast(models=models, freq="D", n_jobs=1)
    forecast = sf.forecast(df=_prepare(series, unique_id), h=horizon, level=level)
    return forecast.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Comorbidity elasticity -- the slider, learned rather than assumed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Elasticity:
    """Fitted log-log relationship between cohort comorbidity and consumption."""

    beta: float
    std_error: float
    r_squared: float
    n: int
    residual_autocorrelation: float = 0.0

    @property
    def is_significant(self) -> bool:
        """Roughly a 95% t-test against beta = 0."""
        return self.std_error > 0 and abs(self.beta / self.std_error) > 1.96

    @property
    def is_suspect(self) -> bool:
        """Flag the spurious-regression signature: strong residual autocorrelation.

        Per fpppy/07-07-regression.md, a high R-squared together with heavily
        autocorrelated residuals is the classic sign that a regression is
        picking up shared drift rather than a real relationship.
        """
        return abs(self.residual_autocorrelation) > 0.5

    @property
    def is_usable(self) -> bool:
        return self.is_significant and not self.is_suspect

    def adjust(self, forecast: np.ndarray, predicted_score: float, observed_score: float) -> np.ndarray:
        """Scale a forecast to a hypothetical cohort comorbidity score.

        Returns the forecast unchanged when the elasticity is not statistically
        distinguishable from zero -- an unsupported adjustment is worse than no
        adjustment.
        """
        if not self.is_usable or observed_score <= 0 or predicted_score <= 0:
            return forecast
        return forecast * (predicted_score / observed_score) ** self.beta


def fit_elasticity(series: pd.DataFrame, cohort: pd.DataFrame) -> Elasticity:
    """Estimate d(log consumption)/d(log comorbidity) by OLS, net of seasonality.

    Both series are smoothed to weekly means first: daily Poisson noise and the
    weekday cycle would otherwise swamp a slow-moving cohort effect. Zero-count
    weeks are dropped because the log is undefined there.

    Annual Fourier terms are included as controls. Without them the estimate is
    badly confounded: winter raises antibiotic use *and* admits sicker patients,
    so a bare regression credits comorbidity with the entire winter surge. On
    this dataset that inflated the antibiotic elasticities roughly fourfold
    (Ciprofloxacin 2.32 against a true 0.55) -- a textbook omitted-variable
    problem that would have shipped as a confident, wrong slider.
    """
    merged = series.merge(cohort.rename(columns={"Date": "ds"}), on="ds", how="inner")
    weekly = (
        merged.set_index("ds")
        .resample("W")
        .agg({"y": "mean", "Mean_Charlson": "mean"})
        .dropna()
    )
    weekly = weekly[(weekly["y"] > 0) & (weekly["Mean_Charlson"] > 0)]
    if len(weekly) < 10:
        return Elasticity(0.0, 0.0, 0.0, len(weekly))

    x = np.log(weekly["Mean_Charlson"].to_numpy())
    y = np.log(weekly["y"].to_numpy())
    n = len(x)

    # Annual Fourier controls (one harmonic pair) to absorb the seasonal cycle.
    day_of_year = weekly.index.dayofyear.to_numpy()
    angle = 2 * np.pi * day_of_year / 365.25
    design = np.column_stack([np.ones(n), x, np.sin(angle), np.cos(angle)])

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta = coefficients[1]

    residuals = y - design @ coefficients
    dof = n - design.shape[1]
    if dof <= 0:
        return Elasticity(0.0, 0.0, 0.0, n)

    sigma_squared = float(residuals @ residuals) / dof
    covariance = sigma_squared * np.linalg.pinv(design.T @ design)
    std_error = float(np.sqrt(max(covariance[1, 1], 0.0)))

    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else 0.0

    # Lag-1 residual autocorrelation: the spurious-regression tell.
    if residuals.std() > 0:
        autocorrelation = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        autocorrelation = 0.0

    return Elasticity(float(beta), std_error, r_squared, n, autocorrelation)


def observed_cohort_score(cohort: pd.DataFrame, days: int = 90) -> float:
    """Recent mean cohort comorbidity -- the baseline the slider deviates from."""
    recent = cohort.sort_values("Date").tail(days)
    return float(recent["Mean_Charlson"].mean()) if len(recent) else COMORBIDITY_REFERENCE
