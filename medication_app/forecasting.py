"""Aggregation and forecasting.

This is the seam where the real model goes. ``placeholder_forecast`` is NOT a
forecast -- see its docstring. Replacing it (Croston-family for intermittent
items, ETS/ARIMA for high-volume ones, scored against a naive baseline on a
held-out window) is the one outstanding piece of work in this project.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import COMORBIDITY_SCALE


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Trailing mean over ``window`` observations.

    Returns all-NaN when ``window`` exceeds the length of ``series``, which is
    easy to hit here: the history is only 30 days and 30 is a selectable window.
    """
    return series.rolling(window=window).mean()


def daily_totals(consumption: pd.DataFrame, medication: str) -> pd.DataFrame:
    """Daily summed dose for one medication, ascending by date.

    Days with no administrations are absent rather than zero, so the index has
    gaps. Any real model fitted downstream must reindex to a complete daily
    range first.
    """
    selected = consumption[consumption["Medication"] == medication]
    daily = selected.groupby("Date", as_index=False).agg({"Dose": "sum"})
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily.sort_values("Date").reset_index(drop=True)


def comorbidity_summary(patients: pd.DataFrame, metric: str) -> float:
    """Mean or median Charlson score across the cohort."""
    scores = patients["Charlson_Comorbidity_Score"]
    return float(scores.mean() if metric == "mean" else scores.median())


def forecast_dates(daily: pd.DataFrame, horizon: int) -> list[pd.Timestamp]:
    """The ``horizon`` days following the last observed day."""
    last = daily["Date"].max()
    return [last + timedelta(days=i) for i in range(1, horizon + 1)]


def placeholder_forecast(
    daily: pd.DataFrame,
    horizon: int,
    predicted_comorbidity: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform noise between the mean and max of history, scaled by comorbidity.

    WARNING: this is a visual placeholder carried over from the prototype, not a
    model. It has no memory of trend or seasonality, produces a different answer
    on every call, and carries no uncertainty estimate. Its output must not be
    used for stock or dosing decisions.
    """
    rng = rng if rng is not None else np.random.default_rng()
    draws = rng.uniform(daily["Dose"].mean(), daily["Dose"].max(), horizon)
    return draws * (predicted_comorbidity / COMORBIDITY_SCALE)
