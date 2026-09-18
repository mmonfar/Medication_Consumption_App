"""Aggregation helpers: turning event rows into model-ready daily series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INTERMITTENCY_THRESHOLD


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Trailing mean over ``window`` days.

    Safe now that series are reindexed to a complete daily grid: ``window``
    observations are ``window`` calendar days.
    """
    return series.rolling(window=window, min_periods=1).mean()


def daily_counts(consumption: pd.DataFrame, medication: str) -> pd.DataFrame:
    """Daily administration counts for one medication, on a complete date grid.

    Days with no administrations are genuine zeros, not missing rows, so they
    are filled rather than dropped. Counts -- not summed milligrams -- are the
    forecasting unit: summing mg across strengths mixes 1 mg and 500 mg into a
    meaningless total.

    Returns columns ``ds`` (date) and ``y`` (count), the layout statsforecast
    expects.
    """
    selected = consumption[consumption["Medication"] == medication]
    if selected.empty:
        return pd.DataFrame({"ds": pd.to_datetime([]), "y": []})

    counts = selected.groupby("Date").size()
    grid = pd.date_range(consumption["Date"].min(), consumption["Date"].max(), freq="D")
    counts = counts.reindex(grid, fill_value=0)
    return pd.DataFrame({"ds": counts.index, "y": counts.to_numpy(dtype=float)})


def daily_dose_mg(consumption: pd.DataFrame, medication: str) -> pd.DataFrame:
    """Daily total milligrams for one medication (display only, never forecast)."""
    selected = consumption[consumption["Medication"] == medication]
    totals = selected.groupby("Date")["Dose"].sum()
    grid = pd.date_range(consumption["Date"].min(), consumption["Date"].max(), freq="D")
    totals = totals.reindex(grid, fill_value=0.0)
    return pd.DataFrame({"ds": totals.index, "mg": totals.to_numpy(dtype=float)})


def zero_share(series: pd.Series | np.ndarray) -> float:
    """Proportion of days with no administrations."""
    values = np.asarray(series, dtype=float)
    return float((values == 0).mean()) if values.size else 0.0


def is_intermittent(series: pd.Series | np.ndarray) -> bool:
    """Whether a series is intermittent enough to warrant Croston-family models."""
    return zero_share(series) >= INTERMITTENCY_THRESHOLD


def comorbidity_summary(patients: pd.DataFrame, metric: str) -> float:
    """Mean or median Charlson score across the patient roster."""
    scores = patients["Charlson_Comorbidity_Score"]
    return float(scores.mean() if metric == "mean" else scores.median())


def forecast_dates(series: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """The ``horizon`` days following the last observed day."""
    last = pd.Timestamp(series["ds"].max())
    return pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
