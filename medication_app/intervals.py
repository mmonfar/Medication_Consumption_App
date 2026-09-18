"""Distribution-free prediction intervals by split conformal prediction.

Why conformal rather than each model's native intervals:

* Croston's method has no underlying stochastic model, so it admits no
  analytic interval at all -- yet an intermittent restricted antibiotic is
  exactly where an honest uncertainty range matters most.
* Bootstrapped intervals still assume residuals are uncorrelated with constant
  variance. Conformal assumes only exchangeability, and is model-agnostic, so
  one mechanism covers every medication and the intervals stay comparable
  across them.

Method (Hyndman & Athanasopoulos, section 5.5): collect h-step-ahead errors
e(t+h|t) = y(t+h) - yhat(t+h|t) from a calibration set, then form the interval
at horizon h as yhat +/- Q(1-alpha) of the past absolute h-step errors.
Quantiles are computed per horizon step, so intervals widen with horizon the
way forecast uncertainty actually does.

Two practical details matter as much as the method:

* Origins overlap (``step_size=1``). With one origin per horizon the 95th
  percentile of a handful of errors is effectively the maximum of a tiny
  sample -- unstable, and it undercovered badly in practice (80% bands
  achieving 62%). Overlapping origins give hundreds of h-step errors instead.
* Half-widths are forced non-decreasing in h. Uncertainty about next month
  cannot be lower than uncertainty about tomorrow; where the raw quantiles say
  otherwise it is sampling noise, and letting it through would draw an
  interval that narrows with horizon.

Daily bands and the band on the *total* are calibrated separately. Summing the
daily bounds across a horizon would assume the daily errors are perfectly
correlated, which inflates the total wildly -- on this data it turned a central
estimate of 2.5 Meropenem administrations into an interval of 0 to 23. The
interval on cumulative demand is therefore calibrated directly on cumulative
errors, and that is the number a stock decision should use.

Coverage is measured on windows held out from calibration, so the reported
figure is not the one the quantiles were fitted to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

DEFAULT_LEVELS = (80, 95)
_META_COLUMNS = {"unique_id", "ds", "cutoff", "y"}


@dataclass
class ConformalBands:
    """Per-horizon half-widths, and the coverage they actually achieved."""

    levels: tuple[int, ...]
    half_widths: dict[int, list[float]]
    cumulative_half_widths: dict[int, list[float]]
    coverage: dict[int, float]
    cumulative_coverage: dict[int, float]
    calibration_windows: int
    coverage_windows: int

    def apply(self, forecast: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` for a forecast, clipped at zero.

        Counts cannot be negative, so the lower band is floored -- which makes
        the interval asymmetric near zero. That is a property of the data, not
        a defect: for a medication dispensed 0.3 times a day, "somewhere
        between none and three" is the honest statement.
        """
        width = np.asarray(self.half_widths[level][: len(forecast)], dtype=float)
        if width.size < forecast.size:  # horizon longer than calibrated -- hold last
            width = np.pad(width, (0, forecast.size - width.size), mode="edge")
        return np.clip(forecast - width, 0, None), forecast + width


def _cv_errors(
    series: pd.DataFrame,
    model,
    windows: int,
    horizon: int,
    step_size: int = 1,
    unique_id: str = "series",
) -> pd.DataFrame:
    """Rolling-origin errors, labelled by how many steps ahead they were."""
    prepared = series.assign(unique_id=unique_id)[["unique_id", "ds", "y"]]
    sf = StatsForecast(models=[model], freq="D", n_jobs=1)
    predictions = sf.cross_validation(
        df=prepared, h=horizon, n_windows=windows, step_size=step_size
    ).reset_index(drop=True)

    name = next(c for c in predictions.columns if c not in _META_COLUMNS)
    predictions["error"] = predictions["y"] - predictions[name]
    predictions["step"] = predictions.groupby("cutoff").cumcount() + 1
    # Running totals within each origin, for the interval on cumulative demand.
    grouped = predictions.groupby("cutoff")
    predictions["cum_y"] = grouped["y"].cumsum()
    predictions["cum_yhat"] = grouped[name].cumsum()
    predictions["cum_error"] = predictions["cum_y"] - predictions["cum_yhat"]
    return predictions[
        ["cutoff", "step", "y", name, "error", "cum_y", "cum_yhat", "cum_error"]
    ].rename(columns={name: "yhat"})


def fit_conformal(
    series: pd.DataFrame,
    model,
    horizon: int,
    calibration_windows: int = 120,
    coverage_windows: int = 40,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    step_size: int = 1,
) -> ConformalBands:
    """Calibrate conformal bands, then measure their coverage out-of-sample."""
    total_windows = calibration_windows + coverage_windows
    errors = _cv_errors(series, model, total_windows, horizon, step_size=step_size)

    cutoffs = sorted(errors["cutoff"].unique())
    calibration_cutoffs = set(cutoffs[:calibration_windows])
    calibration = errors[errors["cutoff"].isin(calibration_cutoffs)]
    held_out = errors[~errors["cutoff"].isin(calibration_cutoffs)]

    def widths_for(column: str) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for level in levels:
            by_step = (
                calibration.assign(absolute=calibration[column].abs())
                .groupby("step")["absolute"]
                .quantile(level / 100.0)
                .reindex(range(1, horizon + 1))
                .ffill()
                .bfill()
            )
            # Non-decreasing in horizon: see the module docstring.
            out[level] = [float(v) for v in np.maximum.accumulate(by_step.to_numpy())]
        return out

    half_widths = widths_for("error")
    cumulative_half_widths = widths_for("cum_error")

    def coverage_for(widths: dict[int, list[float]], point: str, actual: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for level in levels:
            w = np.array([widths[level][int(s) - 1] for s in held_out["step"]])
            lower = np.clip(held_out[point].to_numpy() - w, 0, None)
            upper = held_out[point].to_numpy() + w
            observed = held_out[actual].to_numpy()
            inside = (observed >= lower) & (observed <= upper)
            out[level] = float(inside.mean()) if inside.size else float("nan")
        return out

    return ConformalBands(
        levels=tuple(levels),
        half_widths=half_widths,
        cumulative_half_widths=cumulative_half_widths,
        coverage=coverage_for(half_widths, "yhat", "y"),
        cumulative_coverage=coverage_for(cumulative_half_widths, "cum_yhat", "cum_y"),
        calibration_windows=calibration_windows,
        coverage_windows=coverage_windows,
    )
