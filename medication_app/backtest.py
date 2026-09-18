"""Rolling-origin cross-validation and model selection.

The vault's application guide (section 4.9) is explicit that methods must be
compared with rolling-origin / time-series cross-validation rather than a
single train/test split, so that is what selection uses here.

Accuracy is scale-free so that medications dispensed at very different volumes
are comparable, and defined when the actual is zero -- which rules MAPE out,
since most days are zero for an intermittent series.

Which scaled error matters depends on the series:

* MASE (absolute error) for regular series.
* RMSSE (squared error) for intermittent series. This is not cosmetic. Absolute
  error is minimised by the conditional *median*, and for a series that is zero
  on most days the median is zero -- so MASE actively rewards a forecast of
  "zero doses forever", which is useless for stock planning. Squared error is
  minimised by the conditional *mean*, which is the quantity Croston estimates
  and the quantity stock decisions need. Both scaled errors are defined in
  Hyndman & Koehler (2006); see fpppy/05-05-toolbox.md.

Cumulative demand over the horizon is also reported, since that -- not
day-by-day accuracy -- is what determines whether stock runs out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from .config import CV_HORIZON, CV_WINDOWS
from .forecasting import is_intermittent
from .models import BENCHMARKS, COMBINATION, candidate_models

_META_COLUMNS = {"unique_id", "ds", "cutoff", "y"}


def mase(actual: np.ndarray, predicted: np.ndarray, insample: np.ndarray, season: int = 1) -> float:
    """Mean absolute scaled error against an in-sample naive benchmark."""
    scale = np.mean(np.abs(insample[season:] - insample[:-season]))
    if not np.isfinite(scale) or scale == 0:
        return float("inf")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def rmsse(actual: np.ndarray, predicted: np.ndarray, insample: np.ndarray, season: int = 1) -> float:
    """Root mean squared scaled error -- the mean-optimal counterpart to MASE."""
    scale = np.mean((insample[season:] - insample[:-season]) ** 2)
    if not np.isfinite(scale) or scale == 0:
        return float("inf")
    return float(np.sqrt(np.mean((actual - predicted) ** 2) / scale))


def cumulative_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Signed error on total demand, aggregated across all windows.

    Aggregated rather than averaged per window: an individual window of an
    intermittent series can have zero total demand, which makes a per-window
    relative error undefined. Positive means over-forecasting (overstock),
    negative means under-forecasting (stockout risk) -- the asymmetry that
    matters clinically, and which a symmetric error metric hides.
    """
    total = float(np.sum(actual))
    if total == 0:
        return float("nan")
    return (float(np.sum(predicted)) - total) / total


def cross_validate(
    series: pd.DataFrame,
    models: list | None = None,
    windows: int = CV_WINDOWS,
    horizon: int = CV_HORIZON,
    unique_id: str = "series",
) -> pd.DataFrame:
    """Rolling-origin CV. Returns one row per model with its mean MASE."""
    models = models if models is not None else candidate_models(series)
    prepared = series.assign(unique_id=unique_id)[["unique_id", "ds", "y"]]

    required = windows * horizon + horizon
    if len(prepared) < required:
        raise ValueError(
            f"need at least {required} observations for {windows} windows "
            f"of horizon {horizon}; got {len(prepared)}"
        )

    sf = StatsForecast(models=models, freq="D", n_jobs=1)
    predictions = sf.cross_validation(
        df=prepared, h=horizon, n_windows=windows, step_size=horizon
    ).reset_index(drop=True)

    # A simple average of the non-benchmark candidates, scored as its own
    # entry. Combining forecasts is a reliable, cheap accuracy gain (Bates &
    # Granger, 1969) and it competes here on the same held-out footing as the
    # models it averages -- it is not assumed to help.
    components = [
        c for c in predictions.columns if c not in _META_COLUMNS and c not in BENCHMARKS
    ]
    if len(components) > 1:
        predictions[COMBINATION] = predictions[components].mean(axis="columns")

    insample = prepared["y"].to_numpy()
    intermittent = is_intermittent(insample)
    primary = "rmsse" if intermittent else "mase"
    model_names = [c for c in predictions.columns if c not in _META_COLUMNS]

    rows = []
    for name in model_names:
        windows_ = [w for _, w in predictions.groupby("cutoff")]
        rows.append(
            {
                "model": name,
                "mase": float(np.mean([mase(w["y"].to_numpy(), w[name].to_numpy(), insample) for w in windows_])),
                "rmsse": float(np.mean([rmsse(w["y"].to_numpy(), w[name].to_numpy(), insample) for w in windows_])),
                "cum_bias": cumulative_bias(
                    np.concatenate([w["y"].to_numpy() for w in windows_]),
                    np.concatenate([w[name].to_numpy() for w in windows_]),
                ),
                "windows": len(windows_),
                "is_benchmark": name in BENCHMARKS,
            }
        )

    scores = pd.DataFrame(rows)
    scores["primary"] = scores[primary]
    scores.attrs["primary_metric"] = primary
    scores.attrs["intermittent"] = intermittent
    return scores.sort_values("primary").reset_index(drop=True)


#: A model that under-forecasts total demand by more than this over the
#: backtest is disqualified regardless of its point accuracy.
MAX_UNDER_SUPPLY = 0.50


def eligible(scores: pd.DataFrame) -> pd.DataFrame:
    """Drop models that systematically under-supply.

    On this dataset the Naive model wins both MASE and RMSSE for Meropenem
    while forecasting zero doses for the entire 14-day horizon -- a cumulative
    bias of -100%, i.e. a guaranteed stockout for a restricted antibiotic. It
    scores well because point-accuracy metrics reward matching the many zero
    days and barely penalise missing the rare non-zero ones. Accuracy alone is
    therefore not a safe selection rule for intermittent medication demand:
    over- and under-forecasting have very different consequences, and only one
    of them harms a patient.
    """
    keep = scores["cum_bias"].isna() | (scores["cum_bias"] > -MAX_UNDER_SUPPLY)
    return scores[keep] if keep.any() else scores


def select_model(scores: pd.DataFrame) -> tuple[str, bool]:
    """Pick the winner and report whether it actually beat the benchmarks.

    Returns ``(model_name, beats_benchmark)``. Models that systematically
    under-supply are disqualified first. Among the rest, when no candidate
    beats the best benchmark the benchmark itself is returned -- per the
    guide, a complex model that loses to seasonal naive is evidence of a bug,
    not a reason to ship the complex model.
    """
    viable = eligible(scores)
    best = viable.iloc[0]
    benchmarks = viable[viable["is_benchmark"]]
    best_benchmark = benchmarks["primary"].min() if len(benchmarks) else np.inf

    if bool(best["is_benchmark"]) or best["primary"] >= best_benchmark:
        winner = benchmarks.iloc[0] if len(benchmarks) else best
        return str(winner["model"]), False
    return str(best["model"]), True
