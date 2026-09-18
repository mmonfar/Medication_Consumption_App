"""Offline training pipeline: select a model per medication and cache the result.

Model selection runs rolling-origin cross-validation over every candidate for
every medication, which takes over a minute -- far too slow to do at app
startup on each boot. So it runs once, ahead of time, and writes an artifact
the app loads in milliseconds:

    python -m medication_app.train

The app degrades honestly when the artifact is missing or stale: it falls back
to a seasonal-naive benchmark and says so in the UI, rather than silently
serving forecasts from a model nobody validated.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import cross_validate, select_model
from .config import MAX_HORIZON, MEDICATIONS
from .forecasting import daily_counts, zero_share
from .models import NO_INTERVAL_MODELS, Elasticity, fit_elasticity, fit_predict, observed_cohort_score

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "models.json"
ARTIFACT_VERSION = 1


@dataclass
class MedicationForecast:
    """Everything the app needs to serve one medication, decided offline."""

    medication: str
    model: str
    primary_metric: str
    score: float
    cumulative_bias: float
    beats_benchmark: bool
    intermittent: bool
    zero_share: float
    has_intervals: bool
    elasticity_beta: float
    elasticity_se: float
    elasticity_usable: bool
    observed_cohort_score: float
    forecast_dates: list[str]
    forecast: list[float]
    leaderboard: list[dict] = field(default_factory=list)

    @property
    def elasticity(self) -> Elasticity:
        return Elasticity(
            self.elasticity_beta,
            self.elasticity_se,
            0.0,
            0,
            0.0 if self.elasticity_usable else 1.0,
        )


def train_medication(
    medication: str, consumption: pd.DataFrame, cohort: pd.DataFrame, horizon: int = MAX_HORIZON
) -> MedicationForecast:
    """Cross-validate every candidate for one medication and fit the winner."""
    series = daily_counts(consumption, medication)
    scores = cross_validate(series)
    model_name, beats = select_model(scores)
    elasticity = fit_elasticity(series, cohort)

    chosen = next(m for m in _instantiate(model_name, series))
    forecast = fit_predict(series, horizon, models=[chosen])
    values = forecast[model_name].clip(lower=0).tolist()

    row = scores[scores["model"] == model_name].iloc[0]
    return MedicationForecast(
        medication=medication,
        model=model_name,
        primary_metric=str(scores.attrs["primary_metric"]),
        score=float(row["primary"]),
        cumulative_bias=float(row["cum_bias"]) if pd.notna(row["cum_bias"]) else 0.0,
        beats_benchmark=beats,
        intermittent=bool(scores.attrs["intermittent"]),
        zero_share=zero_share(series["y"]),
        has_intervals=model_name not in NO_INTERVAL_MODELS,
        elasticity_beta=elasticity.beta,
        elasticity_se=elasticity.std_error,
        elasticity_usable=elasticity.is_usable,
        observed_cohort_score=observed_cohort_score(cohort),
        forecast_dates=[d.strftime("%Y-%m-%d") for d in pd.to_datetime(forecast["ds"])],
        forecast=values,
        leaderboard=scores[["model", "primary", "cum_bias"]].round(4).to_dict("records"),
    )


def _instantiate(model_name: str, series: pd.DataFrame):
    """Find the candidate object matching a model's reported name."""
    from .models import candidate_models

    matches = [m for m in candidate_models(series) if repr(m) == model_name or type(m).__name__ == model_name]
    if not matches:
        raise ValueError(f"no candidate model named {model_name!r}")
    return matches


def train_all(consumption: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, MedicationForecast]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {med: train_medication(med, consumption, cohort) for med in MEDICATIONS}


def save(results: dict[str, MedicationForecast], path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ARTIFACT_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "medications": {name: asdict(result) for name, result in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(path: Path = ARTIFACT_PATH) -> tuple[dict[str, MedicationForecast], str] | tuple[None, None]:
    """Load the trained artifact, or ``(None, None)`` if absent/incompatible."""
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != ARTIFACT_VERSION:
            return None, None
        results = {
            name: MedicationForecast(**data) for name, data in payload["medications"].items()
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, None
    return results, payload.get("trained_at", "unknown")
