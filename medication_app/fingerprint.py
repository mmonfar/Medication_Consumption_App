"""Fingerprinting, so a trained artifact cannot silently outlive its data.

A cached model is a claim about data that existed when it was fitted. Nothing
in the app forced that claim to stay true: edit a generator profile, change the
seed, or simply come back a week later, and the artifact would keep serving
forecasts fitted to data that no longer exists, with no visible sign.

Two different things can go stale, and they deserve different responses:

* **The generator's configuration changed** -- a different seed, a different
  medication profile, a different history length. The artifact now describes a
  different world. This is a hard mismatch; the forecast is wrong, not merely
  old.
* **The data has moved on** while the configuration is unchanged. Because the
  synthetic history is anchored to today, this happens every day, and it will
  happen with a real dispensing feed too. This is a soft condition measured in
  days, not a binary: a model fitted to data ending last Tuesday is usually
  still serviceable on Wednesday and questionable a month later.

Collapsing both into one boolean would either cry wolf daily or hide a real
mismatch, so they are reported separately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .config import (
    COMORBIDITY_REFERENCE,
    HISTORY_DAYS,
    INTERMITTENCY_THRESHOLD,
    NUM_PATIENTS,
    PROFILES,
    RANDOM_SEED,
    WEEKLY_SEASONALITY,
)

#: Beyond this, a model is reported as due for retraining.
STALE_AFTER_DAYS = 14


def config_fingerprint() -> str:
    """Hash of everything that determines what the data and models mean.

    Covers the generator's parameters and the settings that drive model choice.
    A change to any of them invalidates a cached artifact outright.
    """
    payload = {
        "seed": RANDOM_SEED,
        "patients": NUM_PATIENTS,
        "history_days": HISTORY_DAYS,
        "comorbidity_reference": COMORBIDITY_REFERENCE,
        "intermittency_threshold": INTERMITTENCY_THRESHOLD,
        "weekly_seasonality": WEEKLY_SEASONALITY,
        "profiles": {
            name: [
                profile.dose_mg,
                profile.base_rate,
                profile.weekend_factor,
                profile.winter_amplitude,
                profile.annual_trend,
                profile.comorbidity_beta,
                profile.intermittent,
            ]
            for name, profile in sorted(PROFILES.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def data_extent(consumption: pd.DataFrame) -> dict[str, object]:
    """The shape of the data a model was fitted to."""
    return {
        "rows": int(len(consumption)),
        "first_date": consumption["Date"].min().strftime("%Y-%m-%d"),
        "last_date": consumption["Date"].max().strftime("%Y-%m-%d"),
    }


@dataclass(frozen=True)
class Freshness:
    """Whether a cached artifact still describes the data in front of us."""

    config_matches: bool
    days_behind: int
    artifact_last_date: str
    current_last_date: str

    @property
    def is_usable(self) -> bool:
        """A configuration mismatch makes the artifact wrong, not just old."""
        return self.config_matches

    @property
    def is_stale(self) -> bool:
        return self.days_behind > STALE_AFTER_DAYS

    def describe(self) -> str:
        if not self.config_matches:
            return (
                "The trained models were fitted under a different data configuration "
                "and no longer describe this dataset. Retrain with "
                "`python -m medication_app.train`."
            )
        if self.is_stale:
            return (
                f"Models were fitted to data ending {self.artifact_last_date}, "
                f"{self.days_behind} days before the latest observation "
                f"({self.current_last_date}). Retraining is due."
            )
        if self.days_behind > 0:
            return (
                f"Models were fitted to data ending {self.artifact_last_date}, "
                f"{self.days_behind} day(s) behind the latest observation."
            )
        return "Models are current with the data."


def assess(artifact: dict, consumption: pd.DataFrame) -> Freshness:
    """Compare a stored artifact's provenance against the data now in hand."""
    current = data_extent(consumption)
    stored_config = artifact.get("config_fingerprint", "")
    stored_extent = artifact.get("data_extent", {}) or {}
    artifact_last = str(stored_extent.get("last_date", "")) or "unknown"

    days_behind = 0
    if artifact_last != "unknown":
        delta = pd.Timestamp(current["last_date"]) - pd.Timestamp(artifact_last)
        days_behind = max(int(delta.days), 0)

    return Freshness(
        config_matches=stored_config == config_fingerprint(),
        days_behind=days_behind,
        artifact_last_date=artifact_last,
        current_last_date=str(current["last_date"]),
    )
