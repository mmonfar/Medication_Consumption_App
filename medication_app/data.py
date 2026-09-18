"""Synthetic dataset generation.

The app runs on generated data, not a real dispensing extract. The generator
deliberately encodes learnable structure -- weekly and annual seasonality, a
slow trend, per-medication intermittency, and a genuine comorbidity effect --
because a model can only recover signal that was put there. Counts are drawn
from a Poisson distribution, so the series are non-negative integers, which is
what dispensing data actually looks like.

Everything is deterministic given ``RANDOM_SEED``. Replace ``load_sample_data``
with a real loader when a data source exists; the rest of the app depends only
on the returned frames' columns.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import (
    COMORBIDITY_REFERENCE,
    HISTORY_DAYS,
    NUM_PATIENTS,
    PROFILES,
    RANDOM_SEED,
)


def build_patients(rng: np.random.Generator, num_patients: int = NUM_PATIENTS) -> pd.DataFrame:
    """One row per patient: MRN and Charlson comorbidity score.

    Scores are drawn from a right-skewed binomial rather than a uniform, which
    is closer to a real inpatient cohort (most patients low, a long tail high).
    """
    return pd.DataFrame(
        {
            "MRN": [f"P{str(i).zfill(4)}" for i in range(1, num_patients + 1)],
            "Charlson_Comorbidity_Score": rng.binomial(10, 0.35, num_patients),
        }
    )


def build_cohort(
    dates: pd.DatetimeIndex, patients: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Daily mean Charlson score of the admitted cohort.

    This must vary over time, otherwise its effect is absorbed into each
    series' level and the comorbidity elasticity is not identifiable -- the
    slider would be unlearnable by construction. The variation is a slow
    mean-reverting walk around the roster mean, plus a mild winter increase
    (sicker admissions in winter).
    """
    baseline = float(patients["Charlson_Comorbidity_Score"].mean())
    n = len(dates)

    walk = np.zeros(n)
    for i in range(1, n):  # AR(1) around zero
        walk[i] = 0.97 * walk[i - 1] + rng.normal(0, 0.09)

    seasonal = 0.35 * np.cos(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    score = np.clip(baseline + walk + seasonal, 0.5, 10.0)
    return pd.DataFrame({"Date": dates, "Mean_Charlson": score})


def _daily_intensity(
    dates: pd.DatetimeIndex, profile, cohort_score: np.ndarray
) -> np.ndarray:
    """Expected administrations per day for one medication.

    Multiplicative decomposition: base x weekday x annual x trend x comorbidity.
    """
    day_of_year = dates.dayofyear.to_numpy()
    years_elapsed = (dates - dates[0]).days.to_numpy() / 365.25

    weekday = np.where(dates.dayofweek.to_numpy() >= 5, profile.weekend_factor, 1.0)
    # Peaks in January (northern-hemisphere winter), troughs in July.
    annual = 1.0 + profile.winter_amplitude * np.cos(2 * np.pi * day_of_year / 365.25)
    trend = profile.annual_trend**years_elapsed
    comorbidity = (cohort_score / COMORBIDITY_REFERENCE) ** profile.comorbidity_beta

    return profile.base_rate * weekday * annual * trend * comorbidity


def build_consumption(
    patients: pd.DataFrame,
    cohort: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per administration event over the last ``history_days`` days.

    ``Date`` is normalised to midnight so grouping by date is a true daily
    aggregation. Patients are sampled with probability proportional to their
    comorbidity score, so sicker patients genuinely receive more medication.
    """
    dates = pd.DatetimeIndex(cohort["Date"])
    cohort_score = cohort["Mean_Charlson"].to_numpy()

    scores = patients["Charlson_Comorbidity_Score"].to_numpy()
    # +1 so score-0 patients are still reachable.
    weights = (scores + 1) / (scores + 1).sum()

    frames = []
    for medication, profile in PROFILES.items():
        intensity = _daily_intensity(dates, profile, cohort_score)
        counts = rng.poisson(intensity)
        total = int(counts.sum())
        if total == 0:
            continue

        frames.append(
            pd.DataFrame(
                {
                    "MRN": rng.choice(patients["MRN"], total, p=weights),
                    "Date": np.repeat(dates.to_numpy(), counts),
                    "Time": [
                        f"{h:02}:{m:02}"
                        for h, m in zip(rng.integers(6, 23, total), rng.integers(0, 60, total))
                    ],
                    "Medication": medication,
                    "Dose": profile.dose_mg,
                }
            )
        )

    consumption = pd.concat(frames, ignore_index=True)
    consumption["Date"] = pd.to_datetime(consumption["Date"])
    return consumption.sort_values(["Date", "Medication"]).reset_index(drop=True)


def load_sample_data(
    seed: int = RANDOM_SEED, history_days: int = HISTORY_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(patients, consumption, cohort)`` for the given seed."""
    rng = np.random.default_rng(seed)
    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(end=end, periods=history_days, freq="D")

    patients = build_patients(rng)
    cohort = build_cohort(dates, patients, rng)
    consumption = build_consumption(patients, cohort, rng)
    return patients, consumption, cohort
