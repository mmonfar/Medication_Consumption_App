"""Synthetic dataset generation.

The app runs on generated data, not a real dispensing extract. Everything here
is deterministic given ``RANDOM_SEED`` so that two runs are comparable; replace
``load_sample_data`` with a real loader when a data source exists.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import DOSES, HISTORY_DAYS, MEDICATIONS, NUM_PATIENTS, NUM_RECORDS, RANDOM_SEED


def build_patients(rng: np.random.Generator, num_patients: int = NUM_PATIENTS) -> pd.DataFrame:
    """One row per patient: MRN and Charlson comorbidity score."""
    return pd.DataFrame(
        {
            "MRN": [f"P{str(i).zfill(4)}" for i in range(1, num_patients + 1)],
            "Charlson_Comorbidity_Score": rng.integers(0, 10, num_patients),
        }
    )


def build_consumption(
    patients: pd.DataFrame,
    rng: np.random.Generator,
    num_records: int = NUM_RECORDS,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """One row per administration event over the last ``history_days`` days.

    ``Date`` is normalised to midnight so that grouping by date is a true daily
    aggregation rather than an aggregation by timestamp.
    """
    start_date = (datetime.now() - timedelta(days=history_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    offsets = rng.integers(0, history_days, num_records)

    df = pd.DataFrame(
        {
            "MRN": rng.choice(patients["MRN"], num_records),
            "Date": [start_date + timedelta(days=int(o)) for o in offsets],
            "Time": [f"{h:02}:{m:02}" for h, m in zip(rng.integers(0, 24, num_records), rng.integers(0, 60, num_records))],
            "Medication": rng.choice(MEDICATIONS, num_records),
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["Dose"] = df["Medication"].map(DOSES)
    return df


def load_sample_data(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(patients, consumption)`` for the given seed."""
    rng = np.random.default_rng(seed)
    patients = build_patients(rng)
    return patients, build_consumption(patients, rng)
