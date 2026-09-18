"""Static configuration: the medication catalogue and app-wide defaults."""

from __future__ import annotations

# Dose in mg per administration, keyed by the label shown in the UI.
DOSES: dict[str, float] = {
    "Omeprazole 20 mg": 20,
    "Omeprazole 40 mg": 40,
    "Bisoprolol 2.5 mg": 2.5,
    "Bisoprolol 5 mg": 5,
    "Meropenem 1g": 1,
    "Ciprofloxacin 500 mg": 500,
}

MEDICATIONS: list[str] = list(DOSES)

# Synthetic-data parameters. Seeded so every run produces the same dataset.
RANDOM_SEED = 42
NUM_PATIENTS = 50
NUM_RECORDS = 300
HISTORY_DAYS = 30

# UI defaults.
DEFAULT_MEDICATION = "Meropenem 1g"
DEFAULT_FORECAST_DAYS = 7
DEFAULT_MA_PERIOD = 3
DEFAULT_COMORBIDITY = 5.0
MA_PERIODS = [3, 7, 14, 30]
COMORBIDITY_SCALE = 5.0  # divisor in the placeholder forecast adjustment

SERVER_PORT = 8501
