"""Static configuration: medication catalogue, generator profiles, UI defaults."""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Medication catalogue
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MedicationProfile:
    """Parameters driving the synthetic generator for one medication.

    These encode the clinical behaviour we expect a forecaster to recover:
    a baseline rate, a weekday/weekend split, an annual (winter) cycle, a slow
    trend, and how strongly the cohort's comorbidity burden drives usage.
    """

    dose_mg: float
    base_rate: float        # mean administrations per day at baseline
    weekend_factor: float   # multiplier on Sat/Sun
    winter_amplitude: float # 0 = no annual cycle; 0.4 = +/-40% peak-to-mean
    annual_trend: float     # multiplicative change per year (1.0 = flat)
    comorbidity_beta: float # elasticity w.r.t. cohort mean Charlson score
    intermittent: bool      # sporadic, restricted use -> many zero days


PROFILES: dict[str, MedicationProfile] = {
    # High-volume ward staples: steady, weekday-heavy, little seasonality.
    "Omeprazole 20 mg": MedicationProfile(20, 14.0, 0.82, 0.05, 1.04, 0.35, False),
    "Omeprazole 40 mg": MedicationProfile(40, 6.0, 0.85, 0.05, 1.06, 0.45, False),
    "Bisoprolol 2.5 mg": MedicationProfile(2.5, 9.0, 0.90, 0.02, 1.02, 0.30, False),
    "Bisoprolol 5 mg": MedicationProfile(5, 5.0, 0.90, 0.02, 1.03, 0.30, False),
    # Antibiotics: strong winter peak. Meropenem is restricted -> intermittent.
    "Ciprofloxacin 500 mg": MedicationProfile(500, 3.5, 0.95, 0.45, 1.01, 0.55, False),
    "Meropenem 1g": MedicationProfile(1, 0.45, 0.98, 0.60, 1.08, 0.80, True),
}

DOSES: dict[str, float] = {name: p.dose_mg for name, p in PROFILES.items()}
MEDICATIONS: list[str] = list(PROFILES)

# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
NUM_PATIENTS = 400
HISTORY_DAYS = 730          # two years: enough for annual + weekly cycles
COMORBIDITY_REFERENCE = 5.0 # cohort score at which profiles sit at base_rate

# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #

WEEKLY_SEASONALITY = 7
MAX_HORIZON = 30
# A series is treated as intermittent when this share of days are zero.
INTERMITTENCY_THRESHOLD = 0.30
# Rolling-origin cross-validation.
CV_WINDOWS = 5
CV_HORIZON = 14

# --------------------------------------------------------------------------- #
# UI defaults
# --------------------------------------------------------------------------- #

DEFAULT_MEDICATION = "Meropenem 1g"
DEFAULT_FORECAST_DAYS = 14
DEFAULT_MA_PERIOD = 7
DEFAULT_COMORBIDITY = 5.0
MA_PERIODS = [3, 7, 14, 30]
SERVER_PORT = 8501
