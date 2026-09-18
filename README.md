# Medication Consumption Forecast Dashboard

A Dash application for visualising and forecasting medication consumption for
patients, with an adjustment for comorbidity burden.

> **Status:** prototype. The data is synthetic (generated at import time in
> `app.py`) and the forecast is a placeholder, not a fitted model. See
> "Forecasting approach" below.

## Features
- Select medication type, forecast horizon, and moving-average window.
- Choose mean or median Charlson comorbidity score.
- Adjust the forecast with a predicted comorbidity score for the horizon.
- Line plot of historical consumption, moving average, and forecast.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

To reproduce the exact environment captured on 2026-08-29 instead:

```bash
pip install -r requirements-lock.txt
```

## Running

```bash
python app.py
```

Then open <http://127.0.0.1:8501>.

## Forecasting approach

The current forecast is **not** a model: it draws uniform random values between
the mean and max of observed daily doses and scales them by
`predicted_comorbidity / 5`. It is a visual placeholder only — the numbers it
reports are not predictions and must not be used for stock or dosing decisions.

It lives in `medication_app/forecasting.py` as `placeholder_forecast`, which is
the single seam a real model replaces.

Planned replacement (see the project notes): a per-medication baseline
(seasonal naive / drift) as a reference, Croston-family methods for
intermittently-dispensed items, and ETS/ARIMA for high-volume items, all scored
against a held-out test window before any of them is shown as a forecast.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Entry point -- builds the app and serves it |
| `medication_app/config.py` | Medication catalogue, seed, UI defaults |
| `medication_app/data.py` | Synthetic dataset generation (swap for a real loader) |
| `medication_app/forecasting.py` | Aggregation, moving average, placeholder forecast |
| `medication_app/layout.py` | Dash layout (presentation only) |
| `medication_app/callbacks.py` | Callback wiring and figure construction |
| `medication_app/app.py` | `create_app()` application factory |
| `tests/` | pytest suite for the aggregation/forecast helpers |
| `requirements.txt` | Curated top-level dependencies |
| `requirements-dev.txt` | The above plus pytest |
| `requirements-lock.txt` | Pinned versions of the original environment |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```
