# Medication Consumption Forecast Dashboard

A Dash application for visualising and forecasting medication consumption for
patients, with an adjustment for comorbidity burden.

> **Status:** prototype on synthetic data. The forecasting pipeline is real --
> models are selected per medication by rolling-origin cross-validation -- but
> the data is generated, so the numbers are not clinically meaningful.

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

A model is selected per medication by rolling-origin cross-validation, never by
a single train/test split. Candidates always include naive, seasonal-naive and
historic-average benchmarks; nothing ships unless it beats the right benchmark.
Intermittent series additionally get Croston's method and the Syntetos-Boylan
bias-corrected variant; regular series get AutoETS and AutoARIMA.

Current selections (synthetic data, 730 days):

| Medication | zero-days | model | metric | score |
| --- | --- | --- | --- | --- |
| Omeprazole 20 mg | 0% | AutoETS | MASE 0.708 | beats benchmark |
| Omeprazole 40 mg | 1% | AutoETS | MASE 0.810 | beats benchmark |
| Bisoprolol 2.5 mg | 0% | AutoETS | MASE 0.744 | beats benchmark |
| Bisoprolol 5 mg | 1% | AutoETS | MASE 0.784 | beats benchmark |
| Ciprofloxacin 500 mg | 10% | AutoARIMA | MASE 0.638 | beats benchmark |
| Meropenem 1g | 71% | CrostonSBA | RMSSE 0.436 | beats benchmark |

Three decisions are load-bearing and easy to get wrong:

- **Intermittent series are scored on RMSSE, not MASE.** Absolute error is
  minimised by the conditional median, which is zero when most days are zero,
  so MASE ranked a forecast of *zero Meropenem for 14 days* as the best model.
- **Models that systematically under-supply are disqualified** regardless of
  accuracy. Over- and under-forecasting have very different consequences.
- **Intervals are conformal**, calibrated from past forecast errors. Croston
  admits no analytic interval at all. The interval on cumulative demand is
  calibrated separately from the daily bands, because summing daily bounds
  assumes perfectly correlated errors and grossly over-widens the total.

An equal-weight **combination** of the non-benchmark candidates competes on the
same held-out footing. On this data it places second everywhere and wins
nowhere -- the expected "rarely best, never bad" behaviour -- so it is retained
as a candidate but does not currently ship for any medication.

Every methodological claim traces to a source in `references/citations.db`; see
`REFERENCES.md`.

## Training

Model selection and interval calibration run offline (about two minutes) and
write `artifacts/models.json`, which the app loads at startup:

```bash
python -m medication_app.train
```

The artifact records the configuration it was trained under and the extent of
the data it saw, so it cannot silently outlive them. A **configuration change**
(seed, medication profile, history length) makes the artifact wrong, and the
app discards it rather than serving it. **Data moving on** is reported in days
and only flagged past a threshold, because the synthetic history is anchored to
today and a one-day lag is routine.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Entry point -- builds the app and serves it |
| `medication_app/config.py` | Medication catalogue, seed, UI defaults |
| `medication_app/data.py` | Synthetic dataset generation (swap for a real loader) |
| `medication_app/forecasting.py` | Aggregation: daily count series, moving average |
| `medication_app/models.py` | Candidate models, comorbidity elasticity |
| `medication_app/backtest.py` | Rolling-origin CV, MASE/RMSSE, selection guard |
| `medication_app/intervals.py` | Conformal prediction intervals |
| `medication_app/pipeline.py` | Offline training, artifact read/write |
| `medication_app/fingerprint.py` | Artifact freshness: config hash, data extent |
| `medication_app/citations.py` | Citation database schema and export |
| `references/citations.db` | Research trail: sources, retrievals, claims |
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

## Licence

Code is licensed under **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)); a commercial licence is available on request from the author via [LinkedIn](https://www.linkedin.com/in/martin-monteagudo-farina/). Non-code content is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Details in [`LICENSING.md`](LICENSING.md).

## Disclaimer

Research and demonstration software. Not a medical device and not intended for clinical decision-making, diagnosis or treatment. Provided "as is", without warranty of any kind; the author accepts no liability for any use. Uses synthetic data only.
