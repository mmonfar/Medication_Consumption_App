"""Application factory."""

from __future__ import annotations

import dash

from .callbacks import register_callbacks
from .data import load_sample_data
from .layout import build_layout
from .models import observed_cohort_score
from .pipeline import load as load_artifact
from .pipeline import train_all


def create_app(allow_training: bool = True) -> dash.Dash:
    """Build the Dash app from the cached training artifact.

    Falls back to training in-process if no artifact exists, which is slow --
    run ``python -m medication_app.train`` to avoid it.
    """
    patients, consumption, cohort = load_sample_data()

    trained, trained_at = load_artifact()
    if trained is None:
        if not allow_training:
            raise RuntimeError("no trained artifact; run `python -m medication_app.train`")
        print("no artifact found -- training in-process (run `python -m medication_app.train`)")
        trained = train_all(consumption, cohort)
        trained_at = "just now (in-process)"

    app = dash.Dash(__name__)
    app.title = "Medication Consumption Forecast"
    app.layout = build_layout(observed_cohort_score(cohort))
    register_callbacks(app, patients, consumption, cohort, trained, trained_at)
    return app
