"""Application factory."""

from __future__ import annotations

import dash

from .callbacks import register_callbacks
from .data import load_sample_data
from .layout import build_layout
from .models import observed_cohort_score
from .pipeline import Artifact, load as load_artifact
from .pipeline import save, train_all


def create_app(allow_training: bool = True) -> dash.Dash:
    """Build the Dash app from the cached training artifact.

    Falls back to training in-process if no artifact exists, which is slow --
    run ``python -m medication_app.train`` to avoid it.
    """
    patients, consumption, cohort = load_sample_data()

    artifact = load_artifact(consumption=consumption)
    if artifact is not None and artifact.freshness is not None and not artifact.freshness.is_usable:
        # A configuration mismatch means the cached models describe different
        # data. Serving them would be silently wrong, so they are discarded.
        print(f"artifact rejected: {artifact.freshness.describe()}")
        artifact = None

    if artifact is None:
        if not allow_training:
            raise RuntimeError("no usable artifact; run `python -m medication_app.train`")
        print("no usable artifact -- training in-process (run `python -m medication_app.train`)")
        results = train_all(consumption, cohort)
        save(results, consumption)
        artifact = Artifact(results, "just now (in-process)")

    app = dash.Dash(__name__)
    app.title = "Medication Consumption Forecast"
    app.layout = build_layout(observed_cohort_score(cohort))
    register_callbacks(
        app, patients, consumption, cohort, artifact.models, artifact.trained_at, artifact.freshness
    )
    return app
