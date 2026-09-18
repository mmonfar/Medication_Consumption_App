"""Application factory."""

from __future__ import annotations

import dash

from .callbacks import register_callbacks
from .data import load_sample_data
from .layout import build_layout


def create_app() -> dash.Dash:
    patients, consumption = load_sample_data()
    app = dash.Dash(__name__)
    app.layout = build_layout()
    register_callbacks(app, patients, consumption)
    return app
