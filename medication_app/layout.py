"""Dash layout. Pure presentation -- no data access, no forecasting."""

from __future__ import annotations

from dash import dcc, html

from .config import (
    DEFAULT_COMORBIDITY,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_MA_PERIOD,
    DEFAULT_MEDICATION,
    MA_PERIODS,
    MEDICATIONS,
)

_SECTION = {"padding": "10px"}
_SLIDER_BOX = {"width": "1000px", "padding": "10px", "backgroundColor": "#f0f0f0"}


def _labelled(label: str, control) -> html.Div:
    return html.Div([html.Label(label), control], style=_SECTION)


def build_layout() -> html.Div:
    return html.Div(
        [
            html.H1("Medication Consumption and Forecast Analysis"),
            _labelled(
                "Select Medication:",
                dcc.Dropdown(
                    id="medication-dropdown",
                    options=[{"label": med, "value": med} for med in MEDICATIONS],
                    value=DEFAULT_MEDICATION,
                ),
            ),
            _labelled(
                "Select Number of Forecast Days:",
                html.Div(
                    dcc.Slider(
                        id="forecast-days",
                        min=1,
                        max=30,
                        step=1,
                        value=DEFAULT_FORECAST_DAYS,
                        marks={i: f"{i}" for i in range(1, 31)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    style=_SLIDER_BOX,
                ),
            ),
            _labelled(
                "Select Moving Average Period:",
                dcc.Dropdown(
                    id="ma-dropdown",
                    options=[{"label": f"{p} Days", "value": p} for p in MA_PERIODS],
                    value=DEFAULT_MA_PERIOD,
                ),
            ),
            _labelled(
                "Select Comorbidity Metric:",
                dcc.Dropdown(
                    id="comorbidity-selection",
                    options=[
                        {"label": "Comorbidity Mean", "value": "mean"},
                        {"label": "Comorbidity Median", "value": "median"},
                    ],
                    value="mean",
                ),
            ),
            _labelled(
                "Select Predicted Comorbidity Score for Forecast Period:",
                html.Div(
                    dcc.Slider(
                        id="predicted-comorbidity-slider",
                        min=0,
                        max=10,
                        step=0.1,
                        value=DEFAULT_COMORBIDITY,
                        marks={i: f"{i}" for i in range(0, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    style=_SLIDER_BOX,
                ),
            ),
            dcc.Graph(id="medication-line-bar-chart"),
            html.Div(id="summary-message", style=_SECTION),
        ]
    )
