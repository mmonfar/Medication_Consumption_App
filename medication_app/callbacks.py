"""Callback wiring: serves the offline-trained forecast for the chosen medication."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, html

from .forecasting import daily_counts, moving_average
from .pipeline import MedicationForecast

_WARNING_STYLE = {"color": "#8a4b00", "fontWeight": "600"}
_MUTED_STYLE = {"color": "#555"}


def build_figure(
    history: pd.DataFrame, ma_period: int, dates, forecast: np.ndarray, medication: str
) -> go.Figure:
    figure = go.Figure(
        [
            go.Scatter(
                x=history["ds"],
                y=history["y"],
                mode="lines",
                name="Daily administrations",
                line=dict(color="#9ec5e8", width=1),
            ),
            go.Scatter(
                x=history["ds"],
                y=moving_average(history["y"], ma_period),
                mode="lines",
                name=f"{ma_period}-day MA",
                line=dict(color="#1f77b4", width=2),
            ),
            go.Scatter(
                x=dates,
                y=forecast,
                mode="lines",
                name="Forecast",
                line=dict(color="#d55e00", width=2, dash="dash"),
            ),
        ]
    )
    figure.update_layout(
        title=f"Medication consumption and forecast — {medication}",
        xaxis={"title": "Date", "rangeslider": {"visible": True}},
        yaxis={"title": "Administrations per day", "rangemode": "tozero"},
        showlegend=True,
        margin={"t": 60, "b": 40},
    )
    return figure


def _provenance(result: MedicationForecast, adjusted: bool) -> list:
    """Say which model is serving this forecast and what it cannot do."""
    lines = [
        html.Div(
            f"Model: {result.model} — selected by rolling-origin cross-validation "
            f"({result.primary_metric.upper()} {result.score:.3f}, "
            f"cumulative bias {result.cumulative_bias:+.1%})."
        )
    ]
    if not result.beats_benchmark:
        lines.append(
            html.Div(
                "No candidate model beat the benchmark for this series; the benchmark is serving.",
                style=_WARNING_STYLE,
            )
        )
    if not result.has_intervals:
        lines.append(
            html.Div(
                "Croston's method has no underlying stochastic model, so no prediction "
                "interval can be computed. Treat this as a central estimate only.",
                style=_WARNING_STYLE,
            )
        )
    if result.intermittent:
        lines.append(
            html.Div(
                f"Intermittent series ({result.zero_share:.0%} of days have no administrations) — "
                "scored on RMSSE, since absolute error would reward forecasting zero.",
                style=_MUTED_STYLE,
            )
        )
    if not result.elasticity_usable:
        lines.append(
            html.Div(
                "Comorbidity elasticity is not statistically distinguishable from zero for this "
                "medication, so the comorbidity slider does not adjust the forecast.",
                style=_MUTED_STYLE,
            )
        )
    elif adjusted:
        lines.append(
            html.Div(
                f"Comorbidity scenario applied: elasticity {result.elasticity_beta:+.2f} "
                f"(SE {result.elasticity_se:.2f}) against a recent cohort mean of "
                f"{result.observed_cohort_score:.2f}.",
                style=_MUTED_STYLE,
            )
        )
    return lines


def register_callbacks(
    app,
    patients: pd.DataFrame,
    consumption: pd.DataFrame,
    cohort: pd.DataFrame,
    trained: dict[str, MedicationForecast],
    trained_at: str | None,
) -> None:
    @app.callback(
        [
            Output("medication-line-bar-chart", "figure"),
            Output("summary-message", "children"),
        ],
        [
            Input("medication-dropdown", "value"),
            Input("forecast-days", "value"),
            Input("ma-dropdown", "value"),
            Input("predicted-comorbidity-slider", "value"),
        ],
    )
    def update_graph(medication, forecast_days, ma_period, predicted_comorbidity):
        history = daily_counts(consumption, medication)
        result = trained[medication]

        dates = pd.to_datetime(result.forecast_dates[:forecast_days])
        forecast = np.asarray(result.forecast[:forecast_days], dtype=float)

        adjusted = abs(predicted_comorbidity - result.observed_cohort_score) > 1e-9
        forecast = result.elasticity.adjust(
            forecast, predicted_comorbidity, result.observed_cohort_score
        )

        figure = build_figure(history, ma_period, dates, forecast, medication)

        total = float(forecast.sum())
        summary = [
            html.H4(
                f"{total:.0f} administrations forecast over the next {forecast_days} days "
                f"({total / max(forecast_days, 1):.1f}/day)"
            ),
            *_provenance(result, adjusted),
        ]
        if trained_at:
            summary.append(
                html.Div(f"Models trained {trained_at}.", style={"fontSize": "0.85em", "color": "#777"})
            )
        return figure, summary
