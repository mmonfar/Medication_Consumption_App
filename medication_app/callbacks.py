"""Callback wiring: serves the offline-trained forecast for the chosen medication.

The trust panel is not decoration. Every model here carries a limitation that
changes how its number should be read -- Croston admits no analytic interval,
some elasticities are indistinguishable from zero, some bands under-cover --
and a forecast presented without those is more dangerous than no forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, html

from .forecasting import daily_counts, moving_average
from .layout import ACCENT, INK, LINE, MUTED, WARN
from .pipeline import MedicationForecast

_HISTORY = "#b9c6d2"
_FORECAST = "#d55e00"
_BAND = "rgba(213, 94, 0, 0.16)"

_NOTE = {"color": MUTED, "fontSize": "0.88em", "margin": "4px 0"}
_FLAG = {"color": WARN, "fontSize": "0.88em", "margin": "4px 0", "fontWeight": "600"}


def build_figure(
    history: pd.DataFrame,
    ma_period: int,
    dates: pd.DatetimeIndex,
    forecast: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    medication: str,
) -> go.Figure:
    traces = [
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            mode="lines",
            name="Observed",
            line={"color": _HISTORY, "width": 1},
            hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} administrations<extra></extra>",
        ),
        go.Scatter(
            x=history["ds"],
            y=moving_average(history["y"], ma_period),
            mode="lines",
            name=f"{ma_period}-day average",
            line={"color": ACCENT, "width": 2},
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} avg<extra></extra>",
        ),
    ]

    if bounds is not None:
        lower, upper = bounds
        traces.append(
            go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=_BAND,
                line={"width": 0},
                name="Prediction interval",
                hoverinfo="skip",
            )
        )

    traces.append(
        go.Scatter(
            x=dates,
            y=forecast,
            mode="lines",
            name="Forecast",
            line={"color": _FORECAST, "width": 2.5, "dash": "dash"},
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} forecast<extra></extra>",
        )
    )

    figure = go.Figure(traces)
    if len(dates):
        figure.add_vline(
            x=dates[0], line_width=1, line_dash="dot", line_color=MUTED, opacity=0.6
        )
    figure.update_layout(
        title={"text": medication, "x": 0.01, "font": {"size": 15, "color": INK}},
        xaxis={"title": "", "gridcolor": LINE, "showline": True, "linecolor": LINE},
        yaxis={
            "title": "Administrations per day",
            "rangemode": "tozero",
            "gridcolor": LINE,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.16, "x": 0},
        margin={"t": 44, "b": 30, "l": 56, "r": 16},
        height=420,
    )
    return figure


def _headline(
    result: MedicationForecast,
    forecast: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    horizon: int,
    level: int,
) -> list:
    total = float(forecast.sum())
    parts = [
        html.Div(
            f"Next {horizon} days", style={"color": MUTED, "fontSize": "0.85em", "letterSpacing": "0.04em"}
        ),
        html.Div(
            [
                html.Span(f"{total:,.0f}", style={"fontSize": "2.4em", "fontWeight": "700"}),
                html.Span(" administrations", style={"fontSize": "1em", "color": MUTED, "marginLeft": "8px"}),
            ]
        ),
    ]
    if level:
        low, high = result.total_bounds(total, horizon, level)
        parts.append(
            html.Div(
                f"{level}% interval on the total: {low:,.0f} to {high:,.0f} "
                f"({total / horizon:.1f}/day central)",
                style=_NOTE,
            )
        )
    else:
        parts.append(html.Div(f"{total / horizon:.1f} per day on average", style=_NOTE))
    return parts


def _trust_panel(result: MedicationForecast, level: int) -> list:
    rows: list = [
        html.Div(
            "How this number was produced",
            style={"fontWeight": "700", "marginBottom": "10px", "fontSize": "0.95em"},
        ),
        html.Div(
            f"Model: {result.model}, selected from {len(result.leaderboard)} candidates by "
            f"rolling-origin cross-validation.",
            style=_NOTE,
        ),
        html.Div(
            f"Held-out accuracy: {result.primary_metric.upper()} {result.score:.3f} "
            f"(below 1.0 means it beats an in-sample naive forecast). "
            f"Cumulative bias {result.cumulative_bias:+.1%}.",
            style=_NOTE,
        ),
    ]

    if not result.beats_benchmark:
        rows.append(
            html.Div(
                "No candidate beat the benchmark for this series, so the benchmark is serving.",
                style=_FLAG,
            )
        )

    if result.intermittent:
        rows.append(
            html.Div(
                f"Intermittent series — {result.zero_share:.0%} of days have no administrations. "
                "Scored on RMSSE because absolute error would reward forecasting zero every day.",
                style=_NOTE,
            )
        )

    if not result.native_intervals:
        rows.append(
            html.Div(
                "Croston's method has no underlying stochastic model, so it admits no analytic "
                "prediction interval. The band shown is conformal — calibrated from past "
                "forecast errors rather than derived from the model.",
                style=_FLAG,
            )
        )

    if level:
        achieved = result.coverage.get(str(level))
        if achieved is not None:
            shortfall = achieved < (level / 100.0) - 0.05
            rows.append(
                html.Div(
                    f"The {level}% band covered {achieved:.0%} of actuals on windows held out "
                    f"from its own calibration."
                    + (" It under-covers — treat it as optimistic." if shortfall else ""),
                    style=_FLAG if shortfall else _NOTE,
                )
            )

    if not result.elasticity_usable:
        rows.append(
            html.Div(
                "The comorbidity effect for this medication is not statistically distinguishable "
                "from zero, so the scenario slider deliberately does not move the forecast.",
                style=_NOTE,
            )
        )
    else:
        rows.append(
            html.Div(
                f"Comorbidity elasticity {result.elasticity_beta:+.2f} (SE {result.elasticity_se:.2f}), "
                f"estimated net of seasonality against a recent cohort mean of "
                f"{result.observed_cohort_score:.2f}.",
                style=_NOTE,
            )
        )

    rows.append(
        html.Details(
            [
                html.Summary(
                    "Model leaderboard", style={"cursor": "pointer", "color": MUTED, "fontSize": "0.85em"}
                ),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Model", style={"textAlign": "left", "padding": "4px 12px 4px 0"}),
                                    html.Th(result.primary_metric.upper(), style={"padding": "4px 12px"}),
                                    html.Th("Cumulative bias", style={"padding": "4px 12px"}),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            entry["model"],
                                            style={
                                                "padding": "3px 12px 3px 0",
                                                "fontWeight": "700" if entry["model"] == result.model else "400",
                                            },
                                        ),
                                        html.Td(f"{entry['primary']:.3f}", style={"padding": "3px 12px"}),
                                        html.Td(
                                            "n/a"
                                            if entry["cum_bias"] is None or pd.isna(entry["cum_bias"])
                                            else f"{entry['cum_bias']:+.1%}",
                                            style={"padding": "3px 12px"},
                                        ),
                                    ]
                                )
                                for entry in result.leaderboard
                            ]
                        ),
                    ],
                    style={"fontSize": "0.85em", "marginTop": "8px", "borderCollapse": "collapse"},
                ),
            ],
            style={"marginTop": "10px"},
        )
    )
    return rows


def register_callbacks(
    app,
    patients: pd.DataFrame,
    consumption: pd.DataFrame,
    cohort: pd.DataFrame,
    trained: dict[str, MedicationForecast],
    trained_at: str | None,
    freshness=None,
) -> None:
    @app.callback(
        [
            Output("forecast-chart", "figure"),
            Output("headline", "children"),
            Output("trust-panel", "children"),
            Output("scenario-note", "children"),
        ],
        [
            Input("medication-dropdown", "value"),
            Input("forecast-days", "value"),
            Input("ma-dropdown", "value"),
            Input("predicted-comorbidity-slider", "value"),
            Input("interval-level", "value"),
            Input("history-window", "value"),
        ],
    )
    def update(medication, horizon, ma_period, predicted_comorbidity, level, history_window):
        result = trained[medication]
        history = daily_counts(consumption, medication)
        if history_window:
            history = history.tail(history_window)

        dates = pd.DatetimeIndex(pd.to_datetime(result.forecast_dates[:horizon]))
        central = np.asarray(result.forecast[:horizon], dtype=float)

        adjusted = result.elasticity.adjust(
            central, predicted_comorbidity, result.observed_cohort_score
        )
        scale = np.divide(
            adjusted, central, out=np.ones_like(adjusted), where=central != 0
        ).mean()

        bounds = None
        if level:
            lower, upper = result.bands(central, level)
            bounds = (lower * scale, upper * scale)

        figure = build_figure(history, ma_period, dates, adjusted, bounds, medication)

        note_style = _NOTE if result.elasticity_usable else {**_NOTE, "fontStyle": "italic"}
        if not result.elasticity_usable:
            note = html.Div("No usable comorbidity effect for this medication — forecast unchanged.", style=note_style)
        elif abs(scale - 1.0) < 0.005:  # rounds to 1.00x -- claiming a change would mislead
            note = html.Div("At the observed cohort level.", style=note_style)
        else:
            note = html.Div(f"Scenario applied: forecast scaled {scale:.2f}x.", style=note_style)

        footer = []
        if freshness is not None and (freshness.is_stale or not freshness.config_matches):
            footer.append(html.Div(freshness.describe(), style=_FLAG))
        if trained_at:
            footer.append(
                html.Div(
                    f"Models trained {trained_at}."
                    + (f" {freshness.describe()}" if freshness is not None and not freshness.is_stale and freshness.config_matches else ""),
                    style={"fontSize": "0.8em", "color": MUTED, "marginTop": "10px"},
                )
            )

        return (
            figure,
            _headline(result, adjusted, bounds, horizon, level),
            _trust_panel(result, level) + footer,
            note,
        )
