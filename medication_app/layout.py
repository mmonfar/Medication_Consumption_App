"""Dash layout: a numbered, left-to-right workflow.

The prototype stacked five unlabelled controls above a chart, which gave no
sense of order or consequence -- nothing said which control mattered, or that
the comorbidity slider is a hypothetical rather than an observation. The layout
is now an explicit three-step path down the left column (choose, horizon,
scenario) with results on the right, so the question being asked is readable
before the answer is.

Presentation only: no data access, no forecasting.
"""

from __future__ import annotations

from dash import dcc, html

from .config import (
    DEFAULT_FORECAST_DAYS,
    DEFAULT_MA_PERIOD,
    DEFAULT_MEDICATION,
    MA_PERIODS,
    MAX_HORIZON,
    MEDICATIONS,
)

# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #

INK = "#1c2530"
MUTED = "#5b6672"
LINE = "#dde3ea"
ACCENT = "#1f6f8b"
WARN = "#8a4b00"
CANVAS = "#f6f8fa"

_PAGE = {
    "fontFamily": "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
    "color": INK,
    "backgroundColor": CANVAS,
    "minHeight": "100vh",
    "padding": "24px 32px 48px",
}
_CARD = {
    "backgroundColor": "white",
    "border": f"1px solid {LINE}",
    "borderRadius": "10px",
    "padding": "18px 20px",
    "marginBottom": "16px",
}
_STEP_NUMBER = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "24px",
    "height": "24px",
    "borderRadius": "50%",
    "backgroundColor": ACCENT,
    "color": "white",
    "fontSize": "0.8em",
    "fontWeight": "700",
    "marginRight": "10px",
    "flexShrink": "0",
}
_STEP_TITLE = {"display": "flex", "alignItems": "center", "marginBottom": "4px"}
_HINT = {"color": MUTED, "fontSize": "0.85em", "margin": "0 0 14px 34px"}


def _step(number: int, title: str, hint: str, control) -> html.Div:
    """One numbered step: what to do, why it matters, then the control."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(str(number), style=_STEP_NUMBER),
                    html.Strong(title, style={"fontSize": "0.98em"}),
                ],
                style=_STEP_TITLE,
            ),
            html.P(hint, style=_HINT),
            html.Div(control, style={"marginLeft": "34px"}),
        ],
        style=_CARD,
    )


def build_layout(cohort_score: float) -> html.Div:
    return html.Div(
        [
            # ---------------------------------------------------------------- #
            html.Div(
                [
                    html.H1(
                        "Medication consumption forecast",
                        style={"margin": "0 0 4px", "fontSize": "1.6em"},
                    ),
                    html.P(
                        [
                            "A model is chosen per medication by rolling-origin cross-validation. ",
                            html.Strong("Synthetic data — not for clinical use."),
                        ],
                        style={"color": MUTED, "margin": "0"},
                    ),
                ],
                style={"marginBottom": "22px"},
            ),
            # ---------------------------------------------------------------- #
            html.Div(
                [
                    # Left: the three decisions, in order.
                    html.Div(
                        [
                            _step(
                                1,
                                "Choose a medication",
                                "Each has its own fitted model — pick one to see which, and how well it scored.",
                                dcc.Dropdown(
                                    id="medication-dropdown",
                                    options=[{"label": m, "value": m} for m in MEDICATIONS],
                                    value=DEFAULT_MEDICATION,
                                    clearable=False,
                                ),
                            ),
                            _step(
                                2,
                                "Set the forecast horizon",
                                "How many days ahead to project. Intervals widen with horizon.",
                                dcc.Slider(
                                    id="forecast-days",
                                    min=1,
                                    max=MAX_HORIZON,
                                    step=1,
                                    value=DEFAULT_FORECAST_DAYS,
                                    marks={i: str(i) for i in (1, 7, 14, 21, MAX_HORIZON)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ),
                            _step(
                                3,
                                "Test a comorbidity scenario",
                                f"A hypothetical, not a measurement. The recent cohort mean is "
                                f"{cohort_score:.1f} — move this to ask 'what if patients were sicker?'.",
                                html.Div(
                                    [
                                        dcc.Slider(
                                            id="predicted-comorbidity-slider",
                                            min=0,
                                            max=10,
                                            step=0.1,
                                            value=round(cohort_score, 1),
                                            marks={i: str(i) for i in range(0, 11, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                        html.Div(id="scenario-note", style={"marginTop": "8px"}),
                                    ]
                                ),
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        "Chart options",
                                        style={"cursor": "pointer", "color": MUTED, "fontSize": "0.9em"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Moving average window",
                                                style={"fontSize": "0.85em", "color": MUTED},
                                            ),
                                            dcc.Dropdown(
                                                id="ma-dropdown",
                                                options=[
                                                    {"label": f"{p} days", "value": p} for p in MA_PERIODS
                                                ],
                                                value=DEFAULT_MA_PERIOD,
                                                clearable=False,
                                            ),
                                            html.Label(
                                                "Prediction interval",
                                                style={
                                                    "fontSize": "0.85em",
                                                    "color": MUTED,
                                                    "marginTop": "10px",
                                                    "display": "block",
                                                },
                                            ),
                                            dcc.RadioItems(
                                                id="interval-level",
                                                options=[
                                                    {"label": " 80%", "value": 80},
                                                    {"label": " 95%", "value": 95},
                                                    {"label": " none", "value": 0},
                                                ],
                                                value=95,
                                                inline=True,
                                                labelStyle={"marginRight": "14px"},
                                            ),
                                            html.Label(
                                                "History shown",
                                                style={
                                                    "fontSize": "0.85em",
                                                    "color": MUTED,
                                                    "marginTop": "10px",
                                                    "display": "block",
                                                },
                                            ),
                                            dcc.RadioItems(
                                                id="history-window",
                                                options=[
                                                    {"label": " 90 days", "value": 90},
                                                    {"label": " 1 year", "value": 365},
                                                    {"label": " all", "value": 0},
                                                ],
                                                value=90,
                                                inline=True,
                                                labelStyle={"marginRight": "14px"},
                                            ),
                                        ],
                                        style={"marginTop": "12px"},
                                    ),
                                ],
                                style=_CARD,
                            ),
                        ],
                        style={"flex": "0 0 340px", "minWidth": "300px"},
                    ),
                    # Right: the answer.
                    html.Div(
                        [
                            html.Div(id="headline", style={**_CARD, "marginBottom": "16px"}),
                            html.Div(
                                dcc.Graph(id="forecast-chart", config={"displayModeBar": False}),
                                style={**_CARD, "padding": "8px"},
                            ),
                            html.Div(id="trust-panel", style=_CARD),
                        ],
                        style={"flex": "1 1 640px", "minWidth": "420px"},
                    ),
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "alignItems": "flex-start"},
            ),
            html.Div(
                "Research and demonstration software. Not a medical device and not intended for "
                "clinical decision-making, diagnosis or treatment. Provided \"as is\", without "
                "warranty of any kind; the author accepts no liability for any use. Uses synthetic "
                "data only.",
                style={"fontSize": "11px", "color": "#888", "marginTop": "16px"},
            ),
        ],
        style=_PAGE,
    )
