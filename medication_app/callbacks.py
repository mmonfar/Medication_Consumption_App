"""Callback wiring: pulls aggregation/forecasting together into a figure."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

from .forecasting import (
    comorbidity_summary,
    daily_totals,
    forecast_dates,
    moving_average,
    placeholder_forecast,
)


def build_figure(daily: pd.DataFrame, ma_period: int, dates, forecast) -> dict:
    return {
        "data": [
            go.Scatter(x=daily["Date"], y=daily["Dose"], mode="lines", name="Daily Consumption"),
            go.Scatter(
                x=daily["Date"],
                y=daily["Moving Average"],
                mode="lines",
                name=f"{ma_period}-Day MA",
            ),
            go.Scatter(
                x=dates,
                y=forecast,
                mode="lines",
                name="Forecasted Consumption",
                line=dict(dash="dash"),
            ),
        ],
        "layout": go.Layout(
            xaxis={"title": "Date"},
            yaxis={"title": "Consumption (Doses)"},
            showlegend=True,
        ),
    }


def register_callbacks(app, patients: pd.DataFrame, consumption: pd.DataFrame) -> None:
    @app.callback(
        [
            Output("medication-line-bar-chart", "figure"),
            Output("summary-message", "children"),
        ],
        [
            Input("medication-dropdown", "value"),
            Input("forecast-days", "value"),
            Input("ma-dropdown", "value"),
            Input("comorbidity-selection", "value"),
            Input("predicted-comorbidity-slider", "value"),
        ],
    )
    def update_graph(medication, forecast_days, ma_period, comorbidity_type, predicted_comorbidity):
        daily = daily_totals(consumption, medication)
        daily["Moving Average"] = moving_average(daily["Dose"], ma_period)

        comorbidity_value = comorbidity_summary(patients, comorbidity_type)
        dates = forecast_dates(daily, forecast_days)
        forecast = placeholder_forecast(daily, forecast_days, predicted_comorbidity)

        figure = build_figure(daily, ma_period, dates, forecast)
        figure["layout"].title = f"Medication Consumption and Forecast for {medication}"

        label = "Mean" if comorbidity_type == "mean" else "Median"
        summary = (
            f"Predicted required doses of {medication} for the next {forecast_days} days: "
            f"{forecast.sum():.2f} doses. "
            f"Comorbidity Metric ({label}): {comorbidity_value:.2f}"
            f" Predicted Comorbidity Score: {predicted_comorbidity:.2f}"
        )
        return figure, summary
