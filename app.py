import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Create the app
app = dash.Dash(__name__)

# Create sample data for medication consumption
num_patients = 50
patient_data = {
    'MRN': [f'P{str(i).zfill(4)}' for i in range(1, num_patients + 1)],
    'Charlson_Comorbidity_Score': np.random.randint(0, 10, num_patients)
}
df_patients = pd.DataFrame(patient_data)

# Medication Consumption Data
num_records = 300
medications = ['Omeprazole 20 mg', 'Omeprazole 40 mg', 'Bisoprolol 2.5 mg', 'Bisoprolol 5 mg', 'Meropenem 1g', 'Ciprofloxacin 500 mg']
doses = {
    'Omeprazole 20 mg': 20,
    'Omeprazole 40 mg': 40,
    'Bisoprolol 2.5 mg': 2.5,
    'Bisoprolol 5 mg': 5,
    'Meropenem 1g': 1,
    'Ciprofloxacin 500 mg': 500
}

start_date = datetime.now() - timedelta(days=30)
dates = [start_date + timedelta(days=np.random.randint(0, 30)) for _ in range(num_records)]
times = [f"{np.random.randint(0, 24):02}:{np.random.randint(0, 60):02}" for _ in range(num_records)]

med_consumption_data = {
    'MRN': np.random.choice(df_patients['MRN'], num_records),
    'Date': dates,
    'Time': times,
    'Medication': np.random.choice(medications, num_records)
}

df_med_consumption = pd.DataFrame(med_consumption_data)
df_med_consumption['Dose'] = df_med_consumption['Medication'].map(doses)

# Create a simple function to calculate moving averages
def moving_average(data, window):
    return data.rolling(window=window).mean()

# Layout of the app
app.layout = html.Div([
    html.H1("Medication Consumption and Forecast Analysis"),

    # Dropdown for selecting medication
    html.Div([
        html.Label("Select Medication:"),
        dcc.Dropdown(
            id='medication-dropdown',
            options=[{'label': med, 'value': med} for med in medications],
            value='Meropenem 1g'
        )
    ], style={'padding': '10px'}),

    # Slider for selecting forecast days
    html.Div([
        html.Label("Select Number of Forecast Days:"),
        html.Div(
            dcc.Slider(
                id='forecast-days',
                min=1,
                max=30,
                step=1,
                value=7,
                marks={i: f'{i}' for i in range(1, 31)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            style={'width': '1000px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}
        ),
    ], style={'padding': '10px'}),

    # Dropdown for selecting Moving Average period
    html.Div([
        html.Label("Select Moving Average Period:"),
        dcc.Dropdown(
            id='ma-dropdown',
            options=[
                {'label': '3 Days', 'value': 3},
                {'label': '7 Days', 'value': 7},
                {'label': '14 Days', 'value': 14},
                {'label': '30 Days', 'value': 30},
            ],
            value=3
        )
    ], style={'padding': '10px'}),

    # Dropdown for selecting Comorbidity (Mean or Median)
    html.Div([
        html.Label("Select Comorbidity Metric:"),
        dcc.Dropdown(
            id='comorbidity-selection',
            options=[
                {'label': 'Comorbidity Mean', 'value': 'mean'},
                {'label': 'Comorbidity Median', 'value': 'median'}
            ],
            value='mean'
        )
    ], style={'padding': '10px'}),

    # Slider for selecting Predicted Comorbidity Score to adjust the forecast
    html.Div([
        html.Label("Select Predicted Comorbidity Score for Forecast Period:"),
        html.Div(
            dcc.Slider(
                id='predicted-comorbidity-slider',
                min=0,
                max=10,
                step=0.1,
                value=5,
                marks={i: f'{i}' for i in range(0, 11, 1)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            style={'width': '1000px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}
        ),
    ], style={'padding': '10px'}),

    # Graph for showing the consumption and comorbidity over time
    dcc.Graph(id='medication-line-bar-chart'),

    # Display summary message with prediction results
    html.Div(id='summary-message', style={'padding': '10px'})
])

# Callback to update the graph based on user inputs
@app.callback(
    [Output('medication-line-bar-chart', 'figure'),
     Output('summary-message', 'children')],
    [Input('medication-dropdown', 'value'),
     Input('forecast-days', 'value'),
     Input('ma-dropdown', 'value'),
     Input('comorbidity-selection', 'value'),
     Input('predicted-comorbidity-slider', 'value')]  # Added input for predicted comorbidity score
)
def update_graph(medication, forecast_days, ma_period, comorbidity_type, predicted_comorbidity):
    # Filter data for the selected medication
    df_med_selected = df_med_consumption[df_med_consumption['Medication'] == medication]

    # Group by date and sum the doses
    df_med_daily = df_med_selected.groupby('Date').agg({'Dose': 'sum'}).reset_index()
    df_med_daily['Date'] = pd.to_datetime(df_med_daily['Date'])

    # Calculate moving average for the daily consumption
    df_med_daily['Moving Average'] = moving_average(df_med_daily['Dose'], ma_period)

    # Calculate comorbidity values based on selection (Mean or Median)
    if comorbidity_type == 'mean':
        comorbidity_value = df_patients['Charlson_Comorbidity_Score'].mean()
    else:
        comorbidity_value = df_patients['Charlson_Comorbidity_Score'].median()

    # Forecast the consumption for the selected forecast days using the predicted comorbidity score
    forecast_dates = [df_med_daily['Date'].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Apply the predicted comorbidity value to the forecast consumption (adjusted by predicted comorbidity)
    forecast_consumption = np.random.uniform(df_med_daily['Dose'].mean(), df_med_daily['Dose'].max(), forecast_days)
    forecast_consumption_adjusted = forecast_consumption * (predicted_comorbidity / 5)  # Adjust with predicted comorbidity factor (scaled)

    # Create the plot
    trace1 = go.Scatter(x=df_med_daily['Date'], y=df_med_daily['Dose'], mode='lines', name='Daily Consumption')
    trace2 = go.Scatter(x=df_med_daily['Date'], y=df_med_daily['Moving Average'], mode='lines', name=f'{ma_period}-Day MA')

    forecast_line = go.Scatter(x=forecast_dates, y=forecast_consumption_adjusted, mode='lines', name='Forecasted Consumption', line=dict(dash='dash'))

    layout = go.Layout(
        title=f"Medication Consumption and Forecast for {medication}",
        xaxis={'title': 'Date'},
        yaxis={'title': 'Consumption (Doses)'},
        showlegend=True
    )

    # Constructing the figure
    figure = {
        'data': [trace1, trace2, forecast_line],
        'layout': layout
    }

    # Summary message with prediction result and comorbidity value
    summary_message = f"Predicted required doses of {medication} for the next {forecast_days} days: {forecast_consumption_adjusted.sum():.2f} doses. "
    summary_message += f"Comorbidity Metric ({'Mean' if comorbidity_type == 'mean' else 'Median'}): {comorbidity_value:.2f}"
    summary_message += f" Predicted Comorbidity Score: {predicted_comorbidity:.2f}"

    return figure, summary_message


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8501)
