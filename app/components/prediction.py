import numpy as np
import streamlit as st
from typing import Optional, Any
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def plot_predictions(
    y_train: Any, dates_train: Any,
    y_test: Any, y_test_pred: Any, dates_test: Any,
    y_forecast: Any, y_forecast_pred: Any, dates_forecast: Any
) -> go.Figure:
    """
    Plot three distinct regions: historical, test (actual vs. predicted), and future forecast.
    Args:
        y_train: Actual values for the historical training period (first third)
        dates_train: Dates for the training period
        y_test: Actual values for the test period (second third)
        y_test_pred: Predicted values for the test period
        dates_test: Dates for the test period
        y_forecast: Actual values for the forecast period (if available, else None)
        y_forecast_pred: Predicted values for the forecast period
        dates_forecast: Dates for the forecast period
    Returns:
        go.Figure: Plotly figure visualizing model performance across all periods
    """
    fig = go.Figure()
    # Historical (first third)
    fig.add_trace(go.Scatter(
        x=dates_train,
        y=y_train,
        mode="lines",
        name="History",
        line=dict(color="#1976d2", width=3)
    ))
    # Test: actual values (second third, light blue)
    fig.add_trace(go.Scatter(
        x=dates_test,
        y=y_test,
        mode="lines",
        name="Test Actual",
        line=dict(color="#64b5f6", width=3)
    ))
    # Test: predicted values (second third, light orange)
    fig.add_trace(go.Scatter(
        x=dates_test,
        y=y_test_pred,
        mode="lines+markers",
        name="Test Predicted",
        line=dict(color="#ffd180", width=3)
    ))
    # Forecast: predicted values (last third, orange)
    fig.add_trace(go.Scatter(
        x=dates_forecast,
        y=y_forecast_pred,
        mode="lines+markers",
        name="Future Forecast",
        line=dict(color="#fb8c00", width=3)
    ))
    # Forecast: actual values (if available)
    if y_forecast is not None:
        fig.add_trace(go.Scatter(
            x=dates_forecast,
            y=y_forecast,
            mode="lines",
            name="Future Actual",
            line=dict(color="#bdbdbd", width=2)
        ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(color="#222"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1200
    )
    return fig

def render_prediction_metrics(y_true, y_pred):
    """
    Render key regression metrics (MSE, R2, MAE) for model evaluation.
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    Displays:
        Metric cards for MSE, R², and MAE in the Streamlit dashboard
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    metrics = [
        {"label": "MSE", "value": f"{mse:.2f}"},
        {"label": "R²", "value": f"{r2:.3f}"},
        {"label": "MAE", "value": f"{mae:.2f}"}
    ]
    cols = st.columns(3)
    for idx, m in enumerate(metrics):
        with cols[idx % 3]:
            st.markdown(f"""
                <div style='background-color:#fff;border-radius:10px;border:1px solid #e0e0e0;padding:16px 8px;margin-bottom:10px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
                    <span style='font-size:1.1em;color:#222;font-weight:600;'>{m['label']}</span><br>
                    <span style='font-size:1.5em;color:#1976d2;font-weight:700;'>{m['value']}</span>
                </div>
            """, unsafe_allow_html=True)