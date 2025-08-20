import plotly.graph_objects as go
import numpy as np
from typing import Any


def plot_error(y_true: Any, y_pred: Any, dates: Any = None) -> go.Figure:
    """
    Plots prediction error as a bar chart using Plotly. Compatible with Polars Series, numpy arrays, or lists.
    Args:
        y_true: True target values (Polars Series, numpy array, or list).
        y_pred: Predicted values (Polars Series, numpy array, or list).
        dates: Optional sequence of datetime objects for x-axis.
    Returns:
        go.Figure: Plotly figure object.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = y_true - y_pred
    if dates is not None:
        x_axis = np.array(dates)
        x_title = "Date"
    else:
        x_axis = np.arange(len(error))
        x_title = "Index"

    fig_error = go.Figure()
    fig_error.add_trace(go.Bar(
        x=x_axis,
        y=error,
        name="Error",
        marker_color="#fb8c00"
    ))
    fig_error.update_layout(
        xaxis_title=x_title,
        yaxis_title="Error",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(color="#222"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1200
    )
    return fig_error
