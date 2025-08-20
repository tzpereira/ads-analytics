"""
Ads Analytics MVP - Streamlit Application

Author: Mateus Pereira da Silva

Description:
This application provides an interactive analytics and modeling interface for digital advertising data. It loads pre-processed gold datasets and model artifacts, enables model training and evaluation, and delivers advanced visualizations including KPIs, error analysis, user segmentation, and AB testing. Designed for rapid experimentation and robust insights, the app leverages modular components and reproducible workflows, following best practices in data science and MLOps.
"""
import streamlit as st
import polars as pl
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from components.kpis import render_kpis
from components.filters import render_filters
from components.prediction import plot_predictions, render_prediction_metrics
from components.error import plot_error
from utils.user_segmentation import segment_users
from components.ab_test import render_ab_test
from components.segmentation import render_segmentation
from models.predictive_model import PredictiveModel

GOLD_DATA_PATH = "app/data/bucket/gold_dataset.parquet"

# UI: Model type selection
model_type = st.sidebar.selectbox(
    "Select model type to load",
    ["Decision Tree", "XGBoost", "Random Forest", "LightGBM", "CatBoost"]
)
MODEL_PATH = f"app/models/artifacts/{model_type.replace(' ', '_').lower()}_model.pkl"

st.set_page_config(layout="wide")
st.title("MVP - Ads Analytics")

# Load gold data
if os.path.exists(GOLD_DATA_PATH):
    df = pl.read_parquet(GOLD_DATA_PATH)
else:
    st.error("Gold dataset not found. Please run the pipeline first.")
    st.stop()

# --- Model Parameters ---
default_params = {
    "Decision Tree": {"max_depth": 5},
    "XGBoost": {"max_depth": 5},
    "Random Forest": {"n_estimators": 100, "max_depth": 5},
    "LightGBM": {"n_estimators": 100},
    "CatBoost": {"iterations": 100}
}
params = default_params.get(model_type, {})

# --- Temporal Split (Full Data) ---
df_sorted = df.sort("Date")
n = df_sorted.height
block_size = n // 3
train_df = df_sorted[:block_size]
test_df = df_sorted[block_size:2*block_size]
forecast_df = df_sorted[2*block_size:]

# --- Prepare Features ---
def drop_features(data):
    return data.drop(["Conversion Value", "Date"]) if "Date" in data.columns else data.drop("Conversion Value")

X_train = drop_features(train_df)
y_train = train_df["Conversion Value"]
X_test = drop_features(test_df)
y_test = test_df["Conversion Value"]
X_forecast = drop_features(forecast_df)

# --- Load or Train Model ---
if os.path.exists(MODEL_PATH):
    model = PredictiveModel.load(MODEL_PATH)
else:
    st.warning(f"Model artifact for {model_type} not found. Training a new model...")
    model = PredictiveModel(model_type, params)
    model.train(X_train, y_train)
    model.save(MODEL_PATH.split('/')[-1])
    st.success(f"Model {model_type} trained and saved as {MODEL_PATH}")

# --- UI: Filters ---
filtered_df = render_filters(df)

# --- UI: KPIs ---
render_kpis(filtered_df)

# --- Temporal Split & Modeling (Filtered Data) ---
filtered_sorted = filtered_df.sort("Date")
n_f = filtered_sorted.height
block_size_f = n_f // 3

historical_f = filtered_sorted[:block_size_f]
validation_f = filtered_sorted[block_size_f:2*block_size_f]
forecast_f = filtered_sorted[2*block_size_f:]

# Train only on first third (historical)
X_train_f = drop_features(historical_f)
y_train_f = historical_f["Conversion Value"]
model = PredictiveModel(model_type, params)
model.train_temporal(X_train_f, y_train_f)

# Predict on validation set
X_validation_f = drop_features(validation_f)
y_validation_f = validation_f["Conversion Value"]
predictions_validation = model.forecast(X_validation_f)

# Retrain on 1º+2º thirds before forecasting future
X_train_forecast = drop_features(filtered_sorted[:2*block_size_f])
y_train_forecast = filtered_sorted[:2*block_size_f]["Conversion Value"]
model_forecast = PredictiveModel(model_type, params)
model_forecast.train_temporal(X_train_forecast, y_train_forecast)

X_forecast_f = drop_features(forecast_f)
predictions_forecast = model_forecast.forecast(X_forecast_f)

# Dates for visualization
dates_historical = historical_f["Date"].to_numpy() if "Date" in historical_f.columns else None
dates_validation = validation_f["Date"].to_numpy() if "Date" in validation_f.columns else None
dates_forecast = forecast_f["Date"].to_numpy() if "Date" in forecast_f.columns else None

# --- UI: Predictions ---
st.plotly_chart(
    plot_predictions(
        y_train_f, dates_historical,  # historical (real)
        y_validation_f, predictions_validation, dates_validation,  # validation (real vs predicted)
        None, predictions_forecast, dates_forecast  # forecast only
    ),
    use_container_width=True
)

render_prediction_metrics(y_validation_f, predictions_validation)

# --- UI: Error Analysis ---
st.plotly_chart(plot_error(y_validation_f, predictions_validation, dates_validation), use_container_width=True)

# --- UI: User Segmentation and AB Testing ---
col1, col2 = st.columns([1, 1])
with col1:
    data_segmented = segment_users(filtered_df)
    render_segmentation(data_segmented)
with col2:
    render_ab_test(filtered_df)
