"""
Ads Analytics MVP - Streamlit Application

Author: Mateus Pereira da Silva

Description:
This application provides an interactive analytics and modeling interface for digital advertising data. It loads pre-processed gold datasets and model artifacts, enables model training and evaluation, and delivers advanced visualizations including KPIs, error analysis, user segmentation, and AB testing. Designed for rapid experimentation and robust insights, the app leverages modular components and reproducible workflows, following best practices in data science and MLOps.
"""
import streamlit as st
import polars as pl
import os

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
# Sort the full dataset by date to ensure chronological order
df_sorted = df.sort("Date")
n = df_sorted.height
block_size = n // 3
# First third: historical training set (e.g., first 30 days)
train_df = df_sorted.slice(0, block_size)
# Second third: test set (used to evaluate predictions, never seen by model during training)
test_df = df_sorted.slice(block_size, block_size)
# Last third: future/forecast set (used for future predictions)
forecast_df = df_sorted.slice(2*block_size, n - 2*block_size)

# --- Prepare Features ---
# Remove target and date columns so model cannot use them for prediction
def drop_features(data):
    return data.drop(["Conversion Value", "Date"]) if "Date" in data.columns else data.drop("Conversion Value")
X_train = drop_features(train_df)
y_train = train_df["Conversion Value"]
X_test = drop_features(test_df)
y_test = test_df["Conversion Value"]
X_forecast = drop_features(forecast_df)

# --- Load or Train Model ---
# The model is trained ONLY on the first third (first 30 days) of the data
# It never sees the test or forecast sets during training, ensuring no leakage
if os.path.exists(MODEL_PATH):
    model = PredictiveModel.load(MODEL_PATH)
else:
    st.warning(f"Model artifact for {model_type} not found. Training a new model...")
    model = PredictiveModel(model_type, params)
    model.train(X_train, y_train)
    model.save(MODEL_PATH.split('/')[-1])
    st.success(f"Model {model_type} trained and saved as {MODEL_PATH}")

# --- UI: Filters ---
# Allow user to filter the data interactively (e.g., by campaign, ad group)
filtered_df = render_filters(df)

# --- UI: KPIs ---
# Show key performance indicators for the filtered data
render_kpis(filtered_df)

# --- Temporal Split & Modeling (Filtered Data) ---
# Sort filtered data by date for correct chronological splits
filtered_sorted = filtered_df.sort("Date")
n_f = filtered_sorted.height
block_size_f = n_f // 3
# First third: historical (not used for prediction, just for context)
historical_f = filtered_sorted.slice(0, block_size_f)
# Second third: validation (used for forecast vs real comparison)
validation_f = filtered_sorted.slice(block_size_f, block_size_f)
# Third third: forecast (future, no real comparison)
forecast_f = filtered_sorted.slice(2*block_size_f, n_f - 2*block_size_f)

# Train only on first third
X_train_f = drop_features(historical_f)
y_train_f = historical_f["Conversion Value"]
# Train model ONLY on first third (historical)
# This simulates a real production forecast: model has never seen validation or forecast data
model = PredictiveModel(model_type, params)
model.train_temporal(X_train_f, y_train_f)

# Predict on validation and forecast sets
X_validation_f = drop_features(validation_f)
y_validation_f = validation_f["Conversion Value"]
X_forecast_f = drop_features(forecast_f)
# Forecast on validation and future periods using ONLY the model trained on first third
predictions_validation = model.forecast(X_validation_f)
predictions_forecast = model.forecast(X_forecast_f)

# Dates for visualization
dates_historical = historical_f["Date"].to_numpy() if "Date" in historical_f.columns else None
dates_validation = validation_f["Date"].to_numpy() if "Date" in validation_f.columns else None
dates_forecast = forecast_f["Date"].to_numpy() if "Date" in forecast_f.columns else None

# UI: Predictions
# Plot: historical (real), validation (real vs predicted), forecast (predicted only)
# Validation simulates a real blind forecast: model only knows first third
st.plotly_chart(
    plot_predictions(
        y_train_f, dates_historical,  # historical (real)
        y_validation_f, predictions_validation, dates_validation,  # validation (real vs predicted)
        None, predictions_forecast, dates_forecast  # forecast only
    ),
    use_container_width=True
)

# UI: Prediction Metrics
render_prediction_metrics(y_validation_f, predictions_validation)

# UI: Error Analysis
st.plotly_chart(plot_error(y_validation_f, predictions_validation, dates_validation), use_container_width=True)

# UI: User Segmentation and AB Testing
col1, col2 = st.columns([1, 1])
with col1:
    data_segmented = segment_users(filtered_df)
    render_segmentation(data_segmented)
with col2:
    # AB Test Visualization
    render_ab_test(filtered_df)

