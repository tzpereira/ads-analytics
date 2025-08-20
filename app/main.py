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

# Load or train model
if os.path.exists(MODEL_PATH):
    model = PredictiveModel.load(MODEL_PATH)
else:
    st.warning(f"Model artifact for {model_type} not found. Training a new model...")
    # Default parameters for each model type
    default_params = {
        "Decision Tree": {"max_depth": 5},
        "XGBoost": {"max_depth": 5},
        "Random Forest": {"n_estimators": 100, "max_depth": 5},
        "LightGBM": {"n_estimators": 100},
        "CatBoost": {"iterations": 100}
    }
    params = default_params.get(model_type, {})
    X = df.drop("Conversion Value")
    y = df["Conversion Value"]
    model = PredictiveModel(model_type, params)
    model.train(X, y)
    model.save(MODEL_PATH.split('/')[-1])
    st.success(f"Model {model_type} trained and saved as {MODEL_PATH}")

# UI: Filters
filtered_df = render_filters(df)

# UI: KPIs
render_kpis(filtered_df)

# Temporal split of filtered data for robust model evaluation and forecasting
X_train_f, y_train_f, X_test_f, y_test_f, X_forecast_f = PredictiveModel.split_temporal(filtered_df, target_col="Conversion Value", date_col="Date")

# Train model using the first third of the data (historical training set)
params = {
    "Decision Tree": {"max_depth": 5},
    "XGBoost": {"max_depth": 5},
    "Random Forest": {"n_estimators": 100, "max_depth": 5},
    "LightGBM": {"n_estimators": 100},
    "CatBoost": {"iterations": 100}
}
model = PredictiveModel(model_type, params.get(model_type, {}))
model.train_temporal(X_train_f, y_train_f)

# Generate predictions for test and future (forecast) sets
predictions_test = model.predict(X_test_f)
predictions_forecast = model.forecast(X_forecast_f)

# Extract date columns for visualization
dates_train = X_train_f["Date"].to_numpy() if "Date" in X_train_f.columns else None
dates_test = X_test_f["Date"].to_numpy() if "Date" in X_test_f.columns else None
dates_forecast = X_forecast_f["Date"].to_numpy() if "Date" in X_forecast_f.columns else None

# UI: Predictions
st.plotly_chart(
    plot_predictions(
        y_train_f, dates_train,
        y_test_f, predictions_test, dates_test,
        None, predictions_forecast, dates_forecast
    ),
    use_container_width=True
)

# UI: Prediction Metrics
render_prediction_metrics(y_test_f, predictions_test)

# UI: Error Analysis
st.plotly_chart(plot_error(y_test_f, predictions_test, dates_test), use_container_width=True)

# UI: User Segmentation and AB Testing
col1, col2 = st.columns([1, 1])
with col1:
    data_segmented = segment_users(filtered_df)
    render_segmentation(data_segmented)
with col2:
    # AB Test Visualization
    render_ab_test(filtered_df)

