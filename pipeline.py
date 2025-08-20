"""
Orchestration script for the Ads Analytics data pipeline.
Author: Mateus Pereira da Silva
Description: Executes all pipeline steps in a modular, reproducible way using Polars.
"""
import os
import logging
import polars as pl

from app.data.ingestion import load_data, save_gold_data
from app.data.cleaning import clean_data
from app.data.transformation import transform_features
from app.data.validation import validate_raw_schema, validate_gold_schema
from app.models.predictive_model import PredictiveModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = "app/data/bucket/gold_dataset.parquet"
MODEL_PATH = "app/models/artifacts/model.pkl"


def run_pipeline():
    logging.info("Step 1: Data Ingestion")
    df_raw = load_data()

    logging.info("Step 2: Data Validation (Raw)")
    if not validate_raw_schema(df_raw):
        logging.error("Raw data schema validation failed.")
        return

    logging.info("Step 3: Data Cleaning")
    df_clean = clean_data(df_raw)

    logging.info("Step 4: Feature Engineering / Transformation")
    df_features = transform_features(df_clean)

    logging.info("Step 5: Data Validation (Gold)")
    if not validate_gold_schema(df_features):
        logging.error("Gold data schema validation failed.")
        return

    logging.info("Step 6: Save Gold Dataset")
    os.makedirs("data", exist_ok=True)
    save_gold_data(df_features, DATA_PATH)
    logging.info(f"Gold dataset saved to {DATA_PATH}")

    logging.info("Step 7: Model Training and Forecast (Temporal Split)")
    model_type = "XGBoost"
    params = {"max_depth": 5}

    # Sort and split data
    df_sorted = df_features.sort("Date")
    n = df_sorted.height
    block_size = n // 3

    train_df = df_sorted[:block_size]
    validation_df = df_sorted[block_size:2*block_size]
    forecast_df = df_sorted[2*block_size:]

    def drop_features(data):
        return data.drop(["Conversion Value", "Date"]) if "Date" in data.columns else data.drop("Conversion Value")

    # Train only on first third
    X_train = drop_features(train_df)
    y_train = train_df["Conversion Value"]
    model = PredictiveModel(model_type, params)
    model.train(X_train, y_train)

    # Predict on validation and forecast sets
    X_validation = drop_features(validation_df)
    y_validation = validation_df["Conversion Value"]
    X_forecast = drop_features(forecast_df)
    predictions_validation = model.forecast(X_validation)
    predictions_forecast = model.forecast(X_forecast)

    # Save trained model
    model_filename = f"{model_type.replace(' ', '_').lower()}_model.pkl"
    os.makedirs("app/models/artifacts", exist_ok=True)
    model.save(model_filename)
    logging.info(f"Model saved to app/models/artifacts/{model_filename}")

    # Save gold splits
    os.makedirs("app/data/bucket", exist_ok=True)
    train_df.write_parquet("app/data/bucket/gold_train.parquet")
    validation_df.write_parquet("app/data/bucket/gold_validation.parquet")
    forecast_df.write_parquet("app/data/bucket/gold_forecast.parquet")
    logging.info("Gold splits saved to app/data/bucket/")

    # Save forecast predictions
    forecast_out = forecast_df.with_columns([
        pl.Series("Forecast", predictions_forecast)
    ])
    forecast_out.write_parquet("app/data/bucket/forecast.parquet")
    logging.info("Forecast saved to app/data/bucket/forecast.parquet")

    # Optionally save validation predictions for dashboard
    validation_out = validation_df.with_columns([
        pl.Series("Forecast", predictions_validation)
    ])
    validation_out.write_parquet("app/data/bucket/validation_forecast.parquet")
    logging.info("Validation forecast saved to app/data/bucket/validation_forecast.parquet")

    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
