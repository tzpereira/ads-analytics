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
    
    # Ensure data is sorted by date
    df_features = df_features.sort("Date")
    n = df_features.height
    block_size = n // 3
    
    # Training: first third
    train_df = df_features.slice(0, block_size)
    X_train = train_df.drop("Conversion Value")
    y_train = train_df["Conversion Value"]
    
    # Forecast: last third
    forecast_df = df_features.slice(2*block_size, n - 2*block_size)
    X_forecast = forecast_df.drop("Conversion Value")
    
    # Train and forecast
    model = PredictiveModel(model_type, params)
    model.train(X_train, y_train)
    y_forecast = model.forecast(X_forecast)

    # Save trained model
    model_filename = f"{model_type.replace(' ', '_').lower()}_model.pkl"
    os.makedirs("app/models/artifacts", exist_ok=True)
    model.save(model_filename)
    logging.info(f"Model saved to app/models/artifacts/{model_filename}")

    # Save forecast predictions
    forecast_out = forecast_df.with_columns([
        pl.Series("Forecast", y_forecast)
    ])
    forecast_out.write_parquet("app/data/forecast.parquet")
    logging.info("Forecast saved to app/data/forecast.parquet")

    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
