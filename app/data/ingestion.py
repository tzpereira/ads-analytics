"""
Data ingestion module
"""
import polars as pl
from app.data.dummy_dataset_generator import DummyDatasetGenerator


def load_data(source=None):
    """
    Loads data from a specified source or generates synthetic data.
    Args:
        source (str): Path to data file. If None, generates synthetic data.
    Returns:
        pl.DataFrame: Loaded data as a Polars DataFrame.
    """
    if source:
        return pl.read_csv(source)
    
    generator = DummyDatasetGenerator()
    return generator.generate()


def save_gold_data(df: pl.DataFrame, path: str) -> None:
    """
    Saves the treated (gold) dataset to Parquet format using Polars.
    Args:
        df (pl.DataFrame): Treated (gold) dataset.
        path (str): Path to save the Parquet file.
    """
    df.write_parquet(path)
