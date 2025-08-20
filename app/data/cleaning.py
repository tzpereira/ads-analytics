"""
Data cleaning and validation module
"""
import polars as pl

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Performs basic cleaning: removes duplicates and handles missing values.
    Args:
        df (pl.DataFrame): Raw DataFrame
    Returns:
        pl.DataFrame: Cleaned DataFrame
    """
    df = df.unique()
    df = df.fill_null(0)
    return df
