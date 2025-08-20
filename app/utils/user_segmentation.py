"""
Module for user segmentation logic using Polars DataFrame.
"""
import polars as pl

def segment_users(data: pl.DataFrame) -> pl.DataFrame:
    """
    Segments users based on revenue (Conversion Value) using Polars.

    Args:
        data (pl.DataFrame): Project data as a Polars DataFrame.

    Returns:
        pl.DataFrame: DataFrame with user segmentation.
    """
    # Robust segmentation: create a 'segment' column based on 'Conversion Value'.
    # 'Low Value', 'Medium Value', 'High Value' are values in the 'segment' column, not column names.
    if "Conversion Value" not in data.columns:
        raise ValueError("Column 'Conversion Value' not found in data. Segmentation requires this column.")

    # Ensure 'Conversion Value' is numeric and handle nulls
    data = data.with_columns([
        pl.col("Conversion Value").cast(pl.Float64).fill_null(0).alias("Conversion Value")
    ])

    # Define the segmentation logic
    def get_segment(value: float) -> str:
        if value < 200:
            return "Low Value"
        elif 200 <= value < 500:
            return "Medium Value"
        elif value >= 500:
            return "High Value"
        else:
            return "Unknown"

    data = data.with_columns([
        pl.col("Conversion Value").map_elements(get_segment).alias("segment")
    ])
    return data
