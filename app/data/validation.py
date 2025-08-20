"""
Data schema validation module
"""
import polars as pl
from typing import List

def validate_raw_schema(df: pl.DataFrame) -> bool:
    """
    Validates if the DataFrame contains the expected raw columns after ingestion.
    Args:
        df (pl.DataFrame): DataFrame to validate
    Returns:
        bool: True if valid, False otherwise
    """
    expected_columns = [
        'Campaign', 'Ad group', 'Keyword/Ad', 'Impressions', 'Date'
    ]
    return all(col in df.columns for col in expected_columns)


def validate_gold_schema(df: pl.DataFrame) -> bool:
    """
    Validates if the DataFrame contains the expected engineered (gold) columns after feature engineering.
    Args:
        df (pl.DataFrame): DataFrame to validate
    Returns:
        bool: True if valid, False otherwise
    """
    expected_columns = [
        'Impressions', 'Date', 'Clicks', 'CTR',
        'Avg. CPC', 'Cost', 'Conversion rate', 'Conversions', 'Conversion Value', 'ROAS'
    ]
    
    # Only check columns that must always exist after transformation
    return all(col in df.columns for col in expected_columns)
