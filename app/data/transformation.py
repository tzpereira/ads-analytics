"""
Feature engineering module
"""
import polars as pl

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Manually one-hot encodes all categorical columns in a Polars DataFrame.
    """
    categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    for col in categorical_cols:
        unique_values = df[col].unique().to_list()
        for val in unique_values:
            new_col = f"{col}_{val}"
            df = df.with_columns([
                (pl.col(col) == val).cast(pl.Int8).alias(new_col)
            ])
        df = df.drop(col)
    return df