"""
Module for generating synthetic (dummy) ads datasets for analytics and modeling.
"""
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from typing import List

class DummyDatasetGenerator:
    """
    Class responsible for generating a synthetic (dummy) ads dataset for analytics and modeling using Polars.
    """
    def __init__(self, n: int = None, seed: int = 42):
        """
        Initializes the generator with sample size and random seed.
        Args:
            n (int): Number of rows to generate. If None, generates 90 days x 24h = 2160 rows.
            seed (int): Random seed for reproducibility.
        """
        self.n = n if n is not None else 90 * 24
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self) -> pl.DataFrame:
        """
        Generates the synthetic dataset using Polars.
        Returns:
            pl.DataFrame: Synthetic ads dataset.
        """
        campaigns = ['Brand Campaign', 'Prospecting', 'Retargeting']
        ad_groups = ['Search - Generic', 'Search - Brand', 'Display', 'YouTube']
        keywords_ads = ['Keyword A', 'Keyword B', 'Ad Creative 1', 'Ad Creative 2']

        # Generates 90 days of data, 1 record per hour
        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(self.n)]

        data = {
            'Campaign': np.random.choice(campaigns, self.n),
            'Ad group': np.random.choice(ad_groups, self.n),
            'Keyword/Ad': np.random.choice(keywords_ads, self.n),
            'Impressions': np.random.randint(100, 5000, self.n),
            'Date': dates
        }

        df = pl.DataFrame(data)

        # ---- Clicks / CTR ----
        def generate_clicks(row: List) -> int:
            base_ctr = 0.04 if "Prospecting" in row[0] else 0.08
            hour = row[4].hour
            seasonal_effect = np.sin(2 * np.pi * hour / 24) * 0.02
            noise = np.random.normal(0, 0.01)
            ctr = base_ctr + seasonal_effect + noise
            clicks = np.random.binomial(n=row[3], p=max(ctr, 0))
            return clicks

        clicks = [generate_clicks(row) for row in df.iter_rows()]
        df = df.with_columns([
            pl.Series('Clicks', clicks)
        ])
        df = df.with_columns([
            ((pl.col('Clicks') / pl.col('Impressions')).fill_null(0).replace([np.inf, np.nan], 0) * 100).alias('CTR')
        ])

        # ---- CPC ----
        avg_cpc = np.where(
            df['Ad group'].to_numpy().astype(str).astype('U').view(np.chararray).find('Display') != -1,
            np.random.uniform(0.2, 0.6, self.n),
            np.random.uniform(0.5, 2.0, self.n)
        )
        df = df.with_columns([
            pl.Series('Avg. CPC', avg_cpc)
        ])

        # ---- Cost ----
        df = df.with_columns([
            (pl.col('Clicks') * pl.col('Avg. CPC')).alias('Cost')
        ])

        # ---- Conversions ----
        conv_rate = np.where(
            np.char.find(df['Campaign'].to_numpy().astype(str), 'Retargeting') != -1,
            np.random.uniform(0.08, 0.15, self.n),
            np.random.uniform(0.02, 0.07, self.n)
        )
        conversions = (df['Clicks'].to_numpy() * conv_rate).astype(int)
        df = df.with_columns([
            pl.Series('Conversion rate', conv_rate),
            pl.Series('Conversions', conversions)
        ])

        # ---- Conversion Value / ROAS ----
        conv_value = conversions * np.random.uniform(20, 100, self.n)
        cost = df['Cost'].to_numpy()
        roas = np.divide(conv_value, cost, out=np.zeros_like(conv_value), where=cost!=0)
        df = df.with_columns([
            pl.Series('Conversion Value', conv_value),
            pl.Series('ROAS', roas)
        ])

        # Fill nulls with 0
        df = df.fill_null(0)
        return df
