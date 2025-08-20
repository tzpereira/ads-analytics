"""
Module for generating synthetic (dummy) ads datasets for analytics and modeling.
Improved to simulate real-world human-like behavior.
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
        Generates a more realistic synthetic dataset using Polars.
        Returns:
            pl.DataFrame: Synthetic ads dataset.
        """
        campaigns = ['Brand Campaign', 'Prospecting', 'Retargeting']
        ad_groups = ['Search - Generic', 'Search - Brand', 'Display', 'YouTube']
        keywords_ads = ['Keyword A', 'Keyword B', 'Ad Creative 1', 'Ad Creative 2']

        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(self.n)]

        # Weekly and monthly seasonality
        day_of_week = np.array([d.weekday() for d in dates])
        month = np.array([d.month for d in dates])

        # Campaign/ad group/keyword assignment
        campaign = np.random.choice(campaigns, self.n)
        ad_group = np.random.choice(ad_groups, self.n)
        keyword_ad = np.random.choice(keywords_ads, self.n)

        # Impressions: base + weekly/monthly seasonality + campaign/ad group effect + outliers
        base_impressions = np.random.randint(100, 5000, self.n)
        weekly_season = 1 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
        monthly_season = 1 + 0.15 * np.sin(2 * np.pi * month / 12)
        campaign_effect = np.where(campaign == 'Brand Campaign', 1.2,
                            np.where(campaign == 'Retargeting', 0.8, 1.0))
        ad_group_effect = np.where(ad_group == 'Display', 1.3,
                            np.where(ad_group == 'YouTube', 1.1, 1.0))
        outlier_spikes = np.random.binomial(1, 0.01, self.n) * np.random.randint(5000, 20000, self.n)
        impressions = (base_impressions * weekly_season * monthly_season * campaign_effect * ad_group_effect + outlier_spikes).astype(int)

        # Add some missing values
        missing_mask = np.random.binomial(1, 0.01, self.n)
        impressions[missing_mask == 1] = 0

        data = {
            'Campaign': campaign,
            'Ad group': ad_group,
            'Keyword/Ad': keyword_ad,
            'Impressions': impressions,
            'Date': dates
        }

        df = pl.DataFrame(data)

        # ---- Clicks / CTR ----
        def generate_clicks(row: List) -> int:
            # Base CTR by campaign/ad group/keyword
            base_ctr = 0.04 if "Prospecting" in row[0] else 0.08
            if "Display" in row[1]:
                base_ctr *= 0.7
            if "YouTube" in row[1]:
                base_ctr *= 0.9
            if "Retargeting" in row[0]:
                base_ctr *= 1.2
            # Weekly and hourly seasonality
            day_of_week = row[4].weekday()
            hour = row[4].hour
            week_season = np.sin(2 * np.pi * day_of_week / 7) * 0.02
            hour_season = np.sin(2 * np.pi * hour / 24) * 0.03
            # Random spikes and noise
            spike = np.random.binomial(1, 0.005) * np.random.uniform(0.2, 0.5)
            noise = np.random.normal(0, 0.07)
            ctr = np.clip(base_ctr + week_season + hour_season + spike + noise, 0, 1)
            clicks = np.random.binomial(n=row[3], p=ctr)
            return clicks

        clicks = [generate_clicks(row) for row in df.iter_rows()]
        df = df.with_columns([pl.Series('Clicks', clicks)])
        df = df.with_columns([
            ((pl.col('Clicks') / pl.col('Impressions')).fill_null(0).replace([np.inf, np.nan], 0) * 100).alias('CTR')
        ])

        # ---- CPC ----
        # More variation by ad group and campaign
        avg_cpc = np.where(
            (df['Ad group'] == 'Display').to_numpy(),
            np.random.uniform(0.2, 0.7, self.n),
            np.where((df['Ad group'] == 'YouTube').to_numpy(),
                np.random.uniform(0.3, 1.0, self.n),
                np.random.uniform(0.5, 2.5, self.n))
        )
        # Add campaign effect
        avg_cpc += np.where(df['Campaign'] == 'Brand Campaign', -0.1, 0)
        avg_cpc += np.where(df['Campaign'] == 'Retargeting', 0.15, 0)
        # Add random outliers
        avg_cpc += np.random.binomial(1, 0.01, self.n) * np.random.uniform(2, 5, self.n)
        df = df.with_columns([pl.Series('Avg. CPC', avg_cpc)])

        # ---- Cost ----
        df = df.with_columns([(pl.col('Clicks') * pl.col('Avg. CPC')).alias('Cost')])

        # ---- Conversions ----
        # Conversion rate depends on campaign/ad group/keyword
        conv_rate = np.where(
            (df['Campaign'] == 'Retargeting').to_numpy(),
            np.random.uniform(0.08, 0.18, self.n),
            np.where((df['Campaign'] == 'Brand Campaign').to_numpy(),
                np.random.uniform(0.03, 0.09, self.n),
                np.random.uniform(0.015, 0.08, self.n))
        )
        # Ad group effect
        conv_rate *= np.where(df['Ad group'] == 'Display', 0.7, 1.0)
        conv_rate *= np.where(df['Ad group'] == 'YouTube', 0.8, 1.0)
        # Add random spikes
        conv_rate += np.random.binomial(1, 0.005, self.n) * np.random.uniform(0.1, 0.3, self.n)
        conversions = (df['Clicks'].to_numpy() * conv_rate).astype(int)
        df = df.with_columns([
            pl.Series('Conversion rate', conv_rate),
            pl.Series('Conversions', conversions)
        ])

        # ---- Conversion Value / ROAS ----
        # Value depends on campaign/ad group/keyword
        base_value = np.random.uniform(20, 120, self.n)
        base_value += np.where(df['Campaign'] == 'Retargeting', 10, 0)
        base_value += np.where(df['Ad group'] == 'Display', -5, 0)
        base_value += np.where(df['Ad group'] == 'YouTube', 5, 0)
        # Add random outliers
        base_value += np.random.binomial(1, 0.01, self.n) * np.random.uniform(50, 200, self.n)
        conv_value = conversions * base_value
        cost = df['Cost'].to_numpy()
        roas = np.divide(conv_value, cost, out=np.zeros_like(conv_value), where=cost!=0)
        df = df.with_columns([pl.Series('Conversion Value', conv_value), pl.Series('ROAS', roas)])

        # Fill nulls with 0
        df = df.fill_null(0)
        return df
