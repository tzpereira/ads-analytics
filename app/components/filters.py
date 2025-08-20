import streamlit as st
import polars as pl


def render_filters(df: pl.DataFrame):
    """
    Renders sidebar filters for available columns using Streamlit.
    Filters the Polars DataFrame accordingly.
    
    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        
    Returns:
        pl.DataFrame: Filtered Polars DataFrame.
    """
    with st.sidebar:
        # Campaign filter
        if 'Campaign' in df.columns:
            campaign_options = df['Campaign'].unique().to_list()
            selected_campaign = st.multiselect(
                "Campaign",
                options=campaign_options,
                default=campaign_options
            )
            df = df.filter(pl.col('Campaign').is_in(selected_campaign))

        # Ad Group filter
        if 'Ad group' in df.columns:
            ad_group_options = df['Ad group'].unique().to_list()
            selected_ad_group = st.multiselect(
                "Ad Group",
                options=ad_group_options,
                default=ad_group_options
            )
            df = df.filter(pl.col('Ad group').is_in(selected_ad_group))

        # Keyword/Ad filter
        if 'Keyword/Ad' in df.columns:
            keyword_options = df['Keyword/Ad'].unique().to_list()
            selected_keyword_ad = st.multiselect(
                "Keyword/Ad",
                options=keyword_options,
                default=keyword_options
            )
            df = df.filter(pl.col('Keyword/Ad').is_in(selected_keyword_ad))

    return df

