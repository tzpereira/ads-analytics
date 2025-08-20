import polars as pl
import streamlit as st
from plotly.express import pie


def render_segmentation(data_segmented: pl.DataFrame) -> None:
    """
    Renders user segmentation table and pie chart in Streamlit.
    Args:
        data_segmented (pl.DataFrame): Polars DataFrame with segmentation results.
    """
    
    # Convert to pandas DataFrame for Plotly Express pie chart
    df_plot = data_segmented.to_pandas()
    fig = pie(df_plot, names="segment", title="User Segment Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Segmentation attribute: Revenue")
