import streamlit as st
import plotly.graph_objects as go
import polars as pl


def render_ab_test(filtered_df: pl.DataFrame) -> None:
    """
    Renders a simulated A/B test comparing two campaigns using Polars and Streamlit.

    Args:
        filtered_df (pl.DataFrame): Filtered Polars DataFrame.
    """
    group_a = filtered_df.filter(pl.col('Campaign_Prospecting') == 1)
    group_b = filtered_df.filter(pl.col('Campaign_Retargeting') == 1)

    if group_a.height > 0 and group_b.height > 0:
        ctr_a = group_a['CTR'].mean()
        ctr_b = group_b['CTR'].mean()
        delta = ((ctr_b - ctr_a) / ctr_a * 100) if ctr_a != 0 else 0

        # Comparative bar chart
        fig = go.Figure(data=[
            go.Bar(name='CTR Prospecting', x=['Prospecting'], y=[ctr_a], marker_color='#1976d2'),
            go.Bar(name='CTR Retargeting', x=['Retargeting'], y=[ctr_b], marker_color='#ff9800')
        ])
        fig.update_layout(
            title="Simulated A/B Test: Prospecting vs Retargeting",
            xaxis_title="Group",
            yaxis_title="CTR",
            barmode='group',
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            font=dict(color="#222"),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig)
        
        # Display KPIs in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CTR Prospecting", f"{ctr_a:.2f}%")
        with col2:
            st.metric("CTR Retargeting", f"{ctr_b:.2f}%")
        with col3:
            st.metric("Delta", f"{delta:.2f}%")
    else:
        st.write("Insufficient data for A/B Test with current filters.")
