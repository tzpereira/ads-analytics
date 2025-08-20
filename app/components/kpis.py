import streamlit as st
import polars as pl


def render_kpis(filtered_df: pl.DataFrame) -> None:
    """
    Renders key performance indicators (KPIs) for the filtered dataset using Streamlit.

    Args:
        filtered_df (pl.DataFrame): Filtered Polars DataFrame.
    """
    st.subheader("KPIs")

    impressions = filtered_df['Impressions'].sum()
    clicks = filtered_df['Clicks'].sum()
    ctr = (clicks / impressions * 100) if impressions > 0 else 0
    cost = filtered_df['Cost'].sum()
    avg_cpc = cost / clicks if clicks > 0 else 0
    conversions = filtered_df['Conversions'].sum()
    cvr = (conversions / clicks * 100) if clicks > 0 else 0
    cpa = (cost / conversions) if conversions > 0 else 0
    revenue = filtered_df['Conversion Value'].sum()
    roas = (revenue / cost) if cost > 0 else 0

    # Prepare KPI data for display
    kpi_data = [
        {"label": "Impressions", "value": f"{impressions:,}"},
        {"label": "Clicks", "value": f"{clicks:,}"},
        {"label": "CTR", "value": f"{ctr:.2f}%"},
        {"label": "Total Cost", "value": f"${cost:,.2f}"},
        {"label": "Avg. CPC", "value": f"${avg_cpc:.2f}"},
        {"label": "CPA", "value": f"${cpa:.2f}"},
        {"label": "Conversions", "value": f"{conversions:,}"},
        {"label": "Revenue", "value": f"${revenue:,.2f}"},
        {"label": "ROAS", "value": f"{roas:.2f}x"}
    ]

    cols = st.columns(3)
    for idx, kpi in enumerate(kpi_data):
        with cols[idx % 3]:
            st.markdown(f"""
                <div style='background-color:#fff;border-radius:10px;border:1px solid #e0e0e0;padding:16px 8px;margin-bottom:10px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
                    <span style='font-size:1.1em;color:#222;font-weight:600;'>{kpi['label']}</span><br>
                    <span style='font-size:1.5em;color:#1976d2;font-weight:700;'>{kpi['value']}</span>
                </div>
            """, unsafe_allow_html=True)
