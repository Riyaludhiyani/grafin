"""Plotly interactive stock market charts with date-range selectors and dual comparison views."""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_empty_chart(message: str = "Market data unavailable for this instrument.") -> go.Figure:
    """Return a styled empty Plotly figure with an informative message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#94A3B8"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=380,
    )
    return fig


def create_dual_asset_comparison_chart(
    df_driver: Optional[pd.DataFrame],
    df_target: Optional[pd.DataFrame],
    driver_label: str = "Driver Shock (e.g. Brent Crude)",
    target_label: str = "Affected Asset (e.g. Asian Paints)",
    normalize_base_100: bool = True,
) -> go.Figure:
    """Create an interactive Plotly comparison chart between the shock driver and affected asset.

    Args:
        df_driver: Historical price DataFrame for the driver.
        df_target: Historical price DataFrame for the affected asset.
        driver_label: Human-readable title for driver asset.
        target_label: Human-readable title for affected asset.
        normalize_base_100: If True, re-bases both assets to 100 at start date for direct return comparison.

    Returns:
        Plotly Figure configured with dark fintech styling and date-range slider.
    """
    if df_driver is None or df_driver.empty or "Close" not in df_driver.columns:
        return create_empty_chart(f"No historical price data available for driver instrument: '{driver_label}'")

    if df_target is None or df_target.empty or "Close" not in df_target.columns:
        return create_empty_chart(f"No historical price data available for target instrument: '{target_label}'")

    # Align dates
    s_drv = df_driver["Close"].dropna()
    s_tgt = df_target["Close"].dropna()

    combined = pd.DataFrame({"driver": s_drv, "target": s_tgt}).dropna()
    if len(combined) < 5:
        return create_empty_chart("Insufficient overlapping trading dates between the two instruments.")

    date_idx = combined.index

    if normalize_base_100:
        # Re-base to 100.0 at the earliest common trading date
        base_drv = float(combined["driver"].iloc[0])
        base_tgt = float(combined["target"].iloc[0])

        norm_drv = (combined["driver"] / base_drv) * 100.0
        norm_tgt = (combined["target"] / base_tgt) * 100.0

        fig = go.Figure()

        # Driver trace (Amber Gold)
        fig.add_trace(
            go.Scatter(
                x=date_idx,
                y=norm_drv,
                name=f"{driver_label} (Rebased 100)",
                line=dict(color="#F59E0B", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>" + f"{driver_label}: " + "%{y:.2f}<extra></extra>",
            )
        )

        # Target trace (Cyan Blue)
        fig.add_trace(
            go.Scatter(
                x=date_idx,
                y=norm_tgt,
                name=f"{target_label} (Rebased 100)",
                line=dict(color="#38BDF8", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>" + f"{target_label}: " + "%{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"<b>Normalized Performance Comparison (Base = 100)</b><br><sup>{driver_label} vs {target_label}</sup>",
                font=dict(color="#F8FAFC", size=15),
            ),
            yaxis=dict(
                title="Normalized Value (Base = 100)",
                gridcolor="#1E293B",
                zerolinecolor="#334155",
            ),
        )

    else:
        # Dual Y-Axis Raw Price Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Driver on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=date_idx,
                y=combined["driver"],
                name=driver_label,
                line=dict(color="#F59E0B", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>" + f"{driver_label}: " + "%{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Target on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=date_idx,
                y=combined["target"],
                name=target_label,
                line=dict(color="#38BDF8", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>" + f"{target_label}: " + "%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=dict(
                text=f"<b>Dual-Axis Price Movement</b><br><sup>{driver_label} vs {target_label}</sup>",
                font=dict(color="#F8FAFC", size=15),
            ),
            yaxis=dict(
                title=f"{driver_label} Price",
                title_font=dict(color="#F59E0B"),
                tickfont=dict(color="#F59E0B"),
                gridcolor="#1E293B",
            ),
            yaxis2=dict(
                title=f"{target_label} Price",
                title_font=dict(color="#38BDF8"),
                tickfont=dict(color="#38BDF8"),
                overlaying="y",
                side="right",
            ),
        )

    # General Dark Fintech Layout Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161A22",
        height=420,
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(14, 17, 23, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]),
                bgcolor="#1E293B",
                activecolor="#0284C7",
                font=dict(color="#F8FAFC", size=11),
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
    )

    return fig
