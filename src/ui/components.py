"""UI components for the Ranked Affected-Assets Table, Evidence Cards, and Status Badges."""

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.graph_engine.propagator import AssetImpact, PropagationPath, PropagationResult
from src.market.validator import MarketValidationReport


def get_direction_badge(direction: str) -> str:
    """Return formatted HTML badge for impact direction."""
    d = direction.lower()
    if d == "positive":
        return '<span style="background-color: #064E3B; color: #10B981; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #10B981;">▲ POSITIVE</span>'
    elif d == "negative":
        return '<span style="background-color: #450A0A; color: #EF4444; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #EF4444;">▼ NEGATIVE</span>'
    elif d == "contested":
        return '<span style="background-color: #3B0764; color: #C084FC; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #C084FC;">⚡ CONTESTED</span>'
    else:
        return '<span style="background-color: #451A03; color: #F59E0B; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #F59E0B;">◆ AMBIGUOUS</span>'


def get_confidence_badge(tier: str) -> str:
    """Return formatted HTML badge for confidence tier."""
    t = tier.lower()
    if t == "high":
        return '<span style="background-color: #1E293B; color: #38BDF8; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid #0284C7;">HIGH CONFIDENCE</span>'
    elif t == "medium":
        return '<span style="background-color: #1E293B; color: #FBBF24; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid #D97706;">MEDIUM</span>'
    else:
        return '<span style="background-color: #1E293B; color: #94A3B8; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid #64748B;">LOW</span>'


def get_validation_status_badge(status: str) -> str:
    """Return HTML badge for historical market validation status."""
    s = status.lower()
    if s == "supporting":
        return '<span style="background-color: #064E3B; color: #34D399; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #059669;">✔ SUPPORTING EMPIRICAL EVIDENCE</span>'
    elif s == "contradicting":
        return '<span style="background-color: #450A0A; color: #F87171; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #DC2626;">✖ CONTRADICTING EMPIRICAL EVIDENCE</span>'
    elif s == "insufficient_data":
        return '<span style="background-color: #1E293B; color: #94A3B8; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #475569;">⚠ INSUFFICIENT DATA</span>'
    else:
        return '<span style="background-color: #451A03; color: #FCD34D; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #D97706;">◆ INCONCLUSIVE NOISE BAND</span>'


def build_ranked_assets_dataframe(affected_assets: List[AssetImpact]) -> pd.DataFrame:
    """Transform list of AssetImpact objects into a clean tabular DataFrame for Streamlit."""
    records = []
    for item in affected_assets:
        top_path_str = " → ".join(item.top_path.node_names) if item.top_path else "Direct"
        warning_summary = "; ".join(item.warning_flags) if item.warning_flags else "None"

        records.append({
            "Asset ID": item.asset_id,
            "Asset Name": item.asset_name,
            "Type": item.asset_type.replace("_", " ").title(),
            "Sector": item.sector.replace("_", " ").title(),
            "Ticker": item.ticker or "Non-traded",
            "Direction": item.net_direction.upper(),
            "Score": round(item.top_score, 4),
            "Confidence": item.confidence_tier.upper(),
            "Est. Lag (Days)": item.cumulative_lag_days,
            "Paths Found": len(item.all_paths),
            "Top Transmission Path": top_path_str,
            "Flags": warning_summary,
        })

    return pd.DataFrame(records)


def render_evidence_panel(asset_impact: AssetImpact, validation_report: Optional[MarketValidationReport] = None):
    """Render comprehensive deep dive card with path decomposition, evidence citations, and market check."""
    st.markdown(f"### 🔍 Detailed Impact & Evidence: **{asset_impact.asset_name}**")

    # Header Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Net Direction:** {get_direction_badge(asset_impact.net_direction)}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Confidence:** {get_confidence_badge(asset_impact.confidence_tier)}", unsafe_allow_html=True)
    with col3:
        st.metric(label="Top Path Score", value=f"{asset_impact.top_score:.4f}")
    with col4:
        st.metric(label="Transmission Horizon", value=f"{asset_impact.cumulative_lag_days} Days")

    # Warning / Contested Callout
    if asset_impact.is_contested:
        st.error(
            f"⚡ **CONTESTED ASSET WARNING:** This asset receives both positive "
            f"({asset_impact.positive_paths_count} path) and negative "
            f"({asset_impact.negative_paths_count} path) transmission forces. "
            "These opposing channels are shown independently below and have not been artificially netted out."
        )

    st.markdown("---")

    # Transmission Paths Decomposition
    st.subheader(f"Transmission Channels ({len(asset_impact.all_paths)} Path(s) Discovered)")

    for idx, path in enumerate(asset_impact.all_paths, 1):
        with st.expander(f"Path #{idx}: {' → '.join(path.node_names)} (Score: {path.decayed_score:.4f})", expanded=(idx == 1)):
            st.markdown(f"**Path ID:** `{path.path_id}` | **Hops:** `{path.hops}` | **Direction:** {get_direction_badge(path.net_sign)} | **Est. Lag:** `{path.cumulative_lag_days} days`", unsafe_allow_html=True)

            # Hop-by-hop breakdown table
            hop_data = []
            for hop_num, edge in enumerate(path.edges, 1):
                hop_data.append({
                    "Hop": f"#{hop_num}",
                    "Source": edge.source,
                    "Target": edge.target,
                    "Link Sign": edge.sign.upper(),
                    "Strength": f"{edge.weight:.2f}",
                    "Lag": f"{edge.lag_days}d",
                    "Rationale": edge.rationale,
                })
            st.table(pd.DataFrame(hop_data))

            # Literature & Document Citations
            if path.evidence_refs:
                st.markdown("**Authoritative Documentary Evidence & Literature References:**")
                for ref in path.evidence_refs:
                    st.markdown(f"- 📄 *{ref}*")

    st.markdown("---")

    # Historical Market Validation Card
    st.subheader("Historical Market Validation Check")
    if validation_report:
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.markdown(f"**Empirical Verdict:** {get_validation_status_badge(validation_report.status)}", unsafe_allow_html=True)
            st.markdown(f"- **Evaluated Pair:** `{validation_report.source_symbol}` ➔ `{validation_report.target_symbol}`")
            st.markdown(f"- **Hypothesized Prior:** `{validation_report.prior_sign.upper()}` at Lag `{validation_report.evaluated_lag}d`")
            st.markdown(f"- **Correlation at Evaluated Lag:** `{validation_report.correlation_statistic if validation_report.correlation_statistic is not None else 'N/A'}`")
            st.markdown(f"- **Peak Observed Correlation:** `{validation_report.peak_correlation if validation_report.peak_correlation is not None else 'N/A'}` at Lag `{validation_report.peak_lag}d`")
            st.markdown(f"- **Sample Size:** `{validation_report.sample_size} aligned trading days` ({validation_report.start_date} to {validation_report.end_date})")

        with v_col2:
            st.info(f"**Statistical Assessment:**\n\n{validation_report.explanation}")
            st.caption(f"🛡️ **Empirical Caveat:** {validation_report.causation_disclaimer}")
            if validation_report.limitations:
                with st.expander("Known Empirical Limitations"):
                    for lim in validation_report.limitations:
                        st.markdown(f"- {lim}")
    else:
        st.info("Market data validation is pending or symbol is non-traded.")
