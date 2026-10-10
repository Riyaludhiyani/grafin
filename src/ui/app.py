"""Autonomous Financial Event Market Impact Graph - Interactive Streamlit Dashboard.

Explainable decision-support dashboard connecting financial news to a curated Indian
knowledge graph, tracing signed transmission paths, and validating against historical market data.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repository root is in sys.path for Streamlit Cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_HOP_DECAY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SCORE_THRESHOLD,
    DISCLAIMER_TEXT,
    GRAPH_YAML_PATH,
    SAMPLE_EVENTS_PATH,
)
from src.extraction.deterministic import (
    DeterministicExtractor,
    get_curated_event_by_id,
    load_curated_events,
)
from src.extraction.llm_extractor import classify_event
from src.graph_engine.loader import KnowledgeGraph, load_knowledge_graph
from src.graph_engine.propagator import AssetImpact, PropagationEngine, PropagationResult
from src.market.downloader import MarketDataDownloader
from src.market.validator import MarketValidator
from src.ui.charts import create_dual_asset_comparison_chart
from src.ui.components import (
    build_ranked_assets_dataframe,
    get_confidence_badge,
    get_direction_badge,
    render_evidence_panel,
)
from src.ui.graph_viz import generate_impact_graph_html, render_graph_legend


# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Fintech CSS Styling
st.markdown("""
<style>
    /* Dark fintech aesthetic */
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-card {
        background-color: #1A1D24;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .disclaimer-banner {
        background: linear-gradient(90deg, #3B0764 0%, #1E1B4B 100%);
        border: 1px solid #8B5CF6;
        color: #F8FAFC;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_knowledge_graph() -> KnowledgeGraph:
    """Load and cache the in-memory NetworkX knowledge graph."""
    return load_knowledge_graph(GRAPH_YAML_PATH)


@st.cache_resource
def get_cached_market_services():
    """Initialize cached market downloader and validator instances."""
    downloader = MarketDataDownloader()
    validator = MarketValidator(downloader=downloader)
    return downloader, validator


# Load Resources
kg = get_cached_knowledge_graph()
downloader, validator = get_cached_market_services()
curated_events = load_curated_events(SAMPLE_EVENTS_PATH)


# Persistent Mandatory Disclaimer Banner
st.markdown(
    f'<div class="disclaimer-banner">⚠️ <b>MANDATORY REGULATORY DISCLAIMER:</b> {DISCLAIMER_TEXT} '
    'This decision-support prototype illustrates hypothesized structural transmission channels and historical co-movements; it is not a predictive trading engine.</div>',
    unsafe_allow_html=True,
)

# App Header
st.title(f"🏛️ {APP_TITLE}")
st.caption(f"{APP_SUBTITLE} • Graph Invariants: {kg.graph.number_of_nodes()} Nodes, {kg.graph.number_of_edges()} Signed Edges")

st.markdown("---")


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("🕹️ Event & Propagation Controls")

# Event Selector
event_options = [f"{e.id}: {e.title[:65]}..." for e in curated_events]
event_options.append("Custom News Article Input")

selected_option = st.sidebar.selectbox("Select Financial News Event", event_options, index=0)

is_custom = selected_option == "Custom News Article Input"

if is_custom:
    custom_headline = st.sidebar.text_input("Headline", value="Brent crude jumps 7% amid renewed Middle East supply tensions")
    custom_body = st.sidebar.text_area(
        "Article Text",
        value="Crude oil surged sharply in international trading today, elevating input cost worries for Asian Paints and IndiGo airlines while boosting upstream explorer ONGC.",
        height=140,
    )
    active_event_id = None
    headline_text = custom_headline
    body_text = custom_body
else:
    selected_idx = event_options.index(selected_option)
    active_curated = curated_events[selected_idx]
    active_event_id = active_curated.id
    headline_text = active_curated.title
    body_text = active_curated.body

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")

# Propagation Hyperparameters
max_depth = st.sidebar.slider("Max Propagation Depth (Hops)", min_value=1, max_value=4, value=DEFAULT_MAX_DEPTH, step=1)
hop_decay = st.sidebar.slider("Hop Decay Factor", min_value=0.50, max_value=0.95, value=DEFAULT_HOP_DECAY, step=0.05)
min_score = st.sidebar.slider("Min Score Threshold", min_value=0.05, max_value=0.30, value=DEFAULT_MIN_SCORE_THRESHOLD, step=0.01)

run_market_check = st.sidebar.checkbox("Run Empirical Market Data Validation", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("🛡️ Autonomous Financial Event Market Impact Graph\n*Built for Explainable Decision Support*")


# ==============================================================================
# SECTION 1: STRUCTURED EVENT EXTRACTION
# ==============================================================================
st.subheader("📰 Step 1: Structured Financial News Event Extraction")

with st.spinner("Extracting and resolving structured event..."):
    extracted_event = classify_event(
        text=f"{headline_text}. {body_text}",
        event_id=active_event_id,
        knowledge_graph=kg,
    )

meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

with meta_col1:
    st.markdown(f"**Event Type:** `{extracted_event.event_type}`")
    st.markdown(f"**Status:** `{extracted_event.status.value.upper()}`")
with meta_col2:
    st.markdown(f"**Primary Vector:** {get_direction_badge(extracted_event.direction.value)}", unsafe_allow_html=True)
    mag_str = f"{extracted_event.magnitude.value.upper()} ({extracted_event.magnitude_numeric} {extracted_event.magnitude_unit or ''})".strip()
    st.markdown(f"**Magnitude:** `{mag_str}`")
with meta_col3:
    st.markdown(f"**Geography:** `{extracted_event.geography}`")
    st.markdown(f"**Published:** `{extracted_event.timestamp}`")
with meta_col4:
    origin_name = extracted_event.primary_entity_name
    st.markdown(f"**Primary Origin Node:** ⭐ `{origin_name}` (`{extracted_event.primary_entity_id}`)")
    st.markdown(f"**Source:** *{extracted_event.source}*")

# Evidence snippets
with st.expander("📄 Verbatim Supporting Evidence Quotes from Article", expanded=True):
    for snippet in extracted_event.evidence_snippets:
        st.markdown(f"> *\"{snippet}\"*")

# Collapsible Raw Extraction JSON
with st.expander("🛠️ View Validated Extracted Event JSON", expanded=False):
    st.json(extracted_event.model_dump())

st.markdown("---")


# ==============================================================================
# SECTION 2: GRAPH PROPAGATION & INTERACTIVE PYVIS VISUALIZATION
# ==============================================================================
st.subheader("🌐 Step 2: Signed Weighted Impact Graph Visualization")

# Run Propagation Engine
engine = PropagationEngine(
    knowledge_graph=kg,
    hop_decay=hop_decay,
    max_depth=max_depth,
    min_score_threshold=min_score,
)

propagation_result = engine.propagate(
    origin_id=extracted_event.primary_entity_id,
    initial_direction=extracted_event.direction.value,
)

# Render PyVis Graph
graph_html = generate_impact_graph_html(
    kg=kg,
    propagation_result=propagation_result,
    height="580px",
)

components.html(graph_html, height=600, scrolling=False)

# Visual Legend
st.markdown(render_graph_legend(), unsafe_allow_html=True)

st.markdown("---")


# ==============================================================================
# SECTION 3: RANKED AFFECTED-ASSETS TABLE
# ==============================================================================
st.subheader(f"📊 Step 3: Ranked Affected Assets ({len(propagation_result.affected_assets)} Reached)")

if not propagation_result.affected_assets:
    st.warning("No assets reached within the current depth budget and score threshold. Try increasing max depth or lowering threshold.")
else:
    df_ranked = build_ranked_assets_dataframe(propagation_result.affected_assets)

    # Sector Filter
    available_sectors = ["All"] + sorted(list(df_ranked["Sector"].unique()))
    chosen_sector = st.selectbox("Filter by Sector", available_sectors, index=0)

    if chosen_sector != "All":
        filtered_df = df_ranked[df_ranked["Sector"] == chosen_sector]
    else:
        filtered_df = df_ranked

    st.dataframe(
        filtered_df,
        column_config={
            "Score": st.column_config.NumberColumn("Decayed Score", format="%.4f"),
            "Est. Lag (Days)": st.column_config.NumberColumn("Est. Lag (Days)", format="%d"),
        },
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")


# ==============================================================================
# SECTION 4: DEEP DIVE EVIDENCE & MARKET DATA VALIDATION
# ==============================================================================
st.subheader("🔬 Step 4: Asset Deep Dive, Path Decomposition & Statistical Validation")

if propagation_result.affected_assets:
    asset_select_options = [f"{a.asset_name} ({a.asset_id})" for a in propagation_result.affected_assets]
    selected_asset_str = st.selectbox("Choose an asset to inspect transmission channels:", asset_select_options, index=0)

    selected_asset_idx = asset_select_options.index(selected_asset_str)
    selected_impact: AssetImpact = propagation_result.affected_assets[selected_asset_idx]

    # Run Market Validation if requested
    validation_report = None
    if run_market_check and selected_impact.ticker:
        driver_ticker = kg.get_ticker(extracted_event.primary_entity_id)
        if driver_ticker:
            with st.spinner(f"Computing historical market correlation between {driver_ticker} and {selected_impact.ticker}..."):
                validation_report = validator.validate_relationship(
                    source_symbol=driver_ticker,
                    target_symbol=selected_impact.ticker,
                    prior_sign=selected_impact.top_path.net_sign,
                    expected_lag_days=selected_impact.top_path.cumulative_lag_days,
                )

    render_evidence_panel(selected_impact, validation_report=validation_report)

    st.markdown("---")

    # ==============================================================================
    # SECTION 5: PLOTLY HISTORICAL MARKET TIME SERIES
    # ==============================================================================
    st.subheader("📈 Step 5: Historical Market Price Dynamics")

    driver_ticker = kg.get_ticker(extracted_event.primary_entity_id)
    target_ticker = selected_impact.ticker

    if not driver_ticker or not target_ticker:
        st.info(
            f"Historical market chart unavailable: Either origin shock ({extracted_event.primary_entity_name}) "
            f"or target asset ({selected_impact.asset_name}) is a non-traded structural index without daily ticker data."
        )
    else:
        chart_col1, chart_col2 = st.columns([3, 1])
        with chart_col2:
            chart_mode = st.radio(
                "Display Mode",
                ["Normalized Performance (Base = 100)", "Dual Y-Axis (Raw Prices)"],
                index=0,
            )
            is_norm = (chart_mode == "Normalized Performance (Base = 100)")

        df_driver = downloader.get_historical_prices(driver_ticker)
        df_target = downloader.get_historical_prices(target_ticker)

        fig = create_dual_asset_comparison_chart(
            df_driver=df_driver,
            df_target=df_target,
            driver_label=f"{extracted_event.primary_entity_name} ({driver_ticker})",
            target_label=f"{selected_impact.asset_name} ({target_ticker})",
            normalize_base_100=is_norm,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📊 Data Source: Yahoo Finance daily closes cached locally. "
            "Note: Historical observations reflect past co-movements and are never guarantees of future performance."
        )

# ==============================================================================
# SECTION 6: GROUNDED DECISION-SUPPORT CHATBOT & COUNTERFACTUAL SIMULATOR
# ==============================================================================
st.markdown("---")
st.subheader("💬 Step 6: Grounded Decision-Support Chatbot & What-If Simulator")
st.caption(
    "Answers strictly cite internal Path IDs and Edge IDs, labeling claims as **[Observed Fact]**, "
    "**[Curated Prior]**, or **[Inferred Impact]**. The assistant refuses to invent stock price predictions or price targets."
)

from src.explainer.chatbot import GroundedChatbot
chatbot = GroundedChatbot(knowledge_graph=kg)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Quick Suggestion Buttons
st.markdown("**Quick Inquiries & Counterfactuals:**")
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

suggested_query = None
with btn_col1:
    if st.button("✈️ Why crude affects aviation?"):
        suggested_query = "Why might crude affect aviation?"
with btn_col2:
    if st.button("⚡ Why is Bank Nifty contested?"):
        suggested_query = "Why is Bank Nifty contested?"
with btn_col3:
    if st.button("🔄 Reverse direction (What-If)?"):
        suggested_query = "What if the event direction is reversed (counterfactual)?"
with btn_col4:
    if st.button("🎨 Evidence for Asian Paints?"):
        suggested_query = "What evidence supports Asian Paints?"

user_query = st.text_input(
    "Ask a decision-support question regarding the active event, paths, or counterfactuals:",
    value=suggested_query or "",
    placeholder="e.g. Why might crude affect aviation? What if direction reverses?",
)

if st.button("Submit Question", type="primary") or suggested_query:
    active_q = user_query if user_query else suggested_query
    if active_q:
        v_reports = {}
        if validation_report and selected_impact.ticker:
            v_reports[selected_impact.ticker] = validation_report

        with st.spinner("Synthesizing grounded evidence and path provenance..."):
            ans = chatbot.ask(
                query=active_q,
                event=extracted_event,
                result=propagation_result,
                validation_reports=v_reports,
            )
            st.session_state.chat_history.append((active_q, ans))

# Display Chat History (most recent on top)
if st.session_state.chat_history:
    for q, a in reversed(st.session_state.chat_history[-5:]):
        with st.chat_message("user"):
            st.markdown(f"**{q}**")
        with st.chat_message("assistant"):
            st.markdown(a)

