"""Unit tests for Phase 7: UI components, PyVis graph generation, and Plotly charts."""

import numpy as np
import pandas as pd
import pytest

from src.config import GRAPH_YAML_PATH
from src.graph_engine.loader import load_knowledge_graph
from src.graph_engine.propagator import PropagationEngine
from src.ui.charts import create_dual_asset_comparison_chart, create_empty_chart
from src.ui.components import (
    build_ranked_assets_dataframe,
    get_confidence_badge,
    get_direction_badge,
    get_validation_status_badge,
)
from src.ui.graph_viz import generate_impact_graph_html, render_graph_legend


@pytest.fixture
def loaded_kg():
    return load_knowledge_graph(GRAPH_YAML_PATH)


@pytest.fixture
def sample_propagation_result(loaded_kg):
    engine = PropagationEngine(loaded_kg, hop_decay=0.8, max_depth=3, min_score_threshold=0.10)
    return engine.propagate(origin_id="macro_brent_crude", initial_direction="positive")


def test_badge_formatters():
    """Verify HTML badge generation and distinct styling."""
    pos_badge = get_direction_badge("positive")
    neg_badge = get_direction_badge("negative")
    contested_badge = get_direction_badge("contested")
    amb_badge = get_direction_badge("ambiguous")

    assert "#10B981" in pos_badge and "POSITIVE" in pos_badge
    assert "#EF4444" in neg_badge and "NEGATIVE" in neg_badge
    assert "#C084FC" in contested_badge and "CONTESTED" in contested_badge
    assert "#F59E0B" in amb_badge and "AMBIGUOUS" in amb_badge

    high_badge = get_confidence_badge("high")
    assert "HIGH CONFIDENCE" in high_badge

    val_badge = get_validation_status_badge("supporting")
    assert "SUPPORTING" in val_badge


def test_build_ranked_assets_dataframe(sample_propagation_result):
    """Verify transformation of AssetImpact list into presentation DataFrame."""
    df = build_ranked_assets_dataframe(sample_propagation_result.affected_assets)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_propagation_result.affected_assets)
    expected_cols = [
        "Asset ID",
        "Asset Name",
        "Type",
        "Sector",
        "Ticker",
        "Direction",
        "Score",
        "Confidence",
        "Est. Lag (Days)",
        "Paths Found",
        "Top Transmission Path",
        "Flags",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify score descending order
    scores = list(df["Score"])
    assert scores == sorted(scores, reverse=True)


def test_generate_impact_graph_html(loaded_kg, sample_propagation_result):
    """Verify PyVis HTML graph generator produces valid, styled network markup."""
    html_out = generate_impact_graph_html(loaded_kg, sample_propagation_result)

    assert isinstance(html_out, str)
    assert "<html" in html_out or "<div" in html_out
    assert "vis" in html_out.lower()
    # Check that origin node is included in the rendered graph
    assert "macro_brent_crude" in html_out


def test_render_graph_legend():
    """Verify legend HTML encodes edge styles and sign colors."""
    legend = render_graph_legend()
    assert "Observed Direct Shock" in legend
    assert "Curated Structural Prior" in legend
    assert "Inferred Multi-Hop Path" in legend
    assert "Positive" in legend
    assert "Negative" in legend


def test_create_dual_asset_comparison_chart():
    """Verify Plotly comparison chart builder for normalized and dual-axis modes."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df_a = pd.DataFrame({"Close": np.linspace(100, 150, 50)}, index=dates)
    df_b = pd.DataFrame({"Close": np.linspace(200, 250, 50)}, index=dates)

    # 1. Normalized Mode
    fig_norm = create_dual_asset_comparison_chart(df_a, df_b, "Driver", "Target", normalize_base_100=True)
    assert len(fig_norm.data) == 2
    assert "Base = 100" in fig_norm.layout.title.text

    # 2. Dual Y-Axis Mode
    fig_dual = create_dual_asset_comparison_chart(df_a, df_b, "Driver", "Target", normalize_base_100=False)
    assert len(fig_dual.data) == 2
    assert fig_dual.layout.yaxis2 is not None


def test_chart_empty_data_fallback():
    """Verify chart handles empty or None dataframes with an empty chart fallback."""
    fig = create_dual_asset_comparison_chart(None, None, "Driver", "Target")
    assert len(fig.data) == 0  # No traces, fallback annotation present
