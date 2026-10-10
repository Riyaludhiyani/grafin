"""Unit tests for Phase 3: Signed weighted propagation engine.

Includes:
- Hand-checkable deterministic test with exact arithmetic verification
- Positive path tests
- Negative path tests and sign reversal (+ * - = -, - * - = +)
- Hop decay exponential calculation (0.8^k)
- Score threshold pruning
- Maximum depth cutoff
- Cycle prevention in looping graphs
- Ambiguous sign handling
- Contested asset opposing path detection (no silent cancellation)
- Production graph propagation verification for Crude Oil shock
"""

import pytest
from src.config import GRAPH_YAML_PATH
from src.graph_engine.loader import KnowledgeGraph, load_knowledge_graph
from src.graph_engine.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraphData,
    KnowledgeGraphMetadata,
    NodeType,
    SignType,
    ConfidenceTier,
    ProvenanceType,
)
from src.graph_engine.propagator import (
    PropagationEngine,
    numeric_to_sign,
    sign_to_numeric,
)


@pytest.fixture
def synthetic_toy_graph() -> KnowledgeGraph:
    """Create a minimal synthetic graph with known analytical values for exact hand verification.

    Topology:
      node_a (origin)
        ├── [weight=0.9, sign=negative, lag=2] ──> node_b
        │                                           └── [weight=0.8, sign=negative, lag=3] ──> node_c
        └── [weight=0.7, sign=positive, lag=1] ───> node_d
                                                    └── [weight=0.6, sign=negative, lag=4] ──> node_c
    """
    metadata = KnowledgeGraphMetadata(
        name="Hand Verification Graph",
        version="1.0.0",
        geography="India",
        updated_at="2026-10-11T00:00:00Z",
    )
    nodes = [
        GraphNode(id="node_a", name="Node A", type=NodeType.MACRO_INDICATOR, sector="macro"),
        GraphNode(id="node_b", name="Node B", type=NodeType.SECTOR, sector="materials"),
        GraphNode(id="node_c", name="Node C", type=NodeType.EQUITY, sector="materials"),
        GraphNode(id="node_d", name="Node D", type=NodeType.SECTOR, sector="energy"),
    ]
    edges = [
        GraphEdge(
            source="node_a",
            target="node_b",
            sign=SignType.NEGATIVE,
            weight=0.9,
            lag_days=2,
            rationale="A hurts B directly",
            evidence_refs=["Doc Ref 1"],
        ),
        GraphEdge(
            source="node_b",
            target="node_c",
            sign=SignType.NEGATIVE,
            weight=0.8,
            lag_days=3,
            rationale="B hurts C directly",
            evidence_refs=["Doc Ref 2"],
        ),
        GraphEdge(
            source="node_a",
            target="node_d",
            sign=SignType.POSITIVE,
            weight=0.7,
            lag_days=1,
            rationale="A helps D directly",
            evidence_refs=["Doc Ref 3"],
        ),
        GraphEdge(
            source="node_d",
            target="node_c",
            sign=SignType.NEGATIVE,
            weight=0.6,
            lag_days=4,
            rationale="D hurts C directly",
            evidence_refs=["Doc Ref 4"],
        ),
    ]
    data = KnowledgeGraphData(metadata=metadata, nodes=nodes, edges=edges)
    return KnowledgeGraph(data)


def test_sign_multiplier_rules():
    """Verify core arithmetic sign rules."""
    assert sign_to_numeric("positive") == 1
    assert sign_to_numeric("negative") == -1
    assert sign_to_numeric("ambiguous") == 0

    assert numeric_to_sign(1) == "positive"
    assert numeric_to_sign(-1) == "negative"
    assert numeric_to_sign(0) == "ambiguous"


def test_hand_checked_deterministic_propagation(synthetic_toy_graph):
    """Hand-calculated analytical verification of scores, decay, sign reversal, and contested flags.

    Hand calculations for shock: origin = node_a, initial_direction = positive (+1):
    1. Path to node_b (1 hop):
       - raw_strength = 0.9
       - decayed_score = 0.9 * (0.8 ^ 1) = 0.72
       - net_sign = (+1) * (-1) = -1 ('negative')
       - cumulative_lag = 2 days

    2. Path to node_d (1 hop):
       - raw_strength = 0.7
       - decayed_score = 0.7 * (0.8 ^ 1) = 0.56
       - net_sign = (+1) * (+1) = +1 ('positive')
       - cumulative_lag = 1 day

    3. Path 1 to node_c (via node_b, 2 hops):
       - raw_strength = 0.9 * 0.8 = 0.72
       - decayed_score = 0.72 * (0.8 ^ 2) = 0.72 * 0.64 = 0.4608
       - net_sign = (+1) * (-1) * (-1) = +1 ('positive') [Sign Reversal!]
       - cumulative_lag = 2 + 3 = 5 days

    4. Path 2 to node_c (via node_d, 2 hops):
       - raw_strength = 0.7 * 0.6 = 0.42
       - decayed_score = 0.42 * (0.8 ^ 2) = 0.42 * 0.64 = 0.2688
       - net_sign = (+1) * (+1) * (-1) = -1 ('negative')
       - cumulative_lag = 1 + 4 = 5 days

    5. Consolidated impact on node_c:
       - node_c receives 1 positive path (score 0.4608) AND 1 negative path (score 0.2688).
       - Contested opposing forces detected!
       - is_contested MUST be True.
       - net_direction MUST be 'contested' (never silently netted).
       - top_score MUST be 0.4608.
       - confidence_tier MUST be 'high' (0.4608 >= 0.40).
    """
    engine = PropagationEngine(synthetic_toy_graph, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    result = engine.propagate(origin_id="node_a", initial_direction="positive")

    assert result.total_paths_found == 4

    impact_dict = {item.asset_id: item for item in result.affected_assets}

    # Verify node_b
    b_impact = impact_dict["node_b"]
    assert b_impact.net_direction == "negative"
    assert pytest.approx(b_impact.top_score, rel=1e-5) == 0.72
    assert b_impact.is_contested is False

    # Verify node_d
    d_impact = impact_dict["node_d"]
    assert d_impact.net_direction == "positive"
    assert pytest.approx(d_impact.top_score, rel=1e-5) == 0.56
    assert d_impact.is_contested is False

    # Verify node_c (contested asset with sign reversal)
    c_impact = impact_dict["node_c"]
    assert c_impact.is_contested is True
    assert c_impact.net_direction == "contested"
    assert c_impact.positive_paths_count == 1
    assert c_impact.negative_paths_count == 1
    assert pytest.approx(c_impact.top_score, rel=1e-5) == 0.4608
    assert c_impact.confidence_tier == "high"
    assert len(c_impact.all_paths) == 2

    # Verify individual paths to node_c
    path_b_to_c = [p for p in c_impact.all_paths if "node_b" in p.nodes][0]
    assert path_b_to_c.net_sign == "positive"
    assert pytest.approx(path_b_to_c.decayed_score, rel=1e-5) == 0.4608
    assert path_b_to_c.cumulative_lag_days == 5

    path_d_to_c = [p for p in c_impact.all_paths if "node_d" in p.nodes][0]
    assert path_d_to_c.net_sign == "negative"
    assert pytest.approx(path_d_to_c.decayed_score, rel=1e-5) == 0.2688
    assert path_d_to_c.cumulative_lag_days == 5


def test_sign_reversal_with_negative_origin_shock(synthetic_toy_graph):
    """Verify that a negative shock flips all downstream signs correctly."""
    engine = PropagationEngine(synthetic_toy_graph, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    result = engine.propagate(origin_id="node_a", initial_direction="negative")

    impact_dict = {item.asset_id: item for item in result.affected_assets}

    # Initial shock negative * negative edge = positive
    assert impact_dict["node_b"].net_direction == "positive"
    # Initial shock negative * positive edge = negative
    assert impact_dict["node_d"].net_direction == "negative"


def test_hop_decay_and_threshold_pruning(synthetic_toy_graph):
    """Verify that paths with decayed scores below threshold are pruned."""
    # Set threshold to 0.50.
    # Path to node_b is 0.72 (kept)
    # Path to node_d is 0.56 (kept)
    # Path 1 to node_c is 0.4608 (< 0.50 -> pruned)
    # Path 2 to node_c is 0.2688 (< 0.50 -> pruned)
    engine = PropagationEngine(synthetic_toy_graph, hop_decay=0.8, max_depth=4, min_score_threshold=0.50)
    result = engine.propagate(origin_id="node_a", initial_direction="positive")

    # node_c should not be reached at all because all paths fell below 0.50
    assert "node_c" not in [item.asset_id for item in result.affected_assets]
    assert result.total_paths_found == 2


def test_max_depth_cutoff(synthetic_toy_graph):
    """Verify that traversal does not exceed max_depth."""
    engine = PropagationEngine(synthetic_toy_graph, hop_decay=0.8, max_depth=1, min_score_threshold=0.10)
    result = engine.propagate(origin_id="node_a", initial_direction="positive")

    # Only 1-hop targets (node_b and node_d) should be present
    target_ids = [item.asset_id for item in result.affected_assets]
    assert set(target_ids) == {"node_b", "node_d"}
    assert "node_c" not in target_ids


def test_cycle_prevention():
    """Verify that cyclic graph structures do not cause infinite recursion."""
    metadata = KnowledgeGraphMetadata(
        name="Cyclic Graph",
        version="1.0.0",
        geography="India",
        updated_at="2026-10-11T00:00:00Z",
    )
    nodes = [
        GraphNode(id="node_x", name="Node X", type=NodeType.COMMODITY, sector="energy"),
        GraphNode(id="node_y", name="Node Y", type=NodeType.SECTOR, sector="energy"),
    ]
    # Mutual cycle X -> Y -> X
    edges = [
        GraphEdge(
            source="node_x",
            target="node_y",
            sign=SignType.POSITIVE,
            weight=0.9,
            lag_days=1,
            rationale="X feeds Y",
        ),
        GraphEdge(
            source="node_y",
            target="node_x",
            sign=SignType.POSITIVE,
            weight=0.9,
            lag_days=1,
            rationale="Y feedbacks to X",
        ),
    ]
    data = KnowledgeGraphData(metadata=metadata, nodes=nodes, edges=edges)
    kg = KnowledgeGraph(data)

    engine = PropagationEngine(kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    # Must terminate without infinite loop and visit node_y exactly once from node_x
    result = engine.propagate(origin_id="node_x", initial_direction="positive")

    assert len(result.affected_assets) == 1
    assert result.affected_assets[0].asset_id == "node_y"
    assert result.total_paths_found == 1


def test_ambiguous_sign_propagation():
    """Verify that an ambiguous edge turns downstream net direction ambiguous."""
    metadata = KnowledgeGraphMetadata(
        name="Ambiguous Graph",
        version="1.0.0",
        geography="India",
        updated_at="2026-10-11T00:00:00Z",
    )
    nodes = [
        GraphNode(id="node_1", name="Node 1", type=NodeType.MACRO_INDICATOR, sector="macro"),
        GraphNode(id="node_2", name="Node 2", type=NodeType.SECTOR, sector="energy"),
        GraphNode(id="node_3", name="Node 3", type=NodeType.EQUITY, sector="energy"),
    ]
    edges = [
        GraphEdge(
            source="node_1",
            target="node_2",
            sign=SignType.AMBIGUOUS,
            weight=0.8,
            lag_days=2,
            rationale="Ambiguous transmission",
        ),
        GraphEdge(
            source="node_2",
            target="node_3",
            sign=SignType.POSITIVE,
            weight=0.8,
            lag_days=1,
            rationale="Direct sector link",
        ),
    ]
    data = KnowledgeGraphData(metadata=metadata, nodes=nodes, edges=edges)
    kg = KnowledgeGraph(data)

    engine = PropagationEngine(kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    result = engine.propagate(origin_id="node_1", initial_direction="positive")

    impact_dict = {item.asset_id: item for item in result.affected_assets}
    assert impact_dict["node_2"].net_direction == "ambiguous"
    assert impact_dict["node_3"].net_direction == "ambiguous"


def test_production_graph_crude_oil_shock():
    """Verify propagation on the real production graph for a Crude Oil positive shock."""
    kg = load_knowledge_graph(GRAPH_YAML_PATH)
    engine = PropagationEngine(kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)

    result = engine.propagate(origin_id="macro_brent_crude", initial_direction="positive")

    assert result.total_paths_found > 10
    impact_dict = {item.asset_id: item for item in result.affected_assets}

    # 1. Asian Paints should be negatively impacted
    assert "equity_asianpaints" in impact_dict
    assert impact_dict["equity_asianpaints"].net_direction == "negative"

    # 2. IndiGo airline should be negatively impacted
    assert "equity_indigo" in impact_dict
    assert impact_dict["equity_indigo"].net_direction == "negative"

    # 3. Upstream oil ONGC should be positively impacted
    assert "equity_ongc" in impact_dict
    assert impact_dict["equity_ongc"].net_direction == "positive"

    # 4. Check evidence citations and rationales are preserved
    top_paints_path = impact_dict["equity_asianpaints"].top_path
    assert len(top_paints_path.evidence_refs) > 0
    assert len(top_paints_path.rationales) > 0
    assert top_paints_path.cumulative_lag_days > 0
