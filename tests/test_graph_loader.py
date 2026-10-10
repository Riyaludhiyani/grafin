"""Unit tests for Phase 2: Knowledge graph YAML, loader, schema validation, and entity resolution."""

import pytest
from src.config import GRAPH_YAML_PATH
from src.graph_engine.loader import (
    GraphValidationError,
    KnowledgeGraph,
    load_knowledge_graph,
    normalize_alias,
    validate_graph_structure,
)


def test_production_graph_loads_successfully():
    """Verify that the production graph.yaml loads without errors and satisfies all constraints."""
    kg = load_knowledge_graph(GRAPH_YAML_PATH)

    # 1. Node count must be between 50 and 70 nodes
    summary = kg.get_invariants_summary()
    node_count = summary["node_count"]
    assert 50 <= node_count <= 70, f"Expected 50-70 nodes, but got {node_count}"

    # 2. NetworkX graph properties
    assert kg.graph.number_of_edges() > 40
    assert kg.graph.is_directed()

    # 3. Verify presence of all required domain nodes
    required_node_ids = [
        "macro_brent_crude",
        "currency_usdinr",
        "macro_india_cpi",
        "macro_rbi_repo_rate",
        "macro_india_10y_yield",
        "index_nifty_50",
        "index_nifty_bank",
        "index_nifty_it",
        "index_nifty_auto",
        "index_nifty_metal",
        "index_nifty_realty",
        "sector_banking",
        "sector_aviation",
        "sector_paints",
        "sector_automobiles",
        "sector_it",
        "sector_energy",
        "sector_metals",
        "sector_real_estate",
        "equity_hdfcbank",
        "equity_indigo",
        "equity_asianpaints",
        "equity_maruti",
        "equity_tcs",
        "equity_reliance",
        "equity_tatasteel",
        "equity_dlf",
    ]
    for required_id in required_node_ids:
        node = kg.get_node(required_id)
        assert node is not None, f"Required node '{required_id}' missing from graph."
        assert kg.graph.has_node(required_id)


def test_graph_edges_invariants():
    """Verify that all edges satisfy structural invariants."""
    kg = load_knowledge_graph(GRAPH_YAML_PATH)

    for edge in kg.edges_data:
        assert 0.0 < edge.weight <= 1.0, f"Invalid weight {edge.weight} on edge {edge.source}->{edge.target}"
        assert edge.lag_days >= 0, f"Invalid lag_days {edge.lag_days} on edge {edge.source}->{edge.target}"
        assert edge.sign in ("positive", "negative", "ambiguous")
        assert len(edge.rationale.strip()) > 10, f"Rationale too short for {edge.source}->{edge.target}"
        assert len(edge.evidence_refs) > 0, f"Missing evidence references for {edge.source}->{edge.target}"


def test_malformed_yaml(tmp_path):
    """Verify that malformed YAML syntax raises a descriptive GraphValidationError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("metadata:\n  name: Test\n nodes: [unclosed list", encoding="utf-8")

    with pytest.raises(GraphValidationError) as exc_info:
        load_knowledge_graph(bad_yaml)
    assert "Malformed YAML syntax" in str(exc_info.value)


def test_duplicate_node_ids():
    """Verify that duplicate node IDs are detected and rejected."""
    raw_data = {
        "metadata": {
            "name": "Test Graph",
            "version": "1.0",
            "geography": "India",
            "updated_at": "2026-10-11T00:00:00Z",
        },
        "nodes": [
            {"id": "node_dup", "name": "Node 1", "type": "commodity", "sector": "energy"},
            {"id": "node_dup", "name": "Node 2", "type": "sector", "sector": "energy"},
        ],
        "edges": [],
    }
    with pytest.raises(GraphValidationError) as exc_info:
        validate_graph_structure(raw_data)
    assert "Duplicate node ID detected" in str(exc_info.value)


def test_dangling_edge_endpoints():
    """Verify that edges pointing to nonexistent nodes are caught and rejected."""
    raw_data = {
        "metadata": {
            "name": "Test Graph",
            "version": "1.0",
            "geography": "India",
            "updated_at": "2026-10-11T00:00:00Z",
        },
        "nodes": [
            {"id": "node_a", "name": "Node A", "type": "commodity", "sector": "energy"},
        ],
        "edges": [
            {
                "source": "node_a",
                "target": "nonexistent_node",
                "sign": "positive",
                "weight": 0.8,
                "lag_days": 1,
                "rationale": "Test connection rationale",
            }
        ],
    }
    with pytest.raises(GraphValidationError) as exc_info:
        validate_graph_structure(raw_data)
    assert "Dangling edge" in str(exc_info.value)
    assert "nonexistent_node" in str(exc_info.value)


def test_self_loops_rejected():
    """Verify that an edge cannot have the same node as source and target."""
    raw_data = {
        "metadata": {
            "name": "Test Graph",
            "version": "1.0",
            "geography": "India",
            "updated_at": "2026-10-11T00:00:00Z",
        },
        "nodes": [
            {"id": "node_a", "name": "Node A", "type": "commodity", "sector": "energy"},
        ],
        "edges": [
            {
                "source": "node_a",
                "target": "node_a",
                "sign": "positive",
                "weight": 0.8,
                "lag_days": 0,
                "rationale": "Self loop connection",
            }
        ],
    }
    with pytest.raises(GraphValidationError) as exc_info:
        validate_graph_structure(raw_data)
    assert "Self-loop detected" in str(exc_info.value)


def test_alias_collisions_rejected():
    """Verify that two distinct nodes cannot claim identical aliases."""
    raw_data = {
        "metadata": {
            "name": "Test Graph",
            "version": "1.0",
            "geography": "India",
            "updated_at": "2026-10-11T00:00:00Z",
        },
        "nodes": [
            {"id": "bank_one", "name": "Bank One", "type": "equity", "sector": "banking", "aliases": ["mybank"]},
            {"id": "bank_two", "name": "Bank Two", "type": "equity", "sector": "banking", "aliases": ["mybank"]},
        ],
        "edges": [],
    }
    with pytest.raises(GraphValidationError) as exc_info:
        validate_graph_structure(raw_data)
    assert "Alias collision detected" in str(exc_info.value)
    assert "mybank" in str(exc_info.value)


def test_alias_normalization_and_entity_resolution():
    """Verify that entity resolution correctly resolves colloquial mentions to valid node IDs."""
    kg = load_knowledge_graph(GRAPH_YAML_PATH)

    assert kg.resolve_entity("Crude Oil") == "macro_brent_crude"
    assert kg.resolve_entity("brent") == "macro_brent_crude"
    assert kg.resolve_entity("petroleum") == "macro_brent_crude"
    assert kg.resolve_entity("USD/INR") == "currency_usdinr"
    assert kg.resolve_entity("dollar rupee") == "currency_usdinr"
    assert kg.resolve_entity("rupee") == "currency_usdinr"
    assert kg.resolve_entity("RBI repo rate") == "macro_rbi_repo_rate"
    assert kg.resolve_entity("repo rate") == "macro_rbi_repo_rate"
    assert kg.resolve_entity("retail inflation") == "macro_india_cpi"
    assert kg.resolve_entity("10-year G-sec yield") == "macro_india_10y_yield"
    assert kg.resolve_entity("Asian Paints") == "equity_asianpaints"
    assert kg.resolve_entity("IndiGo Airlines") == "equity_indigo"
    assert kg.resolve_entity("TCS") == "equity_tcs"
    assert kg.resolve_entity("Reliance Industries") == "equity_reliance"
    assert kg.resolve_entity("DLF") == "equity_dlf"
    assert kg.resolve_entity("Nifty Bank") == "index_nifty_bank"


def test_unknown_entities_resolve_to_none():
    """Verify that completely unknown entities return None rather than hallucinating nodes."""
    kg = load_knowledge_graph(GRAPH_YAML_PATH)

    assert kg.resolve_entity("Tesla Inc") is None
    assert kg.resolve_entity("SpaceX") is None
    assert kg.resolve_entity("Cryptocurrency Bitcoin") is None
    assert kg.resolve_entity("Random Nonsense 12345") is None
    assert kg.resolve_entity("") is None
