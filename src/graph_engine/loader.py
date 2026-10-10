"""Knowledge graph loader, validator, NetworkX builder, and entity resolution module."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import yaml
from pydantic import ValidationError

from src.graph_engine.models import (
    GraphEdge,
    GraphNode,
    KnowledgeGraphData,
)


class GraphValidationError(Exception):
    """Raised when the knowledge graph definition fails structural or schema validation."""
    pass


def normalize_alias(text: str) -> str:
    """Normalize alias strings for robust, deterministic entity matching.

    Lowercases, removes special punctuation, and normalizes whitespaces.
    Example: 'USD / INR' -> 'usd inr', 'IndiGo Airlines!' -> 'indigo airlines'
    """
    if not text:
        return ""
    # Replace common symbols like / and - with spaces
    cleaned = re.sub(r"[/_-]", " ", text.lower())
    # Remove remaining punctuation
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    # Collapse multiple whitespace characters into single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class KnowledgeGraph:
    """In-memory validated NetworkX knowledge graph with entity resolution."""

    def __init__(self, data: KnowledgeGraphData):
        self.metadata = data.metadata
        self.nodes_data: Dict[str, GraphNode] = {node.id: node for node in data.nodes}
        self.edges_data: List[GraphEdge] = data.edges
        self.graph: nx.DiGraph = nx.DiGraph()
        self.alias_lookup: Dict[str, str] = {}
        self.ticker_lookup: Dict[str, str] = {}

        self._build_networkx_graph()
        self._build_alias_index()

    def _build_networkx_graph(self) -> None:
        """Construct the NetworkX DiGraph instance from validated node and edge data."""
        for node in self.nodes_data.values():
            self.graph.add_node(
                node.id,
                name=node.name,
                type=node.type.value,
                sector=node.sector,
                ticker=node.ticker,
                aliases=node.aliases,
                description=node.description,
            )

        for edge in self.edges_data:
            self.graph.add_edge(
                edge.source,
                edge.target,
                sign=edge.sign.value,
                weight=edge.weight,
                lag_days=edge.lag_days,
                confidence=edge.confidence.value,
                provenance=edge.provenance.value,
                rationale=edge.rationale,
                evidence_refs=edge.evidence_refs,
            )

    def _build_alias_index(self) -> None:
        """Index all node IDs, names, and aliases into normalized lookup dictionary."""
        for node_id, node in self.nodes_data.items():
            # Index node ID
            self.alias_lookup[normalize_alias(node_id)] = node_id
            # Index official name
            self.alias_lookup[normalize_alias(node.name)] = node_id

            # Index ticker if available
            if node.ticker:
                self.ticker_lookup[node_id] = node.ticker
                clean_ticker = node.ticker.replace(".NS", "").replace("^", "").replace("=F", "").replace("=X", "")
                self.alias_lookup[normalize_alias(clean_ticker)] = node_id
                self.alias_lookup[normalize_alias(node.ticker)] = node_id

            # Index custom aliases
            for alias in node.aliases:
                norm = normalize_alias(alias)
                if norm:
                    self.alias_lookup[norm] = node_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve node specification by node ID."""
        return self.nodes_data.get(node_id)

    def get_ticker(self, node_id: str) -> Optional[str]:
        """Retrieve Yahoo Finance ticker for a node if available."""
        node = self.nodes_data.get(node_id)
        return node.ticker if node else None

    def resolve_entity(self, text: str) -> Optional[str]:
        """Resolve entity mention or alias to a valid graph node ID.

        Returns:
            Resolved node ID string, or None if no entity is matched.
        """
        if not text:
            return None

        norm_text = normalize_alias(text)
        if not norm_text:
            return None

        # 1. Direct match in alias index
        if norm_text in self.alias_lookup:
            return self.alias_lookup[norm_text]

        # 2. Substring matching for multi-word or compound queries
        for alias, node_id in self.alias_lookup.items():
            if len(alias) >= 4:
                # Full word boundary match
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, norm_text):
                    return node_id

        return None

    def get_invariants_summary(self) -> dict:
        """Provide audit metrics on graph invariants."""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "connected_components_count": nx.number_weakly_connected_components(self.graph),
            "ticker_count": len([n for n in self.nodes_data.values() if n.ticker is not None]),
        }


def validate_graph_structure(raw_data: dict) -> KnowledgeGraphData:
    """Validate raw dictionary against schema and strict graph invariants.

    Checks:
    - Schema conformity via Pydantic
    - Duplicate node ID detection
    - Dangling edge endpoints (source or target not found)
    - Alias collisions across distinct nodes
    """
    if not isinstance(raw_data, dict):
        raise GraphValidationError("Graph definition must be a YAML mapping (dictionary).")

    # 1. Pydantic schema validation
    try:
        validated_data = KnowledgeGraphData.model_validate(raw_data)
    except ValidationError as e:
        raise GraphValidationError(f"Knowledge graph schema validation failed:\n{e}") from e

    # 2. Duplicate node ID check
    node_ids: Set[str] = set()
    for node in validated_data.nodes:
        if node.id in node_ids:
            raise GraphValidationError(f"Duplicate node ID detected: '{node.id}'")
        node_ids.add(node.id)

    # 3. Dangling edges check
    for idx, edge in enumerate(validated_data.edges):
        if edge.source not in node_ids:
            raise GraphValidationError(
                f"Dangling edge at index {idx}: source '{edge.source}' does not exist in nodes."
            )
        if edge.target not in node_ids:
            raise GraphValidationError(
                f"Dangling edge at index {idx}: target '{edge.target}' does not exist in nodes."
            )

    # 4. Check for alias collisions across distinct nodes
    seen_aliases: Dict[str, str] = {}
    for node in validated_data.nodes:
        norm_id = normalize_alias(node.id)
        norm_name = normalize_alias(node.name)

        all_names = [norm_id, norm_name] + [normalize_alias(a) for a in node.aliases if normalize_alias(a)]
        for alias in all_names:
            if alias in seen_aliases and seen_aliases[alias] != node.id:
                raise GraphValidationError(
                    f"Alias collision detected: alias '{alias}' is claimed by both "
                    f"'{seen_aliases[alias]}' and '{node.id}'."
                )
            seen_aliases[alias] = node.id

    return validated_data


def load_knowledge_graph(yaml_path: Union[str, Path]) -> KnowledgeGraph:
    """Parse, validate, and load the knowledge graph from a YAML file.

    Args:
        yaml_path: Path to the graph YAML file.

    Returns:
        Instantiated KnowledgeGraph.

    Raises:
        GraphValidationError: If file does not exist, contains invalid YAML, or violates constraints.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise GraphValidationError(f"Knowledge graph file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise GraphValidationError(f"Malformed YAML syntax in knowledge graph file: {e}") from e

    validated_data = validate_graph_structure(raw_content)
    return KnowledgeGraph(validated_data)
