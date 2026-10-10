"""Deterministic signed weighted graph propagation engine.

Implements:
- Path score = (product of edge strengths) * (0.8 ^ number_of_hops)
- Maximum depth constraint (default 4 hops)
- Configurable minimum score threshold (default 0.10)
- Multiplicative sign math (+1, -1, 0)
- Cycle prevention, duplicate path control
- Contested opposing path detection (never silently netting opposite effects)
- Full path provenance, edge evidence, and cumulative lag days
"""

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_HOP_DECAY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SCORE_THRESHOLD,
)
from src.graph_engine.loader import KnowledgeGraph


SignValue = Literal["positive", "negative", "ambiguous", "contested"]
ConfidenceValue = Literal["high", "medium", "low"]


class PathEdgeDetail(BaseModel):
    """Specification of an edge traversed along a propagation path."""
    source: str
    target: str
    sign: str
    weight: float
    lag_days: int
    rationale: str
    evidence_refs: List[str] = Field(default_factory=list)


class PropagationPath(BaseModel):
    """Full path traversed from origin to destination."""
    path_id: str
    origin_id: str
    target_id: str
    nodes: List[str]
    node_names: List[str]
    edges: List[PathEdgeDetail]
    hops: int
    raw_strength: float
    decayed_score: float
    net_sign: Literal["positive", "negative", "ambiguous"]
    cumulative_lag_days: int
    rationales: List[str]
    evidence_refs: List[str]


class AssetImpact(BaseModel):
    """Consolidated impact assessment for a reachable target asset/node."""
    asset_id: str
    asset_name: str
    asset_type: str
    sector: str
    ticker: Optional[str] = None
    net_direction: SignValue
    is_contested: bool
    confidence_tier: ConfidenceValue
    top_score: float
    top_path: PropagationPath
    all_paths: List[PropagationPath]
    positive_paths_count: int
    negative_paths_count: int
    ambiguous_paths_count: int
    cumulative_lag_days: int
    warning_flags: List[str] = Field(default_factory=list)


class PropagationResult(BaseModel):
    """Top-level container for propagation execution results."""
    origin_id: str
    origin_name: str
    initial_direction: str
    max_depth: int
    hop_decay: float
    min_score_threshold: float
    total_paths_found: int
    affected_assets: List[AssetImpact]


def sign_to_numeric(sign_str: str) -> int:
    """Convert sign string to integer multiplier (+1, -1, 0)."""
    clean = sign_str.lower().strip()
    if clean in ("positive", "+", "pos", "bullish", "up"):
        return 1
    elif clean in ("negative", "-", "neg", "bearish", "down"):
        return -1
    elif clean in ("ambiguous", "mixed", "neutral"):
        return 0
    raise ValueError(f"Unrecognized sign '{sign_str}'. Must be 'positive', 'negative', or 'ambiguous'.")


def numeric_to_sign(num: int) -> Literal["positive", "negative", "ambiguous"]:
    """Convert integer multiplier back to canonical sign string."""
    if num > 0:
        return "positive"
    elif num < 0:
        return "negative"
    return "ambiguous"


class PropagationEngine:
    """Deterministic signed weighted propagation engine for financial knowledge graphs."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        hop_decay: float = DEFAULT_HOP_DECAY,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    ):
        self.kg = knowledge_graph
        self.hop_decay = float(hop_decay)
        self.max_depth = int(max_depth)
        self.min_score_threshold = float(min_score_threshold)

    def propagate(
        self,
        origin_id: str,
        initial_direction: str = "positive",
        max_depth: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
    ) -> PropagationResult:
        """Trace all valid transmission paths from origin_id within max depth and score threshold.

        Args:
            origin_id: Node ID where the shock initiates.
            initial_direction: Direction of shock ('positive', 'negative', 'ambiguous').
            max_depth: Maximum hops to traverse (defaults to instance max_depth).
            min_score_threshold: Cutoff score threshold (defaults to instance min_score_threshold).

        Returns:
            PropagationResult containing sorted AssetImpact records with complete provenance.
        """
        origin_node = self.kg.get_node(origin_id)
        if not origin_node:
            raise ValueError(f"Origin node ID '{origin_id}' not found in knowledge graph.")

        depth_limit = max_depth if max_depth is not None else self.max_depth
        score_cutoff = min_score_threshold if min_score_threshold is not None else self.min_score_threshold
        init_sign_num = sign_to_numeric(initial_direction)

        collected_paths: List[PropagationPath] = []

        # Depth-First Search for all simple transmission paths up to depth_limit
        def _dfs(
            current_node_id: str,
            current_path_nodes: List[str],
            current_edge_details: List[PathEdgeDetail],
            current_raw_strength: float,
            current_sign_num: int,
        ):
            current_hops = len(current_edge_details)
            if current_hops >= depth_limit:
                return

            out_edges = self.kg.graph.out_edges(current_node_id, data=True)
            for _, neighbor_id, edge_attrs in out_edges:
                # Cycle prevention: do not revisit any node already in this path
                if neighbor_id in current_path_nodes:
                    continue

                edge_sign_str = edge_attrs.get("sign", "ambiguous")
                edge_weight = float(edge_attrs.get("weight", 0.5))
                edge_lag = int(edge_attrs.get("lag_days", 0))
                edge_rationale = str(edge_attrs.get("rationale", ""))
                edge_evidence = list(edge_attrs.get("evidence_refs", []))

                new_hops = current_hops + 1
                new_raw_strength = current_raw_strength * edge_weight
                new_decayed_score = new_raw_strength * (self.hop_decay ** new_hops)

                # Prune if score falls strictly below threshold
                if new_decayed_score < score_cutoff:
                    continue

                # Sign multiplication rule
                edge_sign_num = sign_to_numeric(edge_sign_str)
                new_sign_num = current_sign_num * edge_sign_num

                edge_detail = PathEdgeDetail(
                    source=current_node_id,
                    target=neighbor_id,
                    sign=edge_sign_str,
                    weight=edge_weight,
                    lag_days=edge_lag,
                    rationale=edge_rationale,
                    evidence_refs=edge_evidence,
                )

                new_path_nodes = current_path_nodes + [neighbor_id]
                new_path_edges = current_edge_details + [edge_detail]

                # Deterministic path ID based on node sequence
                path_id = "PATH_" + "_".join(new_path_nodes)

                # Node names list
                node_names = []
                for n_id in new_path_nodes:
                    n_obj = self.kg.get_node(n_id)
                    node_names.append(n_obj.name if n_obj else n_id)

                # Cumulative evidence & rationales
                all_evidence: List[str] = []
                all_rationales: List[str] = []
                cumulative_lag = 0
                for e in new_path_edges:
                    cumulative_lag += e.lag_days
                    all_rationales.append(e.rationale)
                    for ref in e.evidence_refs:
                        if ref not in all_evidence:
                            all_evidence.append(ref)

                path_record = PropagationPath(
                    path_id=path_id,
                    origin_id=origin_id,
                    target_id=neighbor_id,
                    nodes=new_path_nodes,
                    node_names=node_names,
                    edges=new_path_edges,
                    hops=new_hops,
                    raw_strength=round(new_raw_strength, 6),
                    decayed_score=round(new_decayed_score, 6),
                    net_sign=numeric_to_sign(new_sign_num),
                    cumulative_lag_days=cumulative_lag,
                    rationales=all_rationales,
                    evidence_refs=all_evidence,
                )
                collected_paths.append(path_record)

                # Continue DFS exploration if within hop budget
                _dfs(
                    neighbor_id,
                    new_path_nodes,
                    new_path_edges,
                    new_raw_strength,
                    new_sign_num,
                )

        # Kick off DFS from origin
        _dfs(origin_id, [origin_id], [], 1.0, init_sign_num)

        # Group paths by target asset
        paths_by_target: Dict[str, List[PropagationPath]] = {}
        for p in collected_paths:
            paths_by_target.setdefault(p.target_id, []).append(p)

        # Build AssetImpact records
        affected_assets: List[AssetImpact] = []
        for target_id, paths in paths_by_target.items():
            # Sort paths for this target by decayed_score descending
            paths.sort(key=lambda x: x.decayed_score, reverse=True)
            top_path = paths[0]

            pos_count = sum(1 for p in paths if p.net_sign == "positive")
            neg_count = sum(1 for p in paths if p.net_sign == "negative")
            amb_count = sum(1 for p in paths if p.net_sign == "ambiguous")

            target_node = self.kg.get_node(target_id)
            target_name = target_node.name if target_node else target_id
            target_type = target_node.type.value if target_node else "unknown"
            target_sector = target_node.sector if target_node else "unknown"
            target_ticker = target_node.ticker if target_node else None

            warning_flags: List[str] = []

            # Contested Opposing Path Check
            is_contested = False
            if pos_count > 0 and neg_count > 0:
                is_contested = True
                net_direction: SignValue = "contested"
                warning_flags.append(
                    f"Contested impact: {pos_count} positive and {neg_count} negative transmission paths detected. "
                    "Opposing forces are not netted out."
                )
            elif pos_count > 0 and neg_count == 0 and amb_count == 0:
                net_direction = "positive"
            elif neg_count > 0 and pos_count == 0 and amb_count == 0:
                net_direction = "negative"
            elif amb_count > 0 and pos_count == 0 and neg_count == 0:
                net_direction = "ambiguous"
                warning_flags.append("Transmission direction is ambiguous due to regime-dependent links.")
            else:
                # One predominant direction alongside ambiguous paths
                if pos_count > 0:
                    net_direction = "positive"
                    warning_flags.append(f"Primary direction positive with {amb_count} ambiguous channel(s).")
                else:
                    net_direction = "negative"
                    warning_flags.append(f"Primary direction negative with {amb_count} ambiguous channel(s).")

            # Confidence Tier Assignment based on top path decayed score
            top_score = top_path.decayed_score
            if top_score >= 0.40:
                confidence_tier: ConfidenceValue = "high"
            elif top_score >= 0.20:
                confidence_tier = "medium"
            else:
                confidence_tier = "low"

            impact = AssetImpact(
                asset_id=target_id,
                asset_name=target_name,
                asset_type=target_type,
                sector=target_sector,
                ticker=target_ticker,
                net_direction=net_direction,
                is_contested=is_contested,
                confidence_tier=confidence_tier,
                top_score=top_score,
                top_path=top_path,
                all_paths=paths,
                positive_paths_count=pos_count,
                negative_paths_count=neg_count,
                ambiguous_paths_count=amb_count,
                cumulative_lag_days=top_path.cumulative_lag_days,
                warning_flags=warning_flags,
            )
            affected_assets.append(impact)

        # Sort affected assets globally by top_score descending
        affected_assets.sort(key=lambda x: x.top_score, reverse=True)

        return PropagationResult(
            origin_id=origin_id,
            origin_name=origin_node.name,
            initial_direction=initial_direction,
            max_depth=depth_limit,
            hop_decay=self.hop_decay,
            min_score_threshold=score_cutoff,
            total_paths_found=len(collected_paths),
            affected_assets=affected_assets,
        )
