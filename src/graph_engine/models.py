"""Data models and schemas for the Indian Financial Knowledge Graph."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    """Categorical classification of a knowledge graph node."""
    MACRO_INDICATOR = "macro_indicator"
    MARKET_INDEX = "market_index"
    SECTOR = "sector"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    EQUITY = "equity"


class SignType(str, Enum):
    """Direction of hypothesized transmission impact."""
    POSITIVE = "positive"      # Direct / positive correlation (A increases -> B increases)
    NEGATIVE = "negative"      # Inverse / negative correlation (A increases -> B decreases)
    AMBIGUOUS = "ambiguous"    # Non-linear, contested, or regime-dependent transmission


class ConfidenceTier(str, Enum):
    """Subjective structural confidence in the transmission link."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProvenanceType(str, Enum):
    """Origin and grounding of the relationship."""
    CURATED_PRIOR = "curated_prior"
    OBSERVED_MARKET = "observed_market"
    DOMAIN_LITERATURE = "domain_literature"


class GraphNode(BaseModel):
    """Specification of a node in the financial knowledge graph."""
    id: str = Field(..., description="Unique alphanumeric identifier (e.g., 'macro_brent_crude')")
    name: str = Field(..., description="Official human-readable name")
    type: NodeType = Field(..., description="Entity category")
    sector: str = Field(..., description="Sector or domain tag (e.g., 'banking', 'macro', 'energy')")
    ticker: Optional[str] = Field(None, description="Yahoo Finance market ticker if asset is traded")
    aliases: List[str] = Field(default_factory=list, description="Common colloquial and market synonyms")
    description: str = Field("", description="Economic description of this entity")

    @field_validator("id")
    @classmethod
    def validate_id_not_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Node ID cannot be empty or whitespace.")
        return clean


class GraphEdge(BaseModel):
    """Specification of a directed, signed, weighted transmission relationship."""
    source: str = Field(..., description="Origin node ID")
    target: str = Field(..., description="Destination node ID")
    sign: SignType = Field(..., description="Hypothesized transmission direction (positive/negative/ambiguous)")
    weight: float = Field(..., ge=0.01, le=1.0, description="Hypothesized relationship strength (0.01 to 1.0)")
    lag_days: int = Field(0, ge=0, description="Expected operational or market transmission lag in trading days")
    confidence: ConfidenceTier = Field(ConfidenceTier.MEDIUM, description="Structural confidence tier")
    provenance: ProvenanceType = Field(ProvenanceType.CURATED_PRIOR, description="Origin of link")
    rationale: str = Field(..., description="Economic or business explanation for the transmission link")
    evidence_refs: List[str] = Field(default_factory=list, description="Structural and domain citations")

    @field_validator("source", "target")
    @classmethod
    def validate_node_endpoints(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Edge endpoint IDs cannot be empty.")
        return clean

    @field_validator("target")
    @classmethod
    def validate_no_self_loops(cls, v: str, info) -> str:
        if "source" in info.data and v == info.data["source"]:
            raise ValueError(f"Self-loop detected: source and target are identical ('{v}')")
        return v


class KnowledgeGraphMetadata(BaseModel):
    """Metadata describing the knowledge graph version and scope."""
    name: str = Field(..., description="Name of the knowledge graph")
    version: str = Field(..., description="Semantic version string")
    geography: str = Field("India", description="Geographic domain")
    description: str = Field("", description="Overview of the graph's coverage")
    updated_at: str = Field(..., description="ISO timestamp of last curation")


class KnowledgeGraphData(BaseModel):
    """Top-level container for validated knowledge graph data."""
    metadata: KnowledgeGraphMetadata
    nodes: List[GraphNode]
    edges: List[GraphEdge]
