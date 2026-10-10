"""Strict Pydantic and JSON schemas for structured financial news event extraction."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class EventStatus(str, Enum):
    """Factual status of the reported event."""
    OBSERVED = "observed"          # Confirmed, historical or enacted market occurrence
    PROJECTED = "projected"        # Forward guidance, anticipated forecast, or forecast estimate
    RUMOR = "rumor"                # Unofficial report or speculative speculation
    CONTESTED = "contested"        # Disputed claim or conflicting accounts


class EventDirection(str, Enum):
    """Direction of the primary shock."""
    POSITIVE = "positive"          # Increase, rate hike, price surge, appreciation
    NEGATIVE = "negative"          # Decrease, rate cut, price plunge, depreciation
    AMBIGUOUS = "ambiguous"        # Mixed, multi-directional, or unclear shock


class MagnitudeTier(str, Enum):
    """Categorical scale of the shock."""
    MINOR = "minor"                # < 25 bps, < 2% price swing, or marginal tweak
    MODERATE = "moderate"          # 25-50 bps, 2-5% price move, or standard policy change
    SEVERE = "severe"              # > 50 bps, > 5% price shock, or emergency intervention


class MentionedEntity(BaseModel):
    """Specification of an entity mentioned within the news text."""
    text: str = Field(..., description="Exact entity mention substring from the source text")
    resolved_node_id: Optional[str] = Field(None, description="Graph node ID if successfully resolved, else None")
    entity_type: Optional[str] = Field(None, description="Category of the entity (commodity, equity, macro, etc.)")

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Entity mention text cannot be empty.")
        return clean


class ExtractedEvent(BaseModel):
    """Strictly validated structured financial event payload."""
    event_id: str = Field(..., description="Unique identifier for the event")
    event_type: str = Field(..., description="Categorical classification of the event")
    status: EventStatus = Field(..., description="Factual verification state (observed, projected, etc.)")
    direction: EventDirection = Field(..., description="Primary shock vector (positive/negative/ambiguous)")
    magnitude: MagnitudeTier = Field(..., description="Shock intensity (minor/moderate/severe)")
    magnitude_numeric: Optional[float] = Field(None, description="Parsed numerical magnitude (e.g. 50 bps -> 0.50)")
    magnitude_unit: Optional[str] = Field(None, description="Unit of magnitude (e.g. 'bps', '%', 'USD/bbl')")
    geography: str = Field("India", description="Primary geographic scope")
    timestamp: str = Field(..., description="ISO 8601 publication or enactment timestamp")
    source: str = Field(..., description="Authoritative publication or data attribution")
    primary_entity_id: str = Field(..., description="Graph node ID of the primary origin shock")
    primary_entity_name: str = Field(..., description="Human-readable title of primary origin shock")
    mentioned_entities: List[MentionedEntity] = Field(default_factory=list, description="Extracted entity mentions")
    evidence_snippets: List[str] = Field(default_factory=list, description="Direct verbatim supporting quotes from text")

    @field_validator("primary_entity_id")
    @classmethod
    def validate_primary_entity_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("primary_entity_id cannot be empty.")
        return clean

    @field_validator("evidence_snippets")
    @classmethod
    def validate_evidence_snippets(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one supporting evidence snippet is required.")
        for snippet in v:
            if not snippet.strip():
                raise ValueError("Evidence snippets cannot be empty strings.")
        return v


class CuratedEventRecord(BaseModel):
    """Container for curated financial news event records with pre-cached extraction."""
    id: str = Field(..., description="Unique event ID")
    title: str = Field(..., description="News headline")
    body: str = Field(..., description="Full text of the news article")
    source_label: str = Field(..., description="Source name and attribution tier")
    publication_date: str = Field(..., description="ISO timestamp")
    source_type: str = Field(..., description="Source classification (verified_regulatory, illustrative_synthetic, etc.)")
    claim_status: str = Field(..., description="claim status (verified_historical or illustrative_synthetic)")
    cached_extraction: ExtractedEvent = Field(..., description="Validated structured extraction")
