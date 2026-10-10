"""Unit tests for Phase 4: Curated event dataset, schemas, and deterministic extraction fallback."""

import pytest
from pydantic import ValidationError

from src.config import GRAPH_YAML_PATH, SAMPLE_EVENTS_PATH
from src.extraction.deterministic import (
    DeterministicExtractor,
    get_curated_event_by_id,
    load_curated_events,
)
from src.extraction.schemas import (
    CuratedEventRecord,
    EventDirection,
    EventStatus,
    ExtractedEvent,
    MagnitudeTier,
    MentionedEntity,
)
from src.graph_engine.loader import load_knowledge_graph


@pytest.fixture
def loaded_kg():
    """Load the production knowledge graph."""
    return load_knowledge_graph(GRAPH_YAML_PATH)


def test_all_curated_events_load_and_validate():
    """Verify that all curated events load properly and satisfy the strict schema."""
    events = load_curated_events(SAMPLE_EVENTS_PATH)
    assert len(events) >= 6, f"Expected at least 6 curated events, found {len(events)}"

    event_ids = [e.id for e in events]
    # Verify coverage of all 6 mandatory India-focused domains
    assert "event_crude_oil_spike_2024" in event_ids
    assert "event_rbi_repo_rate_hike_50bps" in event_ids
    assert "event_rupee_depreciation_record_low" in event_ids
    assert "event_monsoon_food_cpi_inflation_spike" in event_ids
    assert "event_rbi_unsecured_loan_risk_weight_tightening" in event_ids
    assert "event_us_fed_hawkish_fii_equity_outflow" in event_ids

    # Validate each record fields and evidence snippets
    for event in events:
        assert len(event.title.strip()) > 10
        assert len(event.body.strip()) > 50
        assert event.source_type in ("verified_regulatory", "synthetic_case")
        assert event.claim_status in ("verified_historical", "illustrative_synthetic")
        assert isinstance(event.cached_extraction, ExtractedEvent)
        assert len(event.cached_extraction.evidence_snippets) >= 2
        assert len(event.cached_extraction.mentioned_entities) >= 3


def test_get_curated_event_by_id():
    """Verify retrieval of a single curated event by ID."""
    ev = get_curated_event_by_id("event_crude_oil_spike_2024")
    assert ev is not None
    assert "Brent crude" in ev.title
    assert ev.cached_extraction.primary_entity_id == "macro_brent_crude"

    # Non-existent ID returns None
    assert get_curated_event_by_id("non_existent_event_id") is None


def test_schema_validation_failures():
    """Verify that schema errors are raised when required fields are missing or invalid."""
    # 1. Missing evidence snippets
    with pytest.raises(ValidationError):
        ExtractedEvent(
            event_id="test_bad_1",
            event_type="commodity_shock",
            status=EventStatus.OBSERVED,
            direction=EventDirection.POSITIVE,
            magnitude=MagnitudeTier.SEVERE,
            timestamp="2024-04-15T09:30:00Z",
            source="Test",
            primary_entity_id="macro_brent_crude",
            primary_entity_name="Brent Crude",
            evidence_snippets=[],  # Empty evidence snippets must be rejected
        )

    # 2. Missing primary_entity_id
    with pytest.raises(ValidationError):
        ExtractedEvent(
            event_id="test_bad_2",
            event_type="commodity_shock",
            status=EventStatus.OBSERVED,
            direction=EventDirection.POSITIVE,
            magnitude=MagnitudeTier.SEVERE,
            timestamp="2024-04-15T09:30:00Z",
            source="Test",
            primary_entity_id="",  # Empty string rejected
            primary_entity_name="Brent Crude",
            evidence_snippets=["Evidence quote"],
        )

    # 3. Invalid direction enum value
    with pytest.raises(ValidationError):
        ExtractedEvent(
            event_id="test_bad_3",
            event_type="commodity_shock",
            status=EventStatus.OBSERVED,
            direction="sideways",  # Invalid enum value
            magnitude=MagnitudeTier.SEVERE,
            timestamp="2024-04-15T09:30:00Z",
            source="Test",
            primary_entity_id="macro_brent_crude",
            primary_entity_name="Brent Crude",
            evidence_snippets=["Evidence quote"],
        )


def test_magnitude_parsing(loaded_kg):
    """Verify deterministic extraction of magnitude tiers and numerical amounts."""
    extractor = DeterministicExtractor(loaded_kg)

    tier, num, unit = extractor.parse_magnitude("RBI hikes repo rate by 50 bps")
    assert tier == MagnitudeTier.MODERATE
    assert num == 50.0
    assert unit == "bps"

    tier, num, unit = extractor.parse_magnitude("Brent crude surges 8.5% overnight")
    assert tier == MagnitudeTier.SEVERE
    assert num == 8.5
    assert unit == "%"

    tier, num, unit = extractor.parse_magnitude("Rupee slips by 38 paise against dollar")
    assert tier == MagnitudeTier.MODERATE
    assert num == 38.0
    assert unit == "paise"


def test_direction_parsing(loaded_kg):
    """Verify direction inference from text verbs."""
    extractor = DeterministicExtractor(loaded_kg)

    assert extractor.parse_direction("Crude prices surge and jump") == EventDirection.POSITIVE
    assert extractor.parse_direction("Rupee tumbles and slides to new low") == EventDirection.NEGATIVE
    assert extractor.parse_direction("Trading session concludes with steady volume") == EventDirection.AMBIGUOUS


def test_entity_resolution_via_graph_aliases(loaded_kg):
    """Verify entity mentions in text are matched to correct graph node IDs."""
    extractor = DeterministicExtractor(loaded_kg)
    text = "Asian Paints and IndiGo faced selling while ONGC rallied as Crude Oil surged."

    entities = extractor.extract_entities(text)
    resolved_ids = {e.resolved_node_id for e in entities if e.resolved_node_id}

    assert "equity_asianpaints" in resolved_ids
    assert "equity_indigo" in resolved_ids
    assert "equity_ongc" in resolved_ids
    assert "macro_brent_crude" in resolved_ids


def test_unknown_entity_handling_does_not_crash(loaded_kg):
    """Verify unmapped or unknown entities do not cause exceptions."""
    extractor = DeterministicExtractor(loaded_kg)
    synthetic_text = "SpaceX launched new satellites while OpenAI announced enterprise partnership."

    # Must extract without crashing
    entities = extractor.extract_entities(synthetic_text)
    # Neither SpaceX nor OpenAI exist in Indian financial graph
    assert len(entities) == 0

    # Test MentionedEntity validator with None resolved_node_id
    ent = MentionedEntity(text="SpaceX", resolved_node_id=None, entity_type="aerospace")
    assert ent.resolved_node_id is None
    assert ent.text == "SpaceX"


def test_offline_curated_pipeline_works_without_llm(loaded_kg):
    """Demonstrate that the entire extraction pipeline operates with zero LLM credentials."""
    extractor = DeterministicExtractor(loaded_kg)

    # 1. Test against curated event
    result_curated = extractor.extract_event(
        headline="Brent crude surges 8% to $92/bbl following geopolitical tensions",
        body="Global crude oil prices witnessed a sudden and sharp spike today as Brent crude rallied over 8%...",
        event_id="event_crude_oil_spike_2024",
    )
    assert result_curated.event_id == "event_crude_oil_spike_2024"
    assert result_curated.primary_entity_id == "macro_brent_crude"
    assert len(result_curated.evidence_snippets) >= 2

    # 2. Test against arbitrary uncurated news text
    result_fresh = extractor.extract_event(
        headline="RBI hikes repo rate by 25 bps amid inflation worries",
        body="The Monetary Policy Committee announced a 25 basis points hike in benchmark repo rate today.",
    )
    assert result_fresh.primary_entity_id == "macro_rbi_repo_rate"
    assert result_fresh.direction == EventDirection.POSITIVE
    assert result_fresh.magnitude_numeric == 25.0
    assert result_fresh.magnitude_unit == "bps"
    assert len(result_fresh.evidence_snippets) > 0
