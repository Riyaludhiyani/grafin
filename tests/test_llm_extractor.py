"""Unit tests for Phase 5: Provider-independent LLM extraction adapter and fallback logic."""

import json
import os
from unittest.mock import MagicMock
import pytest
import requests

from src.config import GRAPH_YAML_PATH
from src.extraction.llm_extractor import (
    BaseLLMAdapter,
    HTTPLLMAdapter,
    LLMGenerationError,
    classify_event,
    clean_llm_json_output,
)
from src.extraction.schemas import EventDirection, EventStatus, ExtractedEvent
from src.graph_engine.loader import load_knowledge_graph


class MockSuccessAdapter(BaseLLMAdapter):
    """Mock adapter returning valid schema-conformant JSON with markdown fencing."""

    def __init__(self, primary_id: str = "macro_brent_crude"):
        self.primary_id = primary_id

    def generate_json(self, prompt: str) -> str:
        payload = {
            "event_id": "mock_event_123",
            "event_type": "commodity_shock",
            "status": "observed",
            "direction": "positive",
            "magnitude": "severe",
            "magnitude_numeric": 8.0,
            "magnitude_unit": "%",
            "geography": "India",
            "timestamp": "2024-04-15T09:30:00Z",
            "source": "Mocked Global News Wire",
            "primary_entity_id": self.primary_id,
            "primary_entity_name": "Brent Crude Oil",
            "mentioned_entities": [
                {"text": "Asian Paints", "resolved_node_id": None, "entity_type": None},
                {"text": "Brent crude", "resolved_node_id": None, "entity_type": None},
            ],
            "evidence_snippets": [
                "Brent crude oil rallied over 8% following escalated supply threats.",
                "Asian Paints shares traded lower as raw material solvent costs jumped.",
            ],
        }
        # Wrap in markdown fences to test stripper
        return f"```json\n{json.dumps(payload)}\n```"


class MockGarbledAdapter(BaseLLMAdapter):
    """Mock adapter returning non-JSON invalid text."""

    def generate_json(self, prompt: str) -> str:
        return "Sorry, as an AI language model I cannot analyze this financial text."


class MockMissingFieldsAdapter(BaseLLMAdapter):
    """Mock adapter returning JSON missing mandatory evidence_snippets."""

    def generate_json(self, prompt: str) -> str:
        return json.dumps({
            "event_id": "bad_event",
            "event_type": "commodity_shock",
            "status": "observed",
            "direction": "positive",
            "magnitude": "severe",
            "timestamp": "2024-04-15T09:30:00Z",
            "source": "Mock Wire",
            "primary_entity_id": "macro_brent_crude",
            "primary_entity_name": "Brent Crude",
            "evidence_snippets": [],  # Violates schema: cannot be empty!
        })


class MockTimeoutAdapter(BaseLLMAdapter):
    """Mock adapter that simulates a network timeout."""

    def generate_json(self, prompt: str) -> str:
        raise LLMGenerationError("Request timed out after 10s")


class MockHttpErrorAdapter(BaseLLMAdapter):
    """Mock adapter that simulates an HTTP 500 error."""

    def generate_json(self, prompt: str) -> str:
        raise LLMGenerationError("HTTP 500 Internal Server Error from LLM provider")


@pytest.fixture
def loaded_kg():
    return load_knowledge_graph(GRAPH_YAML_PATH)


def test_clean_llm_json_output():
    """Verify markdown code fence stripping."""
    assert clean_llm_json_output('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_llm_json_output('```\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_llm_json_output('{"a": 1}') == '{"a": 1}'


def test_mocked_valid_llm_response(loaded_kg):
    """Verify that a valid JSON response from the LLM adapter is parsed and normalized."""
    adapter = MockSuccessAdapter()
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"

    event = classify_event(text=headline, knowledge_graph=loaded_kg, adapter=adapter)

    assert isinstance(event, ExtractedEvent)
    assert event.event_id == "mock_event_123"
    assert event.primary_entity_id == "macro_brent_crude"
    assert event.direction == EventDirection.POSITIVE
    assert len(event.evidence_snippets) == 2

    # Verify that mentioned entities were normalized against the graph
    resolved_ids = {m.resolved_node_id for m in event.mentioned_entities if m.resolved_node_id}
    assert "equity_asianpaints" in resolved_ids
    assert "macro_brent_crude" in resolved_ids


def test_mocked_alias_in_primary_entity_id_resolves(loaded_kg):
    """Verify that if LLM returns an alias instead of exact ID, it resolves cleanly."""
    adapter = MockSuccessAdapter(primary_id="Brent crude")  # Colloquial alias
    headline = "Brent crude surges 8% to $92/bbl"

    event = classify_event(text=headline, knowledge_graph=loaded_kg, adapter=adapter)

    assert event.primary_entity_id == "macro_brent_crude"


def test_mocked_invalid_json_falls_back_to_deterministic(loaded_kg):
    """Verify that non-JSON output triggers fallback to deterministic extraction without crashing."""
    adapter = MockGarbledAdapter()
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"

    event = classify_event(
        text=headline,
        event_id="event_crude_oil_spike_2024",
        knowledge_graph=loaded_kg,
        adapter=adapter,
    )

    assert isinstance(event, ExtractedEvent)
    assert event.event_id == "event_crude_oil_spike_2024"
    assert event.primary_entity_id == "macro_brent_crude"


def test_mocked_missing_required_fields_falls_back(loaded_kg):
    """Verify that missing fields trigger graceful fallback."""
    adapter = MockMissingFieldsAdapter()
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"

    event = classify_event(
        text=headline,
        event_id="event_crude_oil_spike_2024",
        knowledge_graph=loaded_kg,
        adapter=adapter,
    )

    assert isinstance(event, ExtractedEvent)
    assert event.primary_entity_id == "macro_brent_crude"


def test_mocked_hallucinated_node_id_falls_back(loaded_kg):
    """Verify that an unresolvable hallucinated node ID is rejected and falls back safely."""
    adapter = MockSuccessAdapter(primary_id="fake_unregistered_node_99999")
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"

    event = classify_event(
        text=headline,
        event_id="event_crude_oil_spike_2024",
        knowledge_graph=loaded_kg,
        adapter=adapter,
    )

    assert isinstance(event, ExtractedEvent)
    # Must fallback to valid graph node ID
    assert event.primary_entity_id == "macro_brent_crude"


def test_mocked_api_timeout_falls_back(loaded_kg):
    """Verify that network timeouts fall back cleanly without terminating execution."""
    adapter = MockTimeoutAdapter()
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"

    event = classify_event(
        text=headline,
        event_id="event_crude_oil_spike_2024",
        knowledge_graph=loaded_kg,
        adapter=adapter,
    )

    assert isinstance(event, ExtractedEvent)
    assert event.primary_entity_id == "macro_brent_crude"


def test_mocked_api_http_error_falls_back(loaded_kg):
    """Verify that HTTP error responses fall back cleanly without throwing unhandled exceptions."""
    adapter = MockHttpErrorAdapter()
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"

    event = classify_event(
        text=headline,
        event_id="event_crude_oil_spike_2024",
        knowledge_graph=loaded_kg,
        adapter=adapter,
    )

    assert isinstance(event, ExtractedEvent)
    assert event.primary_entity_id == "macro_brent_crude"


def test_missing_credentials_automatically_uses_fallback(loaded_kg):
    """Verify that default classify_event without API credentials runs deterministic fallback."""
    adapter = HTTPLLMAdapter(api_key="", provider="none")
    headline = "RBI raises repo rate by 50 bps to anchor inflation"

    event = classify_event(
        text=headline,
        event_id="event_rbi_repo_rate_hike_50bps",
        knowledge_graph=loaded_kg,
        adapter=adapter,
    )

    assert isinstance(event, ExtractedEvent)
    assert event.primary_entity_id == "macro_rbi_repo_rate"


@pytest.mark.skipif(
    not os.getenv("LLM_API_KEY"),
    reason="Live LLM credentials not configured in environment (skipping real external integration check)",
)
def test_real_llm_integration_check(loaded_kg):
    """Separately labeled real integration check running against live LLM API if key is present."""
    adapter = HTTPLLMAdapter()
    headline = "Brent crude oil prices jump 6% amid Red Sea shipping disruptions"

    # Call live adapter without exposing key
    event = classify_event(text=headline, knowledge_graph=loaded_kg, adapter=adapter)

    assert isinstance(event, ExtractedEvent)
    assert event.primary_entity_id in loaded_kg.nodes_data
    assert len(event.evidence_snippets) > 0
