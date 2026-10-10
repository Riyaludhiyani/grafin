"""Unit tests for Phase 8: Grounded decision-support chatbot and counterfactual simulator."""

import pytest

from src.config import GRAPH_YAML_PATH, SAMPLE_EVENTS_PATH
from src.explainer.chatbot import GroundedChatbot
from src.extraction.deterministic import get_curated_event_by_id
from src.extraction.llm_extractor import BaseLLMAdapter, LLMGenerationError
from src.graph_engine.loader import load_knowledge_graph
from src.graph_engine.propagator import PropagationEngine


class FailingLLMAdapter(BaseLLMAdapter):
    """Simulates an LLM API error or timeout."""

    def generate_json(self, prompt: str) -> str:
        raise LLMGenerationError("Network timeout connecting to external LLM service")


@pytest.fixture
def loaded_kg():
    return load_knowledge_graph(GRAPH_YAML_PATH)


@pytest.fixture
def active_crude_event():
    curated = get_curated_event_by_id("event_crude_oil_spike_2024", SAMPLE_EVENTS_PATH)
    assert curated is not None
    return curated.cached_extraction


@pytest.fixture
def active_propagation_result(loaded_kg, active_crude_event):
    engine = PropagationEngine(loaded_kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    return engine.propagate(origin_id=active_crude_event.primary_entity_id, initial_direction="positive")


def test_unsupported_question_declares_insufficient_context(loaded_kg, active_crude_event, active_propagation_result):
    """Verify that unsupported or out-of-domain questions state that context is insufficient."""
    chatbot = GroundedChatbot(knowledge_graph=loaded_kg)

    ans_btc = chatbot.ask("Will Bitcoin reach $150k next month?", active_crude_event, active_propagation_result)
    assert "insufficient" in ans_btc.lower()
    assert "out of domain" in ans_btc.lower()

    ans_tsla = chatbot.ask("What is the price target for Tesla in 2026?", active_crude_event, active_propagation_result)
    assert "insufficient" in ans_tsla.lower()


def test_deterministic_fallback_crude_aviation(loaded_kg, active_crude_event, active_propagation_result):
    """Verify crude oil to aviation question cites edge IDs, literature references, and distinct labels."""
    chatbot = GroundedChatbot(knowledge_graph=loaded_kg)

    ans = chatbot.ask("Why might crude affect aviation?", active_crude_event, active_propagation_result)

    # 1. Cites specific edge
    assert "macro_brent_crude -> sector_aviation" in ans

    # 2. Distinguishes fact, prior, and inferred impact
    assert "[Observed Fact]" in ans
    assert "[Curated Prior]" in ans
    assert "[Inferred Impact]" in ans

    # 3. Contains official citations
    assert "DGCA" in ans or "InterGlobe Aviation" in ans

    # 4. Mandatory disclaimer
    assert "Hypothesized transmission path, not a stock-price prediction" in ans


def test_specific_asset_evidence_query(loaded_kg, active_crude_event, active_propagation_result):
    """Verify asset queries cite Path ID and edge rationales."""
    chatbot = GroundedChatbot(knowledge_graph=loaded_kg)

    ans = chatbot.ask("What evidence supports Asian Paints?", active_crude_event, active_propagation_result)

    assert "Asian Paints" in ans
    assert "PATH_" in ans
    assert "[Curated Prior]" in ans
    assert "[Inferred Impact]" in ans


def test_counterfactual_calculation_runs_propagation(loaded_kg, active_crude_event, active_propagation_result):
    """Verify counterfactual inquiry genuinely re-computes propagation scores with reversed vector."""
    chatbot = GroundedChatbot(knowledge_graph=loaded_kg)

    cf_result, cf_text = chatbot.run_counterfactual(active_crude_event, active_propagation_result)

    # Flipped from positive (+1) to negative (-1)
    assert cf_result.initial_direction == "negative"
    assert "COUNTERFACTUAL SIMULATION" in cf_text
    assert "direction flips to" in cf_text
    assert "PATH_" in cf_text
    assert "not predictive financial price forecasts" in cf_text

    # Asking via chat input also triggers counterfactual engine
    ans_chat = chatbot.ask("What if the event direction is reversed (counterfactual)?", active_crude_event, active_propagation_result)
    assert "COUNTERFACTUAL SIMULATION" in ans_chat


def test_deterministic_fallback_contested_asset(loaded_kg, active_crude_event, active_propagation_result):
    """Verify contested asset inquiries explain opposing channels without netting."""
    chatbot = GroundedChatbot(knowledge_graph=loaded_kg)

    ans = chatbot.ask("Why is Bank Nifty contested?", active_crude_event, active_propagation_result)
    assert len(ans.strip()) > 50
    # Must mention contested or explain opposing forces
    assert "contested" in ans.lower() or "opposing" in ans.lower() or "audit" in ans.lower()


def test_llm_adapter_failure_falls_back_to_deterministic(loaded_kg, active_crude_event, active_propagation_result):
    """Verify that if live LLM adapter throws an exception, chatbot falls back seamlessly."""
    chatbot = GroundedChatbot(knowledge_graph=loaded_kg, adapter=FailingLLMAdapter())

    # Must not crash!
    ans = chatbot.ask("Why might crude affect aviation?", active_crude_event, active_propagation_result)
    assert "macro_brent_crude -> sector_aviation" in ans
    assert "[Observed Fact]" in ans
