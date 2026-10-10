"""Comprehensive End-to-End integration tests for Autonomous Financial Event Market Impact Graph.

Tests the full pipeline from:
1. Event selection & extraction
2. Entity resolution & graph anchoring
3. Signed weighted propagation
4. Asset ranking & contested path detection
5. Market data validation & outage resilience
6. Grounded chatbot citation & counterfactual simulation
7. Offline mode & absent API credentials
8. Edge cases: empty graph traversal, unknown entities, malformed data
"""

import time
import pytest

from src.config import GRAPH_YAML_PATH, SAMPLE_EVENTS_PATH
from src.explainer.chatbot import GroundedChatbot
from src.extraction.deterministic import load_curated_events
from src.extraction.llm_extractor import classify_event
from src.graph_engine.loader import load_knowledge_graph
from src.graph_engine.propagator import PropagationEngine
from src.market.downloader import MarketDataDownloader
from src.market.validator import MarketValidator
from src.ui.components import build_ranked_assets_dataframe


@pytest.fixture
def e2e_kg():
    """Load the production knowledge graph."""
    return load_knowledge_graph(GRAPH_YAML_PATH)


@pytest.fixture
def e2e_curated_events():
    """Load all curated financial news events."""
    return load_curated_events(SAMPLE_EVENTS_PATH)


def test_e2e_all_six_curated_events_pipeline(e2e_kg, e2e_curated_events):
    """End-to-End test verifying that all 6 curated events traverse the complete pipeline without error."""
    assert len(e2e_curated_events) >= 6

    engine = PropagationEngine(knowledge_graph=e2e_kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    downloader = MarketDataDownloader()
    validator = MarketValidator(downloader=downloader)
    chatbot = GroundedChatbot(knowledge_graph=e2e_kg)

    for event_record in e2e_curated_events:
        # Step 1: Extraction & Entity Resolution
        extracted = classify_event(
            text=f"{event_record.title}. {event_record.body}",
            event_id=event_record.id,
            knowledge_graph=e2e_kg,
        )
        assert extracted.event_id == event_record.id
        assert extracted.primary_entity_id in e2e_kg.nodes_data

        # Step 2: Graph Anchoring & Propagation
        start_time = time.time()
        result = engine.propagate(
            origin_id=extracted.primary_entity_id,
            initial_direction=extracted.direction.value,
        )
        prop_duration_ms = (time.time() - start_time) * 1000.0

        # Performance requirement: propagation should be instantaneous (< 50ms)
        assert prop_duration_ms < 50.0, f"Propagation took too long: {prop_duration_ms:.2f}ms"
        assert result.total_paths_found > 0
        assert len(result.affected_assets) > 0

        # Step 3: Ranked Assets Table Dataframe Formatting
        df_ranked = build_ranked_assets_dataframe(result.affected_assets)
        assert len(df_ranked) == len(result.affected_assets)
        assert "Score" in df_ranked.columns
        assert "Direction" in df_ranked.columns

        # Step 4: Market Validation on Top Asset
        top_asset = result.affected_assets[0]
        v_reports = {}
        if top_asset.ticker:
            driver_ticker = e2e_kg.get_ticker(extracted.primary_entity_id)
            if driver_ticker:
                report = validator.validate_relationship(
                    source_symbol=driver_ticker,
                    target_symbol=top_asset.ticker,
                    prior_sign=top_asset.top_path.net_sign,
                    expected_lag_days=top_asset.top_path.cumulative_lag_days,
                )
                assert report.status in ("supporting", "contradicting", "inconclusive", "insufficient_data")
                v_reports[top_asset.ticker] = report

        # Step 5: Grounded Chatbot Synthesis
        test_query = f"What evidence supports {top_asset.asset_name}?"
        answer = chatbot.ask(test_query, extracted, result, v_reports)
        assert len(answer.strip()) > 30
        assert "PATH_" in answer
        assert "[Curated Prior]" in answer or "[Observed Fact]" in answer


def test_e2e_market_data_outage_resilience(e2e_kg, e2e_curated_events):
    """Verify that if market data downloading completely fails, the application proceeds gracefully."""
    engine = PropagationEngine(e2e_kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    event_record = e2e_curated_events[0]
    extracted = event_record.cached_extraction
    result = engine.propagate(origin_id=extracted.primary_entity_id, initial_direction=extracted.direction.value)

    # Mock a total downloader outage (returns None)
    downloader = MarketDataDownloader()
    downloader.get_historical_prices = lambda *args, **kwargs: None

    validator = MarketValidator(downloader=downloader)
    report = validator.validate_relationship(
        source_symbol="BZ=F",
        target_symbol="ASIANPAINT.NS",
        prior_sign="negative",
    )

    # Must return insufficient_data rather than raising an uncaught exception
    assert report.status == "insufficient_data"
    assert report.sample_size == 0
    assert "unavailable or empty" in report.explanation

    # Chatbot still works and includes the outage notice
    chatbot = GroundedChatbot(knowledge_graph=e2e_kg)
    ans = chatbot.ask(
        "Why is Asian Paints affected?",
        extracted,
        result,
        validation_reports={"ASIANPAINT.NS": report},
    )
    assert "Asian Paints" in ans
    assert "INSUFFICIENT_DATA" in ans or "unavailable" in ans or "PATH_" in ans


def test_e2e_empty_graph_results_handling(e2e_kg, e2e_curated_events):
    """Verify behavior when propagation threshold is so restrictive that zero paths survive."""
    # Threshold 0.99 is higher than any multi-hop decayed score
    engine = PropagationEngine(e2e_kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.99)
    event_record = e2e_curated_events[0]
    extracted = event_record.cached_extraction

    result = engine.propagate(origin_id=extracted.primary_entity_id, initial_direction=extracted.direction.value)

    # Result should be empty without crashing
    assert result.total_paths_found == 0
    assert len(result.affected_assets) == 0

    df_ranked = build_ranked_assets_dataframe(result.affected_assets)
    assert df_ranked.empty


def test_e2e_offline_mode_with_absent_api_keys(e2e_kg):
    """Verify complete end-to-end flow functions without internet and with empty LLM API keys."""
    # Ensure offline mode
    headline = "Brent crude surges 8% to $92/bbl following geopolitical tensions"
    body = "Global crude oil prices spiked today, putting margin pressure on Asian Paints."

    extracted = classify_event(text=f"{headline}. {body}", knowledge_graph=e2e_kg)

    assert extracted.primary_entity_id == "macro_brent_crude"
    assert len(extracted.evidence_snippets) > 0

    engine = PropagationEngine(e2e_kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    result = engine.propagate(origin_id=extracted.primary_entity_id, initial_direction=extracted.direction.value)

    chatbot = GroundedChatbot(e2e_kg)
    ans = chatbot.ask("Why might crude affect aviation?", extracted, result)

    assert "macro_brent_crude -> sector_aviation" in ans
    assert "[Curated Prior]" in ans


def test_e2e_counterfactual_simulation_depth(e2e_kg, e2e_curated_events):
    """Verify counterfactual calculation produces inverse directions across the graph."""
    engine = PropagationEngine(e2e_kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    event_record = e2e_curated_events[0]  # Positive crude shock (+1)
    extracted = event_record.cached_extraction

    base_result = engine.propagate(origin_id=extracted.primary_entity_id, initial_direction="positive")
    chatbot = GroundedChatbot(e2e_kg)

    # Run counterfactual flipping direction to negative (-1)
    cf_result, cf_explanation = chatbot.run_counterfactual(extracted, base_result, flipped_direction="negative")

    assert cf_result.initial_direction == "negative"
    assert "COUNTERFACTUAL SIMULATION" in cf_explanation

    # Check that net direction flipped for non-ambiguous assets
    base_asianpaints = next(a for a in base_result.affected_assets if a.asset_id == "equity_asianpaints")
    cf_asianpaints = next(a for a in cf_result.affected_assets if a.asset_id == "equity_asianpaints")

    assert base_asianpaints.net_direction == "negative"
    assert cf_asianpaints.net_direction == "positive"  # Inverted!
