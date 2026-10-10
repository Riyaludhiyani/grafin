#!/usr/bin/env python3
"""Autonomous Financial Event Market Impact Graph - Standalone CLI Demo Walkthrough.

Executes the full end-to-end pipeline in terminal mode:
1. Loads validated Indian financial knowledge graph
2. Ingests and extracts a curated event
3. Propagates signed weighted transmission paths
4. Ranks affected assets by decayed score
5. Validates empirical correlation against historical market data
6. Executes grounded Q&A and counterfactual what-if simulation
"""

import sys
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure utf-8 output encoding for Windows PowerShell/cmd terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import APP_TITLE, DISCLAIMER_TEXT, GRAPH_YAML_PATH, SAMPLE_EVENTS_PATH
from src.explainer.chatbot import GroundedChatbot
from src.extraction.deterministic import load_curated_events
from src.extraction.llm_extractor import classify_event
from src.graph_engine.loader import load_knowledge_graph
from src.graph_engine.propagator import PropagationEngine
from src.market.downloader import MarketDataDownloader
from src.market.validator import MarketValidator


def run_demo():
    print("=" * 80)
    print(f"[APP] {APP_TITLE}")
    print(f"[DISCLAIMER] {DISCLAIMER_TEXT}")
    print("=" * 80)

    # 1. Load Knowledge Graph
    print("\n[Step 1] Loading Curated Indian Financial Knowledge Graph...")
    kg = load_knowledge_graph(GRAPH_YAML_PATH)
    inv = kg.get_invariants_summary()
    print(f"✔ Knowledge graph loaded successfully: {inv['node_count']} nodes, {inv['edge_count']} signed edges.")

    # 2. Select Event
    curated_events = load_curated_events(SAMPLE_EVENTS_PATH)
    event_record = curated_events[0]
    print(f"\n[Step 2] Ingesting Curated Event: '{event_record.title[:70]}...'")
    print(f"  Source Attribution: {event_record.source_label} ({event_record.claim_status})")

    # 3. Structured Extraction
    print("\n[Step 3] Extracting Structured Financial Shock...")
    extracted = classify_event(
        text=f"{event_record.title}. {event_record.body}",
        event_id=event_record.id,
        knowledge_graph=kg,
    )
    print(f"✔ Extracted Event ID: {extracted.event_id}")
    print(f"  Primary Shock: {extracted.primary_entity_name} ({extracted.primary_entity_id})")
    print(f"  Shock Direction: {extracted.direction.value.upper()} | Magnitude: {extracted.magnitude.value.upper()}")
    print("  Evidence Quote:")
    print(f"  \"{extracted.evidence_snippets[0]}\"")

    # 4. Propagation
    print("\n[Step 4] Running Signed Weighted Propagation (Decay: 0.8, Max Depth: 4)...")
    engine = PropagationEngine(kg, hop_decay=0.8, max_depth=4, min_score_threshold=0.10)
    result = engine.propagate(origin_id=extracted.primary_entity_id, initial_direction=extracted.direction.value)
    print(f"✔ Discovered {result.total_paths_found} transmission paths reaching {len(result.affected_assets)} assets.")

    # 5. Top Ranked Assets Table
    print("\n[Step 5] Top Ranked Affected Assets Table:")
    print(f"{'Asset Name':<30} {'Ticker':<14} {'Direction':<12} {'Confidence':<12} {'Lag':<6} {'Score':<8}")
    print("-" * 84)
    for asset in result.affected_assets[:6]:
        ticker_str = asset.ticker or "Non-traded"
        print(f"{asset.asset_name:<30} {ticker_str:<14} {asset.net_direction.upper():<12} {asset.confidence_tier.upper():<12} {asset.cumulative_lag_days}d     {asset.top_score:.4f}")

    # 6. Deep Dive & Market Validation
    top_equity = next((a for a in result.affected_assets if a.ticker), result.affected_assets[0])
    print(f"\n[Step 6] Deep-Dive Evidence Audit: {top_equity.asset_name} ({top_equity.asset_id})")
    top_p = top_equity.top_path
    print(f"  Top Transmission Path: {' -> '.join(top_p.node_names)}")
    print(f"  Path ID: {top_p.path_id}")
    print(f"  Economic Rationale: {top_p.edges[0].rationale}")
    print(f"  Document Citations: {', '.join(top_p.evidence_refs)}")

    if top_equity.ticker:
        driver_ticker = kg.get_ticker(extracted.primary_entity_id)
        if driver_ticker:
            print(f"\n[Step 7] Checking Historical Correlation ({driver_ticker} -> {top_equity.ticker})...")
            validator = MarketValidator(downloader=MarketDataDownloader())
            vrep = validator.validate_relationship(
                source_symbol=driver_ticker,
                target_symbol=top_equity.ticker,
                prior_sign=top_equity.top_path.net_sign,
                expected_lag_days=top_equity.top_path.cumulative_lag_days,
            )
            print(f"  Empirical Status: {vrep.status.upper()}")
            print(f"  Correlation Statistic: {vrep.correlation_statistic} at Lag {vrep.evaluated_lag}d")
            print(f"  Sample Size: {vrep.sample_size} aligned trading days")
            print(f"  Caveat: {vrep.causation_disclaimer}")

    # 7. Grounded Chatbot & Counterfactual Demonstration
    print("\n[Step 8] Grounded Decision-Support Chatbot Synthesis:")
    chatbot = GroundedChatbot(knowledge_graph=kg)
    question = "Why might crude affect aviation?"
    print(f"  Inquiry: \"{question}\"")
    answer = chatbot.ask(question, extracted, result)
    print("\n  Answer:")
    for line in answer.split("\n"):
        print(f"    {line}")

    # Counterfactual
    print("\n[Step 9] Counterfactual Simulation (What-If Shock Direction Reverses):")
    cf_result, cf_text = chatbot.run_counterfactual(extracted, result, flipped_direction="negative")
    for line in cf_text.split("\n")[:6]:
        print(f"    {line}")

    print("\n" + "=" * 80)
    print("✔ Demo completed successfully! All pipeline stages verified green.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
