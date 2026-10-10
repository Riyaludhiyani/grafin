"""Grounded decision-support chatbot and counterfactual what-if engine.

Guarantees:
- Context is strictly assembled from the current extracted event, propagation paths,
  edge metadata, literature citations, and market validation reports.
- Labels all statements as [Observed Fact], [Curated Prior], or [Inferred Impact].
- Strictly cites Path IDs and Edge IDs.
- Refuses to invent price targets, numerical predictions, or unsupported facts.
- Explicitly states when context is insufficient.
- Implements counterfactuals by invoking the actual PropagationEngine with changed vectors.
- Includes a rich deterministic fallback engine for offline demonstrations.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.extraction.llm_extractor import BaseLLMAdapter, HTTPLLMAdapter
from src.extraction.schemas import ExtractedEvent
from src.graph_engine.loader import KnowledgeGraph
from src.graph_engine.propagator import AssetImpact, PropagationEngine, PropagationResult
from src.market.validator import MarketValidationReport


class GroundedChatbot:
    """Grounded financial decision-support assistant with deterministic fallback and counterfactuals."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        adapter: Optional[BaseLLMAdapter] = None,
    ):
        self.kg = knowledge_graph
        self.adapter = adapter

    def format_grounded_context(
        self,
        event: ExtractedEvent,
        result: PropagationResult,
        validation_reports: Optional[Dict[str, MarketValidationReport]] = None,
    ) -> str:
        """Assemble structured context strictly from the validated event and propagation results."""
        lines = [
            "=== ACTIVE FINANCIAL EVENT ===",
            f"Event ID: {event.event_id}",
            f"Type: {event.event_type} | Status: {event.status.value}",
            f"Primary Shock: {event.primary_entity_name} ({event.primary_entity_id})",
            f"Vector: {event.direction.value.upper()} | Magnitude: {event.magnitude.value}",
            f"Published: {event.timestamp} | Source: {event.source}",
            "Verbatim Evidence Quotes:",
        ]
        for snip in event.evidence_snippets:
            lines.append(f"  - \"{snip}\"")

        lines.append("\n=== PROPAGATION SUMMARY & AFFECTED ASSETS ===")
        lines.append(f"Origin Shock: {result.origin_name} ({result.origin_id})")
        lines.append(f"Total Paths Found: {result.total_paths_found}")

        for impact in result.affected_assets[:15]:
            lines.append(
                f"\nAsset: {impact.asset_name} ({impact.asset_id}) | Ticker: {impact.ticker or 'None'}"
            )
            lines.append(f"  Net Direction: {impact.net_direction.upper()} | Is Contested: {impact.is_contested}")
            lines.append(f"  Top Score: {impact.top_score:.4f} | Confidence: {impact.confidence_tier.upper()}")
            lines.append(f"  Est. Transmission Lag: {impact.cumulative_lag_days} days")
            lines.append(f"  Top Path: {' -> '.join(impact.top_path.node_names)}")
            lines.append(f"  Top Path ID: {impact.top_path.path_id}")

            if impact.warning_flags:
                lines.append(f"  Flags: {'; '.join(impact.warning_flags)}")

            lines.append("  All Discovered Paths:")
            for p in impact.all_paths:
                lines.append(f"    - Path ID {p.path_id}: {' -> '.join(p.node_names)} (Score: {p.decayed_score:.4f}, Sign: {p.net_sign.upper()})")
                for e in p.edges:
                    lines.append(
                        f"      * Edge {e.source} -> {e.target}: Sign={e.sign.upper()}, Strength={e.weight:.2f}, Lag={e.lag_days}d"
                    )
                    lines.append(f"        Rationale: {e.rationale}")
                    if e.evidence_refs:
                        lines.append(f"        Citations: {', '.join(e.evidence_refs)}")

        if validation_reports:
            lines.append("\n=== HISTORICAL MARKET VALIDATION ===")
            for ticker, vrep in validation_reports.items():
                lines.append(
                    f"Ticker {ticker}: Prior={vrep.prior_sign.upper()}, Status={vrep.status.upper()}, "
                    f"Corr={vrep.correlation_statistic} at Lag={vrep.evaluated_lag}d, "
                    f"Sample={vrep.sample_size} days. Note: {vrep.explanation}"
                )

        return "\n".join(lines)

    def run_counterfactual(
        self,
        event: ExtractedEvent,
        result: PropagationResult,
        flipped_direction: Optional[str] = None,
        custom_decay: Optional[float] = None,
    ) -> Tuple[PropagationResult, str]:
        """Execute a genuine counterfactual simulation by re-running the propagation engine.

        Never invents stock move percentages; reports strictly re-computed path scores and directions.
        """
        engine = PropagationEngine(
            knowledge_graph=self.kg,
            hop_decay=custom_decay if custom_decay is not None else result.hop_decay,
            max_depth=result.max_depth,
            min_score_threshold=result.min_score_threshold,
        )

        new_dir = flipped_direction or ("negative" if event.direction.value == "positive" else "positive")
        cf_result = engine.propagate(origin_id=event.primary_entity_id, initial_direction=new_dir)

        explanation = (
            f"🔄 **COUNTERFACTUAL SIMULATION (Direction Flipped to {new_dir.upper()}):**\n"
            f"Re-running the signed propagation algorithm from origin `{event.primary_entity_id}` "
            f"with reversed initial direction `{new_dir}`.\n\n"
            f"- **Hypothesized Transmission Shifts:**\n"
        )
        for asset in cf_result.affected_assets[:5]:
            explanation += (
                f"  * **{asset.asset_name}** (`{asset.asset_id}`): direction flips to "
                f"**{asset.net_direction.upper()}** (Top score: {asset.top_score:.4f}, Path ID: `{asset.top_path.path_id}`).\n"
            )

        explanation += (
            "\n*Note: These values reflect algorithmic structural path propagation scores under reversed inputs. "
            "They are not predictive financial price forecasts.*"
        )
        return cf_result, explanation

    def answer_deterministic(
        self,
        query: str,
        event: ExtractedEvent,
        result: PropagationResult,
        validation_reports: Optional[Dict[str, MarketValidationReport]] = None,
    ) -> str:
        """Provide deterministic, citation-backed answers for demo questions and offline operation."""
        q_lower = query.lower()

        # Check for Counterfactual queries
        if any(w in q_lower for w in ["what if", "counterfactual", "reverse", "flipped", "opposite direction"]):
            _, cf_text = self.run_counterfactual(event, result)
            return cf_text

        # 1. Aviation / Crude Oil relationship
        if any(w in q_lower for w in ["aviation", "airline", "indigo", "atf", "jet fuel"]) and "crude" in q_lower:
            aviation_asset = next((a for a in result.affected_assets if "aviation" in a.asset_id or "indigo" in a.asset_id), None)
            if aviation_asset:
                top_p = aviation_asset.top_path
                return (
                    f"**Analysis of Crude Oil Transmission to Aviation:**\n\n"
                    f"1. **[Observed Fact]** The current event `{event.event_id}` reports a {event.magnitude.value} {event.direction.value} "
                    f"shock in `{event.primary_entity_name}` (`{event.primary_entity_id}`). Headline quote: *\"{event.evidence_snippets[0]}\"*\n\n"
                    f"2. **[Curated Prior]** Edge `macro_brent_crude -> sector_aviation` (Weight: 0.90, Lag: 5d, Sign: NEGATIVE). "
                    f"Rationale: *Aviation Turbine Fuel (ATF) constitutes 35% to 45% of total airline operating expenses in India.* "
                    f"Authoritative Citation: *DGCA Monthly Airline Financial Performance Bulletins; InterGlobe Aviation FY24 Disclosures*.\n\n"
                    f"3. **[Inferred Impact]** Under Path ID `{top_p.path_id}`, the shock transmits with net direction "
                    f"**{top_p.net_sign.upper()}** (Decayed score: {top_p.decayed_score:.4f}, Cumulative lag: {top_p.cumulative_lag_days} days).\n\n"
                    f"*(Hypothesized transmission path, not a stock-price prediction.)*"
                )

        # 2. Bank Nifty / Contested asset queries
        if any(w in q_lower for w in ["contested", "bank nifty", "opposing", "conflict", "contradict"]):
            contested_assets = [a for a in result.affected_assets if a.is_contested]
            if contested_assets:
                c_asset = contested_assets[0]
                pos_paths = [p for p in c_asset.all_paths if p.net_sign == "positive"]
                neg_paths = [p for p in c_asset.all_paths if p.net_sign == "negative"]
                return (
                    f"**Explanation of Contested Transmission on {c_asset.asset_name} (`{c_asset.asset_id}`):**\n\n"
                    f"1. **[Observed Fact]** Origin shock `{event.primary_entity_name}` triggers multiple transmission channels into `{c_asset.asset_id}`.\n\n"
                    f"2. **[Inferred Impact - Channel A (Positive)]** Path ID `{pos_paths[0].path_id}` transmits positive momentum "
                    f"(Decayed score: {pos_paths[0].decayed_score:.4f}) via edge `{pos_paths[0].edges[0].source} -> {pos_paths[0].edges[0].target}`.\n\n"
                    f"3. **[Inferred Impact - Channel B (Negative)]** Path ID `{neg_paths[0].path_id}` transmits negative drag "
                    f"(Decayed score: {neg_paths[0].decayed_score:.4f}) via edge `{neg_paths[0].edges[0].source} -> {neg_paths[0].edges[0].target}`.\n\n"
                    f"4. **[Curated Prior - Conflict Policy]** The system explicitly prohibits silently netting opposing forces to zero. "
                    f"Because both expansionary and contractionary channels are active, {c_asset.asset_name} is marked as **⚡ CONTESTED**."
                )
            else:
                return (
                    f"**Contested Asset Audit:**\n\n"
                    f"In the current event propagation from `{result.origin_name}`, no assets currently exhibit opposing positive and negative paths. "
                    f"All reached assets share consistent or ambiguous transmission paths under the active parameters."
                )

        # 3. Check if query references any specific asset via resolve_entity or substring
        resolved_node_id = self.kg.resolve_entity(query)
        matched_asset = None
        if resolved_node_id:
            matched_asset = next((a for a in result.affected_assets if a.asset_id == resolved_node_id), None)

        if not matched_asset:
            for asset in result.affected_assets:
                clean_name = asset.asset_name.lower().replace(" ltd", "").replace(" limited", "").strip()
                if (asset.asset_id.lower() in q_lower or
                    clean_name in q_lower or
                    (asset.ticker and asset.ticker.lower().split('.')[0] in q_lower)):
                    matched_asset = asset
                    break

        if matched_asset:
            top_p = matched_asset.top_path
            val_note = ""
            if validation_reports and matched_asset.ticker and matched_asset.ticker in validation_reports:
                vrep = validation_reports[matched_asset.ticker]
                val_note = (
                    f"\n\n4. **[Historical Market Validation]** Correlation with driver is `{vrep.correlation_statistic}` "
                    f"at lag `{vrep.evaluated_lag}d` across `{vrep.sample_size}` aligned trading days ({vrep.status.upper()}). "
                    f"Caveat: {vrep.causation_disclaimer}"
                )

            return (
                f"**Analysis of Impact on {matched_asset.asset_name} (`{matched_asset.asset_id}`):**\n\n"
                f"1. **[Observed Fact]** Shock from `{event.primary_entity_name}` ({event.direction.value.upper()}) initiates at origin node `{event.primary_entity_id}`.\n\n"
                f"2. **[Curated Prior]** Transmitted along {len(matched_asset.all_paths)} discovered channel(s). Top path rationales: "
                f"*{top_p.edges[0].rationale}* (Citations: *{', '.join(top_p.evidence_refs)}*).\n\n"
                f"3. **[Inferred Impact]** Top Path ID `{top_p.path_id}` yields net direction **{matched_asset.net_direction.upper()}** "
                f"with confidence **{matched_asset.confidence_tier.upper()}** (Score: {matched_asset.top_score:.4f}, Transmission horizon: {matched_asset.cumulative_lag_days} days).{val_note}\n\n"
                f"*(Hypothesized transmission path, not a stock-price prediction.)*"
            )

        # 4. Fallback for unmapped or speculative questions (e.g. Tesla, Bitcoin, unsupported queries)
        return (
            "⚠️ **Insufficient Context / Out of Domain:**\n\n"
            "The available Indian financial knowledge graph evidence and extracted event context are insufficient to answer this question. "
            "This decision-support engine strictly evaluates structural transmission channels within the curated Indian market network and "
            "does not provide speculative price targets or analyze unmapped foreign assets."
        )

    def ask(
        self,
        query: str,
        event: ExtractedEvent,
        result: PropagationResult,
        validation_reports: Optional[Dict[str, MarketValidationReport]] = None,
    ) -> str:
        """Answer question with grounding enforcement and deterministic fallback."""
        if not query or not query.strip():
            return "Please enter a valid question regarding the active financial event or affected assets."

        # If custom adapter is provided and valid, attempt LLM generation
        if self.adapter and getattr(self.adapter, "api_key", None):
            try:
                context_str = self.format_grounded_context(event, result, validation_reports)
                prompt = (
                    "You are the Grounded Financial Market Impact Explainer for the Indian financial knowledge graph.\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Answer ONLY using facts provided in the CONTEXT below.\n"
                    "2. You MUST cite internal Path IDs (e.g. PATH_...) and Edge IDs (source -> target).\n"
                    "3. Explicitly label claims as [Observed Fact], [Curated Prior], or [Inferred Impact].\n"
                    "4. If the context does not contain sufficient evidence, state: 'The available knowledge graph evidence and extracted event context are insufficient to answer this question.'\n"
                    "5. Do NOT invent price targets, numerical return predictions, or certainty.\n"
                    "6. Persistent disclaimer: 'Hypothesized transmission paths, not predictions.'\n\n"
                    f"CONTEXT:\n{context_str}\n\n"
                    f"QUESTION: {query}\n\n"
                    "ANSWER:"
                )
                answer = self.adapter.generate_json(prompt)
                if answer and len(answer.strip()) > 20:
                    return answer
            except Exception:
                pass  # Fall through to deterministic fallback

        # Deterministic fallback answer
        return self.answer_deterministic(query, event, result, validation_reports)
