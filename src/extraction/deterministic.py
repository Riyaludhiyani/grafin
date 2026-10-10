"""Deterministic financial news event extractor and curated dataset loader.

Guarantees 100% offline execution without requiring LLM API credentials.
Resolves mentioned entities against graph aliases and extracts evidence snippets.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.config import SAMPLE_EVENTS_PATH
from src.extraction.schemas import (
    CuratedEventRecord,
    EventDirection,
    EventStatus,
    ExtractedEvent,
    MagnitudeTier,
    MentionedEntity,
)
from src.graph_engine.loader import KnowledgeGraph


def load_curated_events(json_path: Union[str, Path] = SAMPLE_EVENTS_PATH) -> List[CuratedEventRecord]:
    """Load and validate all curated sample events from JSON storage."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample events file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if not isinstance(raw_items, list):
        raise ValueError("Curated events file must contain a JSON array of event records.")

    events: List[CuratedEventRecord] = []
    for item in raw_items:
        validated_record = CuratedEventRecord.model_validate(item)
        events.append(validated_record)

    return events


def get_curated_event_by_id(
    event_id: str, json_path: Union[str, Path] = SAMPLE_EVENTS_PATH
) -> Optional[CuratedEventRecord]:
    """Retrieve a single curated event record by its unique ID."""
    events = load_curated_events(json_path)
    for ev in events:
        if ev.id == event_id:
            return ev
    return None


class DeterministicExtractor:
    """Offline rule-based event extractor using regex, heuristics, and graph alias lookup."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        # Cache curated events in memory
        try:
            self.curated_events: Dict[str, CuratedEventRecord] = {
                e.id: e for e in load_curated_events()
            }
        except Exception:
            self.curated_events = {}

    def parse_magnitude(self, text: str) -> Tuple[MagnitudeTier, Optional[float], Optional[str]]:
        """Parse numerical magnitude and unit from headline/text."""
        # Check bps / basis points
        bps_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:bps|basis points)", text, re.IGNORECASE)
        if bps_match:
            val = float(bps_match.group(1))
            tier = MagnitudeTier.SEVERE if val > 50 else (MagnitudeTier.MODERATE if val >= 25 else MagnitudeTier.MINOR)
            return tier, val, "bps"

        # Check percentage
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if pct_match:
            val = float(pct_match.group(1))
            tier = MagnitudeTier.SEVERE if val > 5.0 else (MagnitudeTier.MODERATE if val >= 2.0 else MagnitudeTier.MINOR)
            return tier, val, "%"

        # Check paise
        paise_match = re.search(r"(\d+(?:\.\d+)?)\s*paise", text, re.IGNORECASE)
        if paise_match:
            val = float(paise_match.group(1))
            tier = MagnitudeTier.SEVERE if val > 50 else MagnitudeTier.MODERATE
            return tier, val, "paise"

        # Categorical fallback keywords
        lower = text.lower()
        if any(w in lower for w in ["massive", "historic", "record", "surges", "spikes", "plunges", "emergency"]):
            return MagnitudeTier.SEVERE, None, None
        if any(w in lower for w in ["moderate", "modest", "tweak", "slight"]):
            return MagnitudeTier.MINOR, None, None

        return MagnitudeTier.MODERATE, None, None

    def parse_direction(self, text: str) -> EventDirection:
        """Infer shock vector from action verbs in headline and body."""
        lower = text.lower()
        pos_words = [
            "surge", "surges", "spike", "spikes", "jump", "jumps", "rally", "rallies",
            "hike", "hikes", "rise", "rises", "gain", "gains", "soar", "soars",
            "higher", "inflation surge", "climb", "escalate"
        ]
        neg_words = [
            "fall", "falls", "plunge", "plunges", "tumble", "tumbles", "slide", "slides",
            "cut", "cuts", "drop", "drops", "decline", "declines", "depreciate", "depreciates",
            "slips", "lower", "crash", "sell-off"
        ]

        pos_count = sum(len(re.findall(rf"\b{w}\b", lower)) for w in pos_words)
        neg_count = sum(len(re.findall(rf"\b{w}\b", lower)) for w in neg_words)

        if pos_count > neg_count:
            return EventDirection.POSITIVE
        elif neg_count > pos_count:
            return EventDirection.NEGATIVE
        return EventDirection.AMBIGUOUS

    def extract_evidence_snippets(self, text: str, max_sentences: int = 4) -> List[str]:
        """Extract key sentences containing factual claims as evidence snippets."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return clean_sentences[:max_sentences] if clean_sentences else [text[:200]]

    def extract_entities(self, text: str) -> List[MentionedEntity]:
        """Scan text for mentions matching graph aliases or colloquial financial terms."""
        found: List[MentionedEntity] = []
        seen_texts = set()

        # Check known graph aliases
        for alias, node_id in self.kg.alias_lookup.items():
            if len(alias) >= 3:
                pattern = rf"\b{re.escape(alias)}\b"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    raw_matched = match.group(0)
                    if raw_matched.lower() not in seen_texts:
                        seen_texts.add(raw_matched.lower())
                        node = self.kg.get_node(node_id)
                        entity_type = node.type.value if node else None
                        found.append(
                            MentionedEntity(
                                text=raw_matched,
                                resolved_node_id=node_id,
                                entity_type=entity_type,
                            )
                        )

        return found

    def extract_event(
        self,
        headline: str,
        body: str,
        event_id: Optional[str] = None,
        source: str = "Deterministic Rule-Based Parser",
        publication_date: str = "2026-10-11T00:00:00Z",
    ) -> ExtractedEvent:
        """Extract structured event without calling any external LLM service.

        If event_id corresponds to a curated event, returns the curated extraction directly.
        Otherwise, runs rule-based heuristic extraction with alias resolution.
        """
        # 1. Fast path: check if this is one of our curated events
        if event_id and event_id in self.curated_events:
            return self.curated_events[event_id].cached_extraction

        # Also match by headline prefix in curated events
        for cur_ev in self.curated_events.values():
            if cur_ev.title.lower() in headline.lower() or headline.lower() in cur_ev.title.lower():
                return cur_ev.cached_extraction

        # 2. Rule-based extraction
        combined_text = f"{headline}. {body}"
        mag_tier, mag_num, mag_unit = self.parse_magnitude(headline)
        direction = self.parse_direction(headline)
        evidence = self.extract_evidence_snippets(body)
        entities = self.extract_entities(combined_text)

        # Determine primary origin node
        primary_id = "macro_brent_crude"  # safe fallback
        primary_name = "Brent Crude Oil"

        # Prioritize resolved macro indicator or commodity if found
        for ent in entities:
            if ent.resolved_node_id:
                node = self.kg.get_node(ent.resolved_node_id)
                if node and node.type.value in ("commodity", "macro_indicator", "currency", "sector"):
                    primary_id = ent.resolved_node_id
                    primary_name = node.name
                    break
        else:
            # If no macro entity, take first resolved entity
            for ent in entities:
                if ent.resolved_node_id:
                    node = self.kg.get_node(ent.resolved_node_id)
                    if node:
                        primary_id = ent.resolved_node_id
                        primary_name = node.name
                        break

        # Event type classification heuristic
        lower_hl = headline.lower()
        if "crude" in lower_hl or "oil" in lower_hl:
            ev_type = "commodity_shock"
        elif "repo rate" in lower_hl or "policy rate" in lower_hl or "mpc" in lower_hl:
            ev_type = "monetary_policy"
        elif "rupee" in lower_hl or "usd" in lower_hl or "inr" in lower_hl:
            ev_type = "currency_depreciation"
        elif "cpi" in lower_hl or "inflation" in lower_hl:
            ev_type = "inflation_surge"
        elif "risk weight" in lower_hl or "regulation" in lower_hl or "norm" in lower_hl:
            ev_type = "regulatory_mandate"
        elif "yield" in lower_hl or "fii" in lower_hl or "treasury" in lower_hl:
            ev_type = "capital_flows"
        else:
            ev_type = "macroeconomic_event"

        return ExtractedEvent(
            event_id=event_id or f"event_gen_{abs(hash(headline)) % 100000}",
            event_type=ev_type,
            status=EventStatus.OBSERVED,
            direction=direction,
            magnitude=mag_tier,
            magnitude_numeric=mag_num,
            magnitude_unit=mag_unit,
            geography="India",
            timestamp=publication_date,
            source=source,
            primary_entity_id=primary_id,
            primary_entity_name=primary_name,
            mentioned_entities=entities,
            evidence_snippets=evidence,
        )
