"""Provider-independent LLM event extraction adapter with strict JSON schema validation.

Guarantees:
- Pluggable adapter interface (Gemini / OpenAI / Groq / HTTP).
- Never allows LLM to create arbitrary or unverified propagation edges.
- Validates every response against ExtractedEvent schema.
- Normalizes entity mentions and resolves aliases.
- Automatically falls back to deterministic/curated extraction on missing credentials,
  network timeouts, invalid JSON, missing fields, or hallucinated node IDs.
"""

import abc
import json
import logging
import os
import re
from typing import Optional
import requests

from src.config import (
    GRAPH_YAML_PATH,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    OFFLINE_DEMO_MODE,
)
from src.extraction.deterministic import DeterministicExtractor
from src.extraction.resolver import EntityResolver
from src.extraction.schemas import ExtractedEvent
from src.graph_engine.loader import KnowledgeGraph, load_knowledge_graph

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Raised when an external LLM request fails, times out, or produces invalid output."""
    pass


class BaseLLMAdapter(abc.ABC):
    """Abstract interface for pluggable LLM extraction providers."""

    @abc.abstractmethod
    def generate_json(self, prompt: str) -> str:
        """Submit prompt and return raw text string containing JSON response."""
        pass


class HTTPLLMAdapter(BaseLLMAdapter):
    """Configurable HTTP adapter supporting Gemini and OpenAI-compatible API schemas."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.provider = (provider or LLM_PROVIDER).lower().strip()
        self.api_key = api_key if api_key is not None else LLM_API_KEY
        self.model = model or LLM_MODEL
        self.timeout = timeout if timeout is not None else LLM_TIMEOUT_SECONDS

    def generate_json(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMGenerationError("LLM API key is not configured in environment.")

        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider in ("openai", "groq"):
            return self._call_openai_compatible(prompt)
        else:
            raise LLMGenerationError(f"Unsupported or disabled LLM provider: '{self.provider}'")

    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.Timeout as e:
            raise LLMGenerationError(f"Gemini API request timed out after {self.timeout}s: {e}") from e
        except Exception as e:
            raise LLMGenerationError(f"Gemini API invocation error: {e}") from e

    def _call_openai_compatible(self, prompt: str) -> str:
        if self.provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout as e:
            raise LLMGenerationError(f"LLM API request timed out after {self.timeout}s: {e}") from e
        except Exception as e:
            raise LLMGenerationError(f"OpenAI-compatible API invocation error: {e}") from e


def clean_llm_json_output(raw_text: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")].strip()
    return cleaned


def build_extraction_prompt(news_text: str, kg: KnowledgeGraph) -> str:
    """Build a constrained prompt instructing the LLM to output valid JSON matching our schema."""
    known_node_ids = list(kg.nodes_data.keys())[:35]
    sample_nodes_str = ", ".join(known_node_ids)

    return f"""You are a specialized financial event data extraction engine for the Indian market.
Analyze the following financial news and return a strict JSON object.

CRITICAL CONSTRAINTS:
1. You MUST select primary_entity_id from existing knowledge graph nodes. Examples:
   [{sample_nodes_str}]
2. Do NOT invent new nodes or arbitrary graph edges.
3. Include at least two verbatim evidence snippets from the text.
4. Set direction to "positive" (increase/appreciation/hike), "negative" (decrease/depreciation/cut), or "ambiguous".
5. Set magnitude to "minor", "moderate", or "severe".
6. Return ONLY valid JSON conforming to this schema.

NEWS TEXT:
\"\"\"{news_text}\"\"\"

Required JSON format:
{{
  "event_id": "event_extracted",
  "event_type": "commodity_shock | monetary_policy | currency_depreciation | inflation_surge | regulatory_mandate | capital_flows",
  "status": "observed",
  "direction": "positive | negative | ambiguous",
  "magnitude": "minor | moderate | severe",
  "magnitude_numeric": null,
  "magnitude_unit": null,
  "geography": "India",
  "timestamp": "2024-04-15T09:30:00Z",
  "source": "News Wire Extraction",
  "primary_entity_id": "<valid_node_id>",
  "primary_entity_name": "<node_name>",
  "mentioned_entities": [
    {{"text": "string", "resolved_node_id": "<valid_node_id_or_null>", "entity_type": "commodity | equity | etc"}}
  ],
  "evidence_snippets": [
    "verbatim sentence 1",
    "verbatim sentence 2"
  ]
}}
"""


def classify_event(
    text: str,
    event_id: Optional[str] = None,
    knowledge_graph: Optional[KnowledgeGraph] = None,
    adapter: Optional[BaseLLMAdapter] = None,
) -> ExtractedEvent:
    """Provider-independent event classification with schema enforcement and deterministic fallback.

    Args:
        text: News headline and article text.
        event_id: Optional identifier matching a curated event.
        knowledge_graph: Optional KnowledgeGraph instance (loaded if omitted).
        adapter: Optional custom BaseLLMAdapter (HTTPLLMAdapter created if omitted).

    Returns:
        Strictly validated ExtractedEvent with verified node IDs and evidence snippets.
    """
    kg = knowledge_graph or load_knowledge_graph(GRAPH_YAML_PATH)
    resolver = EntityResolver(kg)
    fallback_extractor = DeterministicExtractor(kg)

    # If custom adapter is provided, invoke it; otherwise check offline demo mode and API credentials
    if adapter is not None:
        llm_adapter = adapter
        should_attempt_llm = True
    else:
        llm_adapter = HTTPLLMAdapter()
        should_attempt_llm = (
            not OFFLINE_DEMO_MODE
            and bool(llm_adapter.api_key)
            and llm_adapter.provider != "none"
        )

    if not should_attempt_llm:
        logger.info("Using deterministic extraction fallback (offline demo mode or absent credentials).")
        return fallback_extractor.extract_event(headline=text, body=text, event_id=event_id)

    # Attempt live LLM extraction
    try:
        prompt = build_extraction_prompt(text, kg)
        raw_output = llm_adapter.generate_json(prompt)
        cleaned_json = clean_llm_json_output(raw_output)
        parsed_data = json.loads(cleaned_json)

        # Schema validation
        candidate_event = ExtractedEvent.model_validate(parsed_data)

        # Verify that primary_entity_id exists in the graph or can be resolved
        valid_primary_id = resolver.validate_or_resolve_node_id(candidate_event.primary_entity_id)
        if not valid_primary_id:
            raise LLMGenerationError(
                f"Model returned unresolvable primary_entity_id: '{candidate_event.primary_entity_id}'"
            )

        candidate_event.primary_entity_id = valid_primary_id
        primary_node = kg.get_node(valid_primary_id)
        if primary_node:
            candidate_event.primary_entity_name = primary_node.name

        # Normalize and resolve mentioned entities
        candidate_event.mentioned_entities = resolver.resolve_mentioned_entities(
            candidate_event.mentioned_entities
        )

        return candidate_event

    except Exception as e:
        logger.warning(
            f"Live LLM extraction encountered error: {e}. Falling back to deterministic extraction."
        )
        return fallback_extractor.extract_event(headline=text, body=text, event_id=event_id)
