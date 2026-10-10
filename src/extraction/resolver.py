"""Entity resolution and graph validation utility.

Ensures LLM or external extractions align strictly with the existing knowledge graph nodes.
Guarantees that no external extractor can create arbitrary, unverified graph edges.
"""

from typing import List, Optional
from src.extraction.schemas import MentionedEntity
from src.graph_engine.loader import KnowledgeGraph


class EntityResolver:
    """Aligns textual entity mentions with validated KnowledgeGraph node IDs."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def validate_or_resolve_node_id(self, candidate_id_or_text: str) -> Optional[str]:
        """Validate candidate node ID or resolve alias to a legitimate graph node ID.

        Returns:
            Valid node ID string if present in graph, or resolved alias node ID, else None.
        """
        if not candidate_id_or_text:
            return None

        clean = candidate_id_or_text.strip()
        # 1. Exact node ID match
        if self.kg.get_node(clean) is not None:
            return clean

        # 2. Attempt alias resolution
        resolved = self.kg.resolve_entity(clean)
        if resolved and self.kg.get_node(resolved) is not None:
            return resolved

        return None

    def resolve_mentioned_entities(self, raw_mentions: List[MentionedEntity]) -> List[MentionedEntity]:
        """Resolve and normalize a list of MentionedEntity instances against graph aliases."""
        resolved_list: List[MentionedEntity] = []

        for mention in raw_mentions:
            candidate_id = mention.resolved_node_id or mention.text
            valid_id = self.validate_or_resolve_node_id(candidate_id)

            node = self.kg.get_node(valid_id) if valid_id else None
            entity_type = node.type.value if node else mention.entity_type

            resolved_list.append(
                MentionedEntity(
                    text=mention.text,
                    resolved_node_id=valid_id,
                    entity_type=entity_type,
                )
            )

        return resolved_list
