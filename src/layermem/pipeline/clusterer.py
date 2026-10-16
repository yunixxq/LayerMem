from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import Entry
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Assignment:
    """Where an atom landed."""

    topic_id: str
    is_new_topic: bool
    similarity: float = 0.0


class Clusterer:
    """Assigns ``topic_id`` to factual/relational atoms."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Any,
        config: Optional[LayerMemConfig] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or LayerMemConfig()

    def embed(self, text: str) -> Sequence[float]:
        return self.embedder.embed(text)

    def assign(
        self,
        entry: Entry,
        vector: Optional[Sequence[float]] = None,
        layer: Optional[str] = None,
    ) -> Assignment:
        """Decide the topic within one independent layer."""
        layer = layer or entry.layer
        vector = vector if vector is not None else self.embed(entry.memory)

        neighbours = self.store.search(
            vector,
            limit=5,
            entry_type=entry.entry_type,
            layer=layer,
        )
        if not neighbours:
            return Assignment(topic_id=str(uuid.uuid4()), is_new_topic=True)

        capacities = {
            "L1": self.config.K1,
            "L2": self.config.K2,
            "L3": self.config.K3,
            "L4": None,
        }
        capacity = capacities.get(layer)
        # Choose the nearest eligible topic. Each layer has an independent
        # topic namespace and capacity.
        for candidate, score in neighbours:
            if score < self.config.theta_cluster or not candidate.topic_id:
                continue
            if capacity is None:
                return Assignment(
                    topic_id=candidate.topic_id,
                    is_new_topic=False,
                    similarity=score,
                )
            members = self.store.scroll(
                limit=capacity,
                topic_id=candidate.topic_id,
                entry_type=entry.entry_type,
                layer=layer,
            )
            if len(members) < capacity:
                return Assignment(
                    topic_id=candidate.topic_id,
                    is_new_topic=False,
                    similarity=score,
                )

        best_score = neighbours[0][1]
        return Assignment(
            topic_id=str(uuid.uuid4()),
            is_new_topic=True,
            similarity=best_score,
        )

    def topic_members(
        self,
        topic_id: str,
        entry_type: Optional[str] = None,
        layer: str = "L1",
        limit: int = 1000,
    ) -> List[Entry]:
        """All atoms currently in a topic at the requested layer."""
        entries = self.store.scroll(
            limit=limit,
            topic_id=topic_id,
            entry_type=entry_type,
            layer=layer,
        )
        return entries
