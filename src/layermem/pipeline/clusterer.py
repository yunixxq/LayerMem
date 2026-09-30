"""Online incremental topic clustering (LayerMem.md §2.2, semantic axis).

Every new atom is embedded and compared against the atoms already stored *in
its own perspective*. Whichever topic the nearest neighbour belongs to is the
topic the new atom joins; below ``theta_cluster`` a fresh topic is created.

Two design points from the design doc drive the implementation:

* Topics are identified by a **stable ID** that atoms carry as a member set —
  not by a position range — so inserting never cascades an update.
* Each perspective clusters in its **own** semantic space. Comparing a state
  atom against factual atoms would merge topics that merely happen to be
  worded alike.
"""

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
    """Where an atom landed, and whether it is a repeat of an existing one."""

    topic_id: str
    is_new_topic: bool
    duplicate_of: Optional[Entry] = None
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

    def assign(self, entry: Entry, vector: Optional[Sequence[float]] = None) -> Assignment:
        """Decide the topic of ``entry`` (does not write it)."""
        vector = vector if vector is not None else self.embed(entry.memory)

        neighbours = self.store.search(
            vector,
            limit=5,
            entry_type=entry.entry_type,
            layer="L1",
        )
        if not neighbours:
            return Assignment(topic_id=str(uuid.uuid4()), is_new_topic=True)

        best_entry, best_score = neighbours[0]
        if best_score < self.config.theta_cluster or not best_entry.topic_id:
            return Assignment(
                topic_id=str(uuid.uuid4()),
                is_new_topic=True,
                similarity=best_score,
            )

        # A near-identical restatement is not a new memory: count the mention
        # instead of storing a duplicate row (LayerMem_imple.md §七.9).
        duplicate = best_entry if best_score >= self.config.theta_dup else None
        return Assignment(
            topic_id=best_entry.topic_id,
            is_new_topic=False,
            duplicate_of=duplicate,
            similarity=best_score,
        )

    def topic_members(
        self,
        topic_id: str,
        entry_type: Optional[str] = None,
        layer: str = "L1",
        limit: int = 1000,
    ) -> List[Entry]:
        """All atoms currently in a topic, oldest first."""
        entries = self.store.scroll(
            limit=limit,
            topic_id=topic_id,
            entry_type=entry_type,
            layer=layer,
        )
        return sorted(entries, key=lambda e: e.event_key)
