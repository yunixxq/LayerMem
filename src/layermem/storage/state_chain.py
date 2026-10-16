"""State chains: SQLite nodes and a Qdrant attribute index (LayerMem.md §2.4).

Every assertion is a node, including repeated values at different times.
Current/historical are derived by event_time; mention_time and insertion id
break ties. No status flags, pointers, or stored validity intervals are needed.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import StateEntry
from layermem.memory.llm.utils import parse_json_object
from layermem.prompts.state import STATE_ATTRIBUTE_RESOLUTION_PROMPT
from layermem.storage.state_attribute import StateAttributeIndex
from layermem.storage.state_store import StateStore, attribute_key
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

ADD = "ADD"
UPDATE = "UPDATE"
HISTORICAL_INSERT = "HISTORICAL_INSERT"


@dataclass
class StateTransition:
    action: str
    entry: StateEntry


class StateChain:
    def __init__(
        self, store: VectorStore, embedder: Any,
        config: Optional[LayerMemConfig] = None, llm: Any = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or LayerMemConfig()
        self.llm = llm
        if not (0 <= self.config.state_attribute_low <= self.config.state_attribute_high <= 1):
            raise ValueError("State attribute thresholds must satisfy 0 <= low <= high <= 1")
        if self.config.state_attribute_top_k < 1:
            raise ValueError("state_attribute_top_k must be positive")
        path = self.config.state_db_path
        if path is None:
            path = self.default_db_path(store.path, store.collection_name)
        self.nodes = StateStore(path)
        self.index = StateAttributeIndex(store)
        self._index_ready = False

    @staticmethod
    def default_db_path(qdrant_path: str, collection: str) -> str:
        if qdrant_path == ":memory:":
            return ":memory:"
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, collection).hex
        return str(Path(qdrant_path) / f"state_chain_{namespace}.sqlite3")

    def close(self) -> None:
        self.nodes.close()

    def handle(self, entry: StateEntry) -> StateTransition:
        """Resolve the attribute, then INSERT a node into the chosen chain."""
        if not isinstance(entry, StateEntry):
            raise TypeError("StateChain accepts StateEntry only")
        if not all(value.strip() for value in (entry.subject, entry.attribute, entry.value, entry.memory)):
            raise ValueError("State nodes require subject, attribute, value, and memory")
        self._ensure_index()
        alias = attribute_key(entry.attribute)
        attribute, vector = self._resolve_attribute(entry.subject, alias)
        entry = replace(entry, attribute=attribute)
        current = self.get_current(entry.subject, attribute)
        if current is None:
            action = ADD
            # Index first. An orphan from a failed SQLite write is ignored by
            # resolution; retrying uses the same deterministic point id.
            if vector is None:
                vector = self.embedder.embed(attribute)
            self.index.upsert(entry.subject, attribute, vector)
        elif entry.event_key < current.event_key:
            action = HISTORICAL_INSERT
        else:
            action = UPDATE
        self.nodes.upsert(entry, alias)
        logger.debug("state %s: %s/%s", action, entry.subject, attribute)
        return StateTransition(action, entry)

    def _ensure_index(self) -> None:
        if self._index_ready:
            return
        self.index.ensure_collection()
        # Only writes rebuild auxiliary points. Reads need neither embeddings
        # nor Qdrant, including when the index has been lost.
        for item in self.nodes.attributes():
            subject, attribute = item["subject"], item["attribute"]
            if not self.index.contains(subject, attribute):
                self.index.upsert(subject, attribute, self.embedder.embed(attribute))
        self._index_ready = True

    def _resolve_attribute(self, subject: str, attribute: str) -> tuple[str, Any]:
        canonical = self.nodes.resolve_alias(subject, attribute)
        if self.nodes.has_chain(subject, canonical):
            return canonical, None
        vector = self.embedder.embed(attribute)
        candidates = [
            (name, score) for name, score in self.index.candidates(
                subject, vector, self.config.state_attribute_top_k,
                self.config.state_attribute_low,
            ) if self.nodes.has_chain(subject, name)
        ]
        if not candidates:
            return attribute, vector
        if candidates[0][1] >= self.config.state_attribute_high:
            return candidates[0][0], None
        if self.llm is None:
            raise ValueError("An LLM is required to resolve ambiguous State attributes")
        text, _ = self.llm.generate_response([{
            "role": "user",
            "content": STATE_ATTRIBUTE_RESOLUTION_PROMPT + "\n\n" + json.dumps({
                "subject": subject, "new_attribute": attribute,
                "candidates": [name for name, _ in candidates],
            }, ensure_ascii=False),
        }])
        decision = parse_json_object(text)
        if "attribute" not in decision:
            raise ValueError("Attribute resolution response must contain 'attribute'")
        chosen = decision["attribute"]
        if chosen is None:
            return attribute, vector
        if chosen not in [name for name, _ in candidates]:
            raise ValueError("Attribute resolution must select a candidate or null")
        return chosen, None

    def query(
        self, subject: str, attribute: str, query_type: str = "current",
        time_lower: str | float | None = None,
        time_upper: str | float | None = None,
    ) -> list[StateEntry]:
        return self.nodes.query(subject, attribute, query_type, time_lower, time_upper)

    def get_current(self, subject: str, attribute: str) -> Optional[StateEntry]:
        result = self.query(subject, attribute)
        return result[0] if result else None

    def get_at_time(
        self, subject: str, attribute: str, target_ts: str | float,
    ) -> Optional[StateEntry]:
        if target_ts is None:
            raise ValueError("A target time is required")
        result = self.query(subject, attribute, time_upper=target_ts)
        return result[0] if result else None

    def versions(self, subject: str, attribute: str) -> list[StateEntry]:
        return self.query(subject, attribute, "all")

    def current_attributes(self, limit: int = 500) -> list[StateEntry]:
        return self.nodes.current_attributes(limit)

    def count(self) -> int:
        return self.nodes.count()

    def update(self, entry_id: str, patch: dict[str, Any]) -> bool:
        entry = self.nodes.get_by_id(entry_id)
        if entry is None:
            return False
        if set(patch) - {"memory", "subject", "attribute", "value", "event_time", "mention_time"}:
            raise ValueError("Unsupported State update fields")
        self.handle(replace(entry, **patch))
        return True

    def migrate_legacy(self) -> int:
        """Import old Qdrant state nodes once; preserve the old collection."""
        if self.nodes.has_metadata("legacy_qdrant_imported"):
            return 0
        collection = f"{self.store.collection_name}_state"
        imported = 0
        if self.store.client.collection_exists(collection):
            offset = None
            while True:
                records, offset = self.store.client.scroll(
                    collection_name=collection, offset=offset, limit=256,
                    with_payload=True, with_vectors=False,
                )
                for record in records:
                    payload = record.payload
                    event = payload.get("event_time", payload.get("float_event_time"))
                    mention = payload.get("mention_time", payload.get("float_mention_time"))
                    if event is None or mention is None:
                        raise ValueError(f"Legacy State {record.id} has missing timestamps")
                    attribute = attribute_key(payload["attribute"])
                    entry = StateEntry(
                        id=str(record.id), subject=payload["subject"], attribute=attribute,
                        value=payload["value"], memory=payload["memory"],
                        event_time=event, mention_time=mention,
                    )
                    if self.nodes.get_by_id(entry.id) is None:
                        self.nodes.upsert(entry, attribute)
                        imported += 1
                if offset is None:
                    break
        self.nodes.set_metadata("legacy_qdrant_imported")
        self._index_ready = False
        return imported
