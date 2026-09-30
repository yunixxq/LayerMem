from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from layermem.compression.layer import LayerCompressor
from layermem.configs.config import LayerMemConfig
from layermem.core.schema import Entry
from layermem.memory.embedder import EmbedderFactory
from layermem.memory.llm import LLMFactory
from layermem.pipeline.buffer import DialogBuffer, Message
from layermem.pipeline.clusterer import Clusterer
from layermem.pipeline.extractor import Extractor
from layermem.pipeline.normalizer import Normalizer
from layermem.retrieval.retriever import RetrievalResult, Retriever
from layermem.retrieval.router import RetrievalRouter
from layermem.storage.state_chain import StateChain
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

class LayerMemory:
    def __init__(
        self,
        config: Optional[LayerMemConfig] = None,
        embedder: Any = None,
        llm: Any = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.config = config or LayerMemConfig()
        self.embedder = embedder or EmbedderFactory.from_config(self.config.embedder)
        self.llm = llm or LLMFactory.from_config(self.config.llm)

        self.store = store or VectorStore(
            path=self.config.qdrant_path,
            collection_name=self.config.collection_name,
            embedding_dims=self._embedding_dims(),
            sparse_dim=self.config.sparse_dim,
            on_disk=self.config.on_disk,
        )
        self.store.connect()

        # 构建时传递layermem.json中的配置参数
        self.buffer = DialogBuffer(
            turn_threshold=self.config.buffer_turn_threshold,
            token_threshold=self.config.buffer_token_threshold,
        )
        self.normalizer = Normalizer()
        self.extractor = Extractor(self.llm, self.config)
        self.clusterer = Clusterer(self.store, self.embedder, self.config)
        self.state_chain = StateChain(self.store, self.embedder, self.config)
        self.compressor = LayerCompressor(
            self.store, self.embedder, self.llm, self.state_chain, self.config
        )
        self.router = RetrievalRouter(self.llm)
        self.retriever = Retriever(self.store, self.embedder, self.state_chain, self.config)

    def add(self, messages: List[Message]) -> None:
        """Stage messages, flushing whenever the buffer crosses a threshold."""
        self.normalizer.observe_speakers(messages)
        for message in messages:
            self.buffer.add(message)
            if self.buffer.should_flush():
                self.flush()

    def flush(self) -> Dict[str, int]:
        """Extract and persist whatever is buffered."""
        batch = self.buffer.drain()
        if not batch:
            return {"factual": 0, "relational": 0, "state": 0}

        atoms = self.extractor.extract(batch)
        counts = {key: len(value) for key, value in atoms.items()}

        for entry in atoms["factual"] + atoms["relational"]:
            self._write_clustered(entry)
        for entry in atoms["state"]:
            self._write_state(entry)

        return counts

    def _write_clustered(self, entry: Entry) -> None:
        vector = self.clusterer.embed(entry.memory)
        assignment = self.clusterer.assign(entry, vector)

        if assignment.duplicate_of is not None:
            # A restatement, not a new memory: bump the count and move on.
            existing = assignment.duplicate_of
            self.store.update_payload(
                existing.id,
                {
                    "mention_count": existing.mention_count + 1,
                    "float_mention_time": max(existing.float_mention_time, entry.float_mention_time),
                },
                entry_type=entry.entry_type,
            )
            return

        entry.topic_id = assignment.topic_id
        self.store.upsert(entry, vector)
        self.compressor.on_write(assignment.topic_id, entry.entry_type)

    def _write_state(self, entry: Entry) -> None:
        entry.subject = self.normalizer.subject(entry.subject)
        entry.attribute = self.normalizer.attribute(entry.attribute)
        if not (entry.subject and entry.attribute and entry.value):
            return
        entry.memory = f"{entry.subject} — {entry.attribute}: {entry.value}"
        self.state_chain.handle(entry)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[str]:
        """Return the memory texts that should be handed to the answer model."""
        plan = self.router.plan(query)
        result = self.retriever.retrieve(query, plan, k=k)
        logger.debug("retrieved tracks=%s n=%d", result.tracks_used, len(result.memories))
        return result.memories

    def retrieve_detailed(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        plan = self.router.plan(query)
        return self.retriever.retrieve(query, plan, k=k)

    def compact(self) -> Dict[str, int]:
        """Merge redundant summaries. Safe to run between sessions."""
        return self.compressor.compact()

    def close(self) -> None:
        self.store.close()

    def stats(self) -> Dict[str, Any]:
        layers = {
            layer: self.store.count(layer=layer)
            for layer in ("L1", "L2", "L3", "L4")
        }
        return {
            "total": self.store.count(),
            "by_layer": layers,
            "by_type": {
                t: self.store.count(entry_type=t)
                for t in ("factual", "relational", "state")
            },
            "state_current": self.store.count(entry_type="state", status="current"),
            "state_superseded": self.store.count(entry_type="state", status="superseded"),
            "embedder": self.embedder.get_stats(),
            "llm": self.llm.get_stats() if hasattr(self.llm, "get_stats") else {},
        }

    def _embedding_dims(self) -> int:
        dims = getattr(self.config.embedder, "embedding_dims", None)
        if dims:
            return int(dims)
        # Learn it from the backend rather than making the caller declare it.
        probe = self.embedder.embed("dimension probe")
        dims = getattr(self.config.embedder, "embedding_dims", None) or len(probe)
        return int(dims)
