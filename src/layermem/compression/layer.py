"""Hierarchical, count-driven compression (LayerMem.md §2.2).

Compression is triggered by **how much has accumulated**, not by a calendar
boundary: a talkative user compresses often, a quiet one never pays for a
summary of two facts. That is the deliberate departure from TiMem's fixed
session/daily/weekly buckets.

An L1 topic is compressed exactly when it reaches K1 atoms. Full topics stop
accepting new L1 atoms, so the same topic cannot trigger the compression again.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import ClusteredEntry, Entry
from layermem.pipeline.clusterer import Clusterer
from layermem.prompts import compression as prompts
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class LayerCompressor:
    """Promotes atoms up the L1 -> L2 -> L3 -> L4 ladder."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Any,
        llm: Any,
        state_chain: Any = None,  # kept for source compatibility; never consulted
        config: Optional[LayerMemConfig] = None,
        clusterer: Optional[Clusterer] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.config = config or LayerMemConfig()
        self.clusterer = clusterer or Clusterer(self.store, self.embedder, self.config)

    # ------------------------------------------------------------------ api

    def on_write(self, topic_id: Optional[str], entry_type: str) -> List[Entry]:
        """Run every trigger that the new write may have armed."""
        created: List[Entry] = []
        if topic_id:
            summary = self.compress_topic(topic_id, entry_type)
            if summary:
                created.append(summary)
                stage = self.compress_stage(summary.topic_id, entry_type)
                if stage:
                    created.append(stage)
                    profile = self.distill_profile(stage.topic_id, entry_type)
                    if profile:
                        created.append(profile)
        return created

    # ------------------------------------------------------- level triggers

    def compress_topic(self, topic_id: str, entry_type: str) -> Optional[Entry]:
        """L1 -> L2 for one topic, if enough uncompressed atoms have piled up."""
        pending = self.store.scroll(
            limit=self.config.K1,
            layer="L1",
            topic_id=topic_id,
            entry_type=entry_type,
        )
        if len(pending) != self.config.K1:
            return None

        body = "\n".join(f"- {e.memory}" for e in pending)
        summary = self._generate(
            prompts.L2_PROMPTS[entry_type].format(entries=body),
            on_failure=lambda: "; ".join(e.memory for e in pending),
        )
        entry = self._make_summary(
            summary,
            entry_type=entry_type,
            layer="L2",
            topic_id=None,
            consumed=pending,
        )
        self._persist(entry)
        logger.debug("L1->L2 topic=%s consumed=%d", topic_id[:8], len(pending))
        return entry

    def compress_stage(self, topic_id: Optional[str], entry_type: str) -> Optional[Entry]:
        """L2 -> L3 for one topic after seven L2 summaries."""
        if not topic_id:
            return None
        pending = self.store.scroll(
            limit=self.config.K2,
            layer="L2",
            topic_id=topic_id,
            entry_type=entry_type,
        )
        if len(pending) != self.config.K2:
            return None

        body = "\n".join(f"- {e.memory}" for e in pending)
        summary = self._generate(
            prompts.L3_PROMPTS[entry_type].format(summaries=body),
            on_failure=lambda: " ".join(e.memory for e in pending),
        )
        entry = self._make_summary(
            summary, entry_type=entry_type, layer="L3", topic_id=None, consumed=pending
        )
        self._persist(entry)
        logger.debug("L2->L3 type=%s consumed=%d", entry_type, len(pending))
        return entry

    def distill_profile(self, topic_id: Optional[str], entry_type: str) -> Optional[Entry]:
        """L3 -> L4 for one topic after five L3 summaries."""
        if not topic_id:
            return None
        pending = self.store.scroll(
            limit=self.config.K3,
            layer="L3",
            topic_id=topic_id,
            entry_type=entry_type,
        )
        if len(pending) != self.config.K3:
            return None

        portraits = "\n".join(f"- {e.memory}" for e in pending)
        summary = self._generate(
            prompts.L4_PROMPTS[entry_type].format(portraits=portraits),
            on_failure=lambda: " ".join(e.memory for e in pending),
        )
        entry = self._make_summary(
            summary, entry_type=entry_type, layer="L4", topic_id=None, consumed=pending
        )
        self._persist(entry)
        logger.debug("L3->L4 consumed=%d", len(pending))
        return entry

    # ------------------------------------------------------------ internals

    def _generate(self, prompt: str, on_failure: Callable[[], str]) -> str:
        """One summarisation call; a provider outage degrades to concatenation.

        Losing a compression must never lose the underlying memory, so a failed
        call falls back to a mechanical join rather than dropping the atoms.
        """
        try:
            text, _ = self.llm.generate_response(
                [{"role": "user", "content": prompt}]
            )
        except Exception as exc:  # pragma: no cover - provider dependent
            logger.warning("compression call failed (%s); using fallback text", exc)
            return on_failure()
        text = (text or "").strip()
        return text or on_failure()

    def _make_summary(
        self,
        text: str,
        entry_type: str,
        layer: str,
        topic_id: Optional[str],
        consumed: List[Entry],
    ) -> Entry:
        return ClusteredEntry(
            entry_type=entry_type,
            layer=layer,
            memory=text,
            topic_id=topic_id,
        )

    def _persist(self, entry: Entry) -> None:
        vector = self.embedder.embed(entry.memory)
        assignment = self.clusterer.assign(entry, vector, layer=entry.layer)
        entry.topic_id = assignment.topic_id
        self.store.upsert(entry, vector)
