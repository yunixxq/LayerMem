"""Hierarchical, count-driven compression (LayerMem.md §2.2, temporal axis).

Compression is triggered by **how much has accumulated**, not by a calendar
boundary: a talkative user compresses often, a quiet one never pays for a
summary of two facts. That is the deliberate departure from TiMem's fixed
session/daily/weekly buckets.

The subtle part is the **watermark**. "count(L1 in topic) >= K1" is
monotonically true once a topic reaches K1, because L1 is never deleted — the
naive trigger would re-summarise the same atoms on every single write. So the
trigger counts only atoms *newer than the last compression*, and each summary
records how far it has consumed via ``compressed_until``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import Entry, now_ts
from layermem.prompts import compression as prompts
from layermem.storage.state_chain import StateChain
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class LayerCompressor:
    """Promotes atoms up the L1 -> L2 -> L3 -> L4 ladder."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Any,
        llm: Any,
        state_chain: Optional[StateChain] = None,
        config: Optional[LayerMemConfig] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.state_chain = state_chain
        self.config = config or LayerMemConfig()

    # ------------------------------------------------------------------ api

    def on_write(self, topic_id: Optional[str], entry_type: str) -> List[Entry]:
        """Run every trigger that the new write may have armed."""
        created: List[Entry] = []
        if topic_id:
            summary = self.compress_topic(topic_id, entry_type)
            if summary:
                created.append(summary)
        stage = self.compress_stage(entry_type)
        if stage:
            created.append(stage)
        profile = self.distill_profile()
        if profile:
            created.append(profile)
        return created

    # ------------------------------------------------------- level triggers

    def compress_topic(self, topic_id: str, entry_type: str) -> Optional[Entry]:
        """L1 -> L2 for one topic, if enough uncompressed atoms have piled up."""
        watermark = self._watermark(layer="L2", topic_id=topic_id)
        pending = self._pending(layer="L1", topic_id=topic_id, watermark=watermark)
        if len(pending) < self.config.K1:
            return None

        body = "\n".join(f"- [{_stamp(e)}] {e.memory}" for e in pending)
        summary = self._generate(
            prompts.L2_TOPIC_SUMMARY_PROMPT.replace("{entries}", body),
            on_failure=lambda: "; ".join(e.memory for e in pending),
        )
        entry = self._make_summary(
            summary,
            entry_type=entry_type,
            layer="L2",
            topic_id=topic_id,
            consumed=pending,
        )
        self._persist(entry)
        logger.debug("L1->L2 topic=%s consumed=%d", topic_id[:8], len(pending))
        return entry

    def compress_stage(self, entry_type: str) -> Optional[Entry]:
        """L2 -> L3 across topics: the stable characteristics of a period."""
        watermark = self._watermark(layer="L3", entry_type=entry_type)
        pending = self._pending(
            layer="L2", entry_type=entry_type, watermark=watermark
        )
        if len(pending) < self.config.K2:
            return None

        body = "\n".join(f"- {e.memory}" for e in pending)
        summary = self._generate(
            prompts.L3_STAGE_PORTRAIT_PROMPT.replace("{summaries}", body),
            on_failure=lambda: " ".join(e.memory for e in pending),
        )
        entry = self._make_summary(
            summary, entry_type=entry_type, layer="L3", topic_id=None, consumed=pending
        )
        self._persist(entry)
        logger.debug("L2->L3 type=%s consumed=%d", entry_type, len(pending))
        return entry

    def distill_profile(self) -> Optional[Entry]:
        """L3 -> L4: one durable profile, rebuilt from all stage portraits."""
        watermark = self._watermark(layer="L4")
        pending = self._pending(layer="L3", watermark=watermark)
        if len(pending) < self.config.K3:
            return None

        portraits = "\n".join(f"- {e.memory}" for e in pending)
        attributes = self._attribute_snapshot()
        summary = self._generate(
            prompts.L4_PROFILE_PROMPT.replace("{portraits}", portraits).replace(
                "{attributes}", attributes
            ),
            on_failure=lambda: " ".join(e.memory for e in pending),
        )
        entry = self._make_summary(
            summary, entry_type="factual", layer="L4", topic_id=None, consumed=pending
        )
        self._persist(entry)
        logger.debug("L3->L4 consumed=%d", len(pending))
        return entry

    # ------------------------------------------------------------ internals

    def _watermark(
        self,
        layer: str,
        topic_id: Optional[str] = None,
        entry_type: Optional[str] = None,
    ) -> Optional[float]:
        """How far this level has already consumed its input."""
        summaries = self.store.scroll(
            limit=1000, layer=layer, topic_id=topic_id, entry_type=entry_type
        )
        if not summaries:
            return None
        return max((s.compressed_until or s.event_key) for s in summaries)

    def _pending(
        self,
        layer: str,
        watermark: Optional[float],
        topic_id: Optional[str] = None,
        entry_type: Optional[str] = None,
    ) -> List[Entry]:
        entries = self.store.scroll(
            limit=2000, layer=layer, topic_id=topic_id, entry_type=entry_type
        )
        if watermark is not None:
            entries = [e for e in entries if e.event_key > watermark]
        return sorted(entries, key=lambda e: e.event_key)

    def _attribute_snapshot(self) -> str:
        if self.state_chain is None:
            return "(none)"
        current = self.state_chain.current_attributes()
        if not current:
            return "(none)"
        return "\n".join(
            f"- {e.subject} — {e.attribute}: {e.value}" for e in current
        )

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
        event_time = max(e.event_key for e in consumed)
        session_ids = sorted(
            {sid for e in consumed for sid in (e.source_dialog_ids or [])}
        )
        return Entry(
            entry_type=entry_type,
            layer=layer,
            memory=text,
            topic_id=topic_id,
            float_event_time=event_time,
            float_mention_time=now_ts(),
            compressed_until=event_time,
            source_dialog_ids=session_ids,
        )

    def _persist(self, entry: Entry) -> None:
        self.store.upsert(entry, self.embedder.embed(entry.memory))

    # ------------------------------------------------------- offline tidy-up

    def compact(self) -> Dict[str, int]:
        """Merge redundant summaries (LayerMem.md §2.2 note 1).

        L2 is merged within a topic, L4 across the profile; L3 dedup is left to
        a similarity pass because stage portraits legitimately overlap.
        """
        stats = {"l2": 0, "l4": 0}

        topics: Dict[str, List[Entry]] = {}
        for entry in self.store.scroll(limit=2000, layer="L2"):
            topics.setdefault(entry.topic_id or "", []).append(entry)
        for topic_id, summaries in topics.items():
            if topic_id and len(summaries) >= self.config.l2_compact_count:
                if self._merge(summaries, prompts.L2_MERGE_PROMPT, "L2", topic_id):
                    stats["l2"] += 1

        profiles = self.store.scroll(limit=100, layer="L4")
        if len(profiles) >= self.config.l4_compact_count:
            if self._merge(profiles, prompts.L4_MERGE_PROMPT, "L4", None, key="profiles"):
                stats["l4"] += 1

        return stats

    def _merge(
        self,
        entries: List[Entry],
        template: str,
        layer: str,
        topic_id: Optional[str],
        key: str = "summaries",
    ) -> bool:
        body = "\n".join(f"- {e.memory}" for e in entries)
        merged = self._generate(
            template.replace("{" + key + "}", body),
            on_failure=lambda: " ".join(e.memory for e in entries),
        )
        entry = self._make_summary(
            merged,
            entry_type=entries[0].entry_type,
            layer=layer,
            topic_id=topic_id,
            consumed=entries,
        )
        self._persist(entry)
        self.store.delete([e.id for e in entries], entry_type=entries[0].entry_type)
        return True


def _stamp(entry: Entry) -> str:
    from layermem.core.schema import ts_to_iso

    return (ts_to_iso(entry.float_event_time) or "")[:10] or "unknown"
