"""Three-track retrieval and the merge (LayerMem.md §2.4).

The two kinds of track are fundamentally different and are treated as such:

* **factual / relational — "find what is relevant".** Hybrid recall over the
  whole collection with no layer filter, because the L1-L4 ladder already
  encodes the concrete-to-abstract axis that the query's own embedding lands
  on. Their answer is a *ranked list*.
* **state — "find what is correct".** No ANN at all: an exact lookup on the
  version chain, because at any point in time an attribute has exactly one
  valid value. Current, interval, and history results are appended after the
  semantic ranking instead of competing on similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import Entry, StateEntry
from layermem.retrieval.router import RetrievalPlan
from layermem.storage.state_chain import StateChain
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    memories: List[str] = field(default_factory=list)
    entries: List[Entry] = field(default_factory=list)
    state_entry: Optional[Entry] = None
    state_entries: List[StateEntry] = field(default_factory=list)
    tracks_used: List[str] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Any,
        state_chain: StateChain,
        config: Optional[LayerMemConfig] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.state_chain = state_chain
        self.config = config or LayerMemConfig()

    def retrieve(self, query: str, plan: RetrievalPlan, k: Optional[int] = None) -> RetrievalResult:
        k = k or self.config.top_k
        state_entries = self._state_track(plan) if plan.wants_state else []

        ranked: List[Tuple[Entry, float]] = []
        tracks_used: List[str] = []
        for track in ("factual", "relational"):
            if track in plan.tracks:
                hits = self._semantic_track(query, plan, track, k)
                if hits:
                    tracks_used.append(track)
                ranked.extend(hits)

        merged = self._fuse(ranked, k)

        entries = [e for e, _ in merged]
        texts = [_render(e) for e in entries]
        if state_entries:
            entries.extend(state_entries)
            texts.extend(_render_state(entry) for entry in state_entries)
            tracks_used.append("state")

        return RetrievalResult(
            memories=texts,
            entries=entries,
            state_entry=state_entries[-1] if state_entries else None,
            state_entries=state_entries,
            tracks_used=tracks_used,
        )

    # ------------------------------------------------------------- tracks

    def _semantic_track(
        self, query: str, plan: RetrievalPlan, entry_type: str, k: int
    ) -> List[Tuple[Entry, float]]:
        """Hybrid recall, then a dense-only score floor.

        The floor is applied to the dense branch alone: RRF scores are ranks,
        and BM25 scores are unbounded, so a single threshold across both would
        silently drop exact-keyword matches that embed poorly
        (LayerMem_imple.md §七.8).
        """
        vector = self.embedder.embed(_embed_text(query, plan.keywords))
        candidate_pool = max(k * self.config.candidate_multiplier, 20)

        # State versions are resolved by the state track, which queries the
        # subject/attribute chain directly instead of going through ANN.
        dense_hits = self.store.search(
            vector,
            limit=candidate_pool,
            entry_type=entry_type,
        )
        kept = {e.id for e, score in dense_hits if score >= self.config.theta_retrieval}

        sparse_query = " ".join(plan.keywords) or query
        hybrid = self.store.hybrid_search(
            vector,
            sparse_query,
            limit=candidate_pool,
            prefetch_limit=candidate_pool,
            entry_type=entry_type,
        )
        # Keep a hybrid hit when it is either lexically strong (it survived
        # fusion although dense ranked it low) or above the dense floor.
        results = [
            (entry, score)
            for entry, score in hybrid
            if entry.id in kept or _mentions_any(entry.memory, plan.keywords)
        ]
        return results[:candidate_pool]

    def _state_track(self, plan: RetrievalPlan) -> List[StateEntry]:
        if not plan.subject or not plan.state_attr:
            return []
        upper = plan.time_upper if plan.time_upper is not None else plan.target_ts
        return self.state_chain.query(
            plan.subject, plan.state_attr, plan.query_type, plan.time_lower, upper,
        )

    # -------------------------------------------------------------- fusion

    @staticmethod
    def _fuse(
        ranked: Sequence[Tuple[Entry, float]], k: int
    ) -> List[Tuple[Entry, float]]:
        """RRF over the already-ranked tracks.

        Each entry's contribution is its rank *within its own track*, so the
        two tracks' incomparable score scales never meet.
        """
        if not ranked:
            return []

        by_track: Dict[str, List[Entry]] = {}
        for entry, _score in ranked:
            by_track.setdefault(entry.entry_type, []).append(entry)

        rrf_k = 60
        scores: Dict[str, float] = {}
        seen: Dict[str, Entry] = {}
        for entries in by_track.values():
            for rank, entry in enumerate(entries, start=1):
                scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (rrf_k + rank)
                seen[entry.id] = entry

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(seen[entry_id], score) for entry_id, score in ordered[:k]]


def _embed_text(query: str, keywords: Sequence[str]) -> str:
    """Embed the question together with its keywords.

    The planner's keywords are usually the answer-bearing terms; folding them
    into the embedded text nudges the query vector towards them.
    """
    extra = " ".join(k for k in keywords if k.lower() not in query.lower())
    return f"{query} {extra}".strip() if extra else query


def _mentions_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(k.lower() in lowered for k in keywords if len(k) > 2)


def _render(entry: Entry) -> str:
    """Format a memory for the answer prompt."""
    stamp = (getattr(entry, "event_time", None) or "")[:10]
    prefix = f"{stamp}: " if stamp else ""
    if entry.layer in ("L2", "L3", "L4"):
        label = {"L2": "topic summary", "L3": "stage portrait", "L4": "profile"}[entry.layer]
        return f"{prefix}[{label}] {entry.memory}"
    return f"{prefix}{entry.memory}"


def _render_state(entry: StateEntry) -> str:
    """State results are emitted as a complete time + subject + attribute + value."""
    stamp = (entry.event_time or "")[:10] or "unknown time"
    return f"{stamp}, {entry.subject} {entry.attribute}: {entry.value}. {entry.memory}"
