"""Qdrant wrapper: one collection **per perspective**, dense + sparse vectors.

Collections
-----------
``<collection_name>_factual`` / ``_relational`` / ``_state``. Splitting by
perspective means a query for one kind of memory only ever touches that
kind's index: the state chain's exact ``(subject, attribute)`` lookups and a
factual ANN scan never share a scan, and each collection can be tuned,
re-indexed or dropped on its own. Within a collection the layer (L1-L4) stays
a payload filter, so recall can still cross abstraction levels.

Methods that name an ``entry_type`` address exactly one collection. Methods
that do not fan out across all three and merge — see :meth:`_collections`.

BM25 without a BM25 dependency
------------------------------
Each collection carries a sparse vector using Qdrant's ``Modifier.IDF``, so
the server computes IDF and the client only sends *term frequencies*. A plain
tokenizer plus a hashing trick is therefore enough: no vocabulary to persist,
no extra model to download. Dense+sparse fusion runs server-side via
``FusionQuery(RRF)``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient, models

from layermem.core.schema import ENTRY_TYPES, Entry

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

# One collection per perspective; ``entry_type`` is implied by the collection
# rather than filtered on.
COLLECTIONS = ENTRY_TYPES

# Payload fields that get an index inside each collection. Without one,
# Qdrant filters degrade to a full scan.
INDEXED_FIELDS = ("layer", "topic_id", "subject", "attribute", "status")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have i in is it its of on or that the
    to was were will with""".split()
)


def _hash_token(token: str, dim: int) -> int:
    """Map a token into the sparse index space.

    Collisions simply make two rare terms share an IDF; at 2**20 buckets that
    is a good trade for not having to persist and migrate a vocabulary.
    """
    from zlib import crc32  # stable across processes and runs, unlike hash()

    return crc32(token.encode("utf-8")) % dim


def sparse_encode(text: str, dim: int) -> models.SparseVector:
    """Term-frequency sparse vector (IDF is applied server-side)."""
    counts: Dict[int, float] = {}
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        index = _hash_token(token, dim)
        counts[index] = counts.get(index, 0.0) + 1.0
    if not counts:
        return models.SparseVector(indices=[], values=[])
    indices = sorted(counts)
    return models.SparseVector(indices=indices, values=[counts[i] for i in indices])


class VectorStore:
    """Persistence for every :class:`Entry`, partitioned by perspective."""

    def __init__(
        self,
        path: str,
        collection_name: str,
        embedding_dims: int,
        sparse_dim: int = 2 ** 20,
        on_disk: bool = True,
        client: Optional[QdrantClient] = None,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.sparse_dim = sparse_dim
        self.embedding_dims = embedding_dims
        self._on_disk = on_disk
        self._client = client

    # ---------------------------------------------------------------- setup

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("VectorStore.connect() must be called first")
        return self._client

    def collection_for(self, entry_type: str) -> str:
        """Physical collection holding one perspective."""
        if entry_type not in COLLECTIONS:
            raise ValueError(
                f"Unknown entry_type {entry_type!r}; expected one of {COLLECTIONS}"
            )
        return f"{self.collection_name}_{entry_type}"

    def _collections(self, entry_type: Optional[str] = None) -> List[str]:
        """Collections a query must touch.

        Naming an ``entry_type`` is the fast path — one collection. Leaving it
        out fans out across all three, which is occasionally needed (compaction
        over a layer, ``get_by_id`` without knowing the type) but costs three
        round trips.
        """
        if entry_type is not None:
            if isinstance(entry_type, (list, tuple, set)):
                return [self.collection_for(t) for t in entry_type]
            return [self.collection_for(entry_type)]
        return [self.collection_for(t) for t in COLLECTIONS]

    def connect(self, path: Optional[str] = None) -> "VectorStore":
        self._client = QdrantClient(path=path or self.path)
        self.ensure_collections()
        return self

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def ensure_collections(self) -> None:
        """Create any missing per-perspective collection."""
        for entry_type in COLLECTIONS:
            name = self.collection_for(entry_type)
            if self.client.collection_exists(name):
                continue
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=self.embedding_dims,
                        distance=models.Distance.COSINE,
                        on_disk=self._on_disk,
                    )
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            for field in INDEXED_FIELDS:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    # ---------------------------------------------------------------- write

    def upsert(self, entry: Entry, dense: Sequence[float]) -> None:
        payload = entry.payload()
        payload["id"] = entry.id
        self.client.upsert(
            collection_name=self.collection_for(entry.entry_type),
            points=[
                models.PointStruct(
                    id=entry.id,
                    vector={
                        DENSE: list(dense),
                        SPARSE: sparse_encode(entry.memory, self.sparse_dim),
                    },
                    payload=payload,
                )
            ],
        )

    def delete(self, ids: Iterable[str], entry_type: Optional[str] = None) -> None:
        """Delete by id. Fans out unless the caller knows the perspective."""
        ids = list(ids)
        if not ids:
            return
        for collection in self._collections(entry_type):
            self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=ids),
            )

    def update_payload(
        self, entry_id: str, patch: Dict[str, Any], entry_type: Optional[str] = None
    ) -> bool:
        """Patch a payload, in whichever perspective collection holds the id.

        The state chain always knows its perspective and passes
        ``entry_type="state"``; the clusterer does not and fans out.

        Fanning out has a wrinkle worth knowing: Qdrant's *embedded* client
        raises ``KeyError`` when asked to patch a point that is not in that
        collection, whereas a server-side deployment treats it as a no-op. So
        "not in this collection" is caught and treated as "keep looking".

        Returns ``True`` when some collection held the id. A ``False`` means
        the caller passed an id that is not stored anywhere — usually a stale
        reference — and is worth surfacing rather than swallowing.
        """
        for collection in self._collections(entry_type):
            try:
                self.client.set_payload(
                    collection_name=collection,
                    payload=patch,
                    points=[entry_id],
                )
            except Exception:  # noqa: BLE001 - missing point in this collection
                continue
            return True
        logger.warning("update_payload: no collection holds id %s", entry_id)
        return False

    # ---------------------------------------------------------------- read

    @staticmethod
    def _build_filter(
        layer: Optional[str | Sequence[str]] = None,
        topic_id: Optional[str] = None,
        subject: Optional[str] = None,
        attribute: Optional[str] = None,
        status: Optional[str] = None,
        exclude_status: Optional[str] = None,
    ) -> Optional[models.Filter]:
        def match(value):
            if isinstance(value, (list, tuple, set)):
                return models.MatchAny(any=list(value))
            return models.MatchValue(value=value)

        conditions: List[models.Condition] = []
        if layer is not None:
            conditions.append(models.FieldCondition(key="layer", match=match(layer)))
        if topic_id is not None:
            conditions.append(models.FieldCondition(key="topic_id", match=match(topic_id)))
        if subject is not None:
            conditions.append(models.FieldCondition(key="subject", match=match(subject)))
        if attribute is not None:
            conditions.append(models.FieldCondition(key="attribute", match=match(attribute)))
        if status is not None:
            conditions.append(models.FieldCondition(key="status", match=match(status)))
        if exclude_status is not None:
            conditions.append(
                models.FieldCondition(
                    key="status", match=models.MatchExcept(**{"except": [exclude_status]})
                )
            )
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _to_entry(point: Any) -> Entry:
        payload = dict(point.payload or {})
        # The point id is authoritative: without it Entry would fall back to
        # its default factory and hand back a fresh UUID, which would make
        # every later update_payload/delete target a row that does not exist.
        payload["id"] = str(point.id)
        return Entry.from_payload(payload)

    def search(
        self,
        dense: Sequence[float],
        limit: int,
        entry_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
        **filters: Any,
    ) -> List[Tuple[Entry, float]]:
        """Dense-only ANN search.

        Cosine scores are comparable across collections, so an unfiltered
        search simply merges all three by score.
        """
        query_filter = self._build_filter(**filters)
        hits: List[Tuple[Entry, float]] = []
        for collection in self._collections(entry_type):
            response = self.client.query_points(
                collection_name=collection,
                query=list(dense),
                using=DENSE,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            hits.extend((self._to_entry(p), float(p.score)) for p in response.points)
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:limit]

    def hybrid_search(
        self,
        dense: Sequence[float],
        query_text: str,
        limit: int,
        entry_type: Optional[str] = None,
        prefetch_limit: Optional[int] = None,
        **filters: Any,
    ) -> List[Tuple[Entry, float]]:
        """Dense + sparse recall fused server-side with RRF, per collection.

        Fusion scores are *rank*-based, so they are not comparable between
        collections; when this fans out, results are interleaved round-robin
        rather than merged by score. Pass ``entry_type`` whenever it is known.

        No absolute similarity threshold applies here — callers wanting a
        dense-only floor should use :meth:`search`.
        """
        prefetch_limit = prefetch_limit or max(limit * 4, 20)
        sparse_query = sparse_encode(query_text, self.sparse_dim)
        query_filter = self._build_filter(**filters)

        if not sparse_query.indices:
            return self.search(
                dense, limit, entry_type=entry_type, **filters
            )

        per_collection: List[List[Tuple[Entry, float]]] = []
        for collection in self._collections(entry_type):
            response = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=list(dense),
                        using=DENSE,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=SPARSE,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            per_collection.append(
                [(self._to_entry(p), float(p.score)) for p in response.points]
            )

        if len(per_collection) == 1:
            return per_collection[0]

        # Round-robin across collections: each one's rank-1 gets a slot before
        # any rank-2, which keeps every perspective represented without
        # pretending their fusion scores are on one scale.
        merged: List[Tuple[Entry, float]] = []
        for rank in range(max((len(c) for c in per_collection), default=0)):
            for hits in per_collection:
                if rank < len(hits):
                    merged.append(hits[rank])
                if len(merged) >= limit:
                    return merged
        return merged[:limit]

    def scroll(
        self,
        limit: int = 1000,
        entry_type: Optional[str] = None,
        **filters: Any,
    ) -> List[Entry]:
        """Exact filter, no ANN. Used by the state chain and the compressor.

        ``limit`` is **per collection**, not a global cap. When this fans out,
        every perspective gets its own budget — otherwise one perspective with
        many matching rows would crowd the others out entirely, which is the
        wrong default for the callers here (compaction and watermark scans
        want everything that matches).
        """
        query_filter = self._build_filter(**filters)
        entries: List[Entry] = []
        for collection in self._collections(entry_type):
            records, _ = self.client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            entries.extend(self._to_entry(r) for r in records)
        return entries

    def get_by_id(
        self, entry_id: str, entry_type: Optional[str] = None
    ) -> Optional[Entry]:
        for collection in self._collections(entry_type):
            records = self.client.retrieve(
                collection_name=collection,
                ids=[entry_id],
                with_payload=True,
            )
            if records:
                return self._to_entry(records[0])
        return None

    def count(self, entry_type: Optional[str] = None, **filters: Any) -> int:
        query_filter = self._build_filter(**filters)
        total = 0
        for collection in self._collections(entry_type):
            result = self.client.count(
                collection_name=collection,
                count_filter=query_filter,
                exact=True,
            )
            total += int(result.count)
        return total
