"""Qdrant attribute candidates used only on the State write path."""

from __future__ import annotations

import json
import uuid
from typing import Sequence

from qdrant_client import models

from layermem.storage.vector_store import VectorStore


class StateAttributeIndex:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self.collection = f"{store.collection_name}_state_attribute"

    def ensure_collection(self) -> None:
        if self.store.client.collection_exists(self.collection):
            return
        self.store.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.store.embedding_dims, distance=models.Distance.COSINE,
            ),
        )
        self.store.client.create_payload_index(
            collection_name=self.collection, field_name="subject",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def point_id(subject: str, attribute: str) -> str:
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL, json.dumps([subject, attribute], ensure_ascii=False),
        ))

    def contains(self, subject: str, attribute: str) -> bool:
        return bool(self.store.client.retrieve(
            collection_name=self.collection, ids=[self.point_id(subject, attribute)],
            with_payload=False, with_vectors=False,
        ))

    def upsert(self, subject: str, attribute: str, vector: Sequence[float]) -> None:
        self.store.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(
                id=self.point_id(subject, attribute), vector=list(vector),
                payload={"subject": subject, "attribute": attribute},
            )],
            wait=True,
        )

    def candidates(
        self, subject: str, vector: Sequence[float], limit: int, threshold: float,
    ) -> list[tuple[str, float]]:
        response = self.store.client.query_points(
            collection_name=self.collection, query=list(vector), limit=limit,
            score_threshold=threshold,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="subject", match=models.MatchValue(value=subject),
            )]),
            with_payload=True,
        )
        return [(point.payload["attribute"], float(point.score)) for point in response.points]
