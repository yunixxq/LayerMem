"""Persistence adapters and version-chain storage."""

from .state_chain import ADD, HISTORICAL_INSERT, UPDATE, StateChain, StateTransition
from .vector_store import VectorStore, sparse_encode

__all__ = [
    "ADD",
    "HISTORICAL_INSERT",
    "UPDATE",
    "StateChain",
    "StateTransition",
    "VectorStore",
    "sparse_encode",
]
