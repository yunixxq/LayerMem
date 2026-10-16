"""Central configuration models for LayerMem.

The embedder and LLM configurations live here so that they can be composed
into the future top-level LayerMem configuration together with buffer,
retrieval, storage, and compression settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EmbedderConfig:
    """Configuration for the supported text embedding providers."""

    model_name: str = "openai"
    model: Optional[str] = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Compatibility alias used by the original LightMem configuration.
    openai_base_url: Optional[str] = None
    embedding_dims: Optional[int] = 1536
    # Only a few providers implement the OpenAI ``dimensions`` parameter.
    # vLLM, TEI and most other compatible endpoints reject it with a 400, so
    # it stays opt-in.
    pass_dimensions: bool = False
    # LightMem disables TLS verification globally to reach internal endpoints
    # with self-signed certificates. Keep the secure default and let callers
    # opt out explicitly instead.
    verify_ssl: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = self.openai_base_url


@dataclass
class LLMConfig:
    """Configuration shared by LayerMem's LLM extraction backends."""

    model_name: str = "openai"
    api_type: str = "responses"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 1.0
    do_sample: bool = False
    trust_remote_code: bool = True
    device_map: Any = None
    verify_ssl: bool = True


@dataclass
class LayerMemConfig:
    """Every tunable of the LayerMem pipeline (LayerMem_imple.md §六)."""

    # --- backends ---
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # --- buffer (LayerMem.md §2.3) ---
    buffer_turn_threshold: int = 10
    buffer_token_threshold: int = 4096

    # --- semantic axis ---
    theta_cluster: float = 0.75      # join an existing topic above this
    theta_dup: float = 0.92          # count as a repeat mention above this

    # --- temporal axis: compression triggers ---
    K1: int = 10                     # L1 atoms per topic before L2 compression
    K2: int = 7                      # L2 per topic -> L3
    K3: int = 5                      # L3 per topic -> L4

    # --- state chain ---
    theta_confirm: float = 0.90      # legacy config; State no longer drops confirmations
    state_attribute_high: float = 0.90
    state_attribute_low: float = 0.75
    state_attribute_top_k: int = 5
    state_db_path: Optional[str] = None  # default: per-collection SQLite under qdrant_path

    # --- retrieval ---
    theta_retrieval: float = 0.6     # dense-only score floor
    top_k: int = 10
    rrf_k: int = 60
    candidate_multiplier: int = 4    # prefetch size = top_k * this

    # --- storage ---
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "layermem"
    sparse_dim: int = 2 ** 20        # hashing space for the sparse index
    on_disk: bool = True
