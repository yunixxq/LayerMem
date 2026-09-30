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

    model_name: str = "huggingface"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Compatibility aliases used by the original LightMem configuration.
    openai_base_url: Optional[str] = None
    huggingface_base_url: Optional[str] = None
    embedding_dims: Optional[int] = None
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
            self.base_url = self.openai_base_url or self.huggingface_base_url


@dataclass
class LLMConfig:
    """Configuration shared by LayerMem's LLM extraction backends."""

    model_name: str = "openai"
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
    # Qwen3-class models reason in a  thinking block by default, which eats the
    # whole completion budget on a structured-output task and leaves no JSON.
    # Ignored by chat templates that do not understand the flag.
    enable_thinking: Optional[bool] = False


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
    K1: int = 5                      # uncompressed L1 per topic -> L2
    K2: int = 5                      # L2 per entry_type -> L3
    K3: int = 3                      # L3 per entry_type -> L4

    # --- state chain ---
    theta_confirm: float = 0.90      # same value restated above this

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

    # --- offline compaction ---
    l2_compact_count: int = 3
    l3_compact_count: int = 5
    l3_sim_threshold: float = 0.85
    l4_compact_count: int = 3
