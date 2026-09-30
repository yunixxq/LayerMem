from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import BaseMemoryLayer

logger = logging.getLogger(__name__)


class LayerMemConfig(BaseModel):
    user_id: str = Field(..., description="The user id of the memory system.")
    save_dir: str = Field(
        default="vector_store/layermem",
        description="Directory holding the Qdrant store for this user.",
    )

    retriever_name_or_path: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name or local path.",
    )
    embedding_model_dims: int = Field(default=384, description="Embedding dimension.")
    use_gpu: str = Field(default="cpu", description="Unused; kept for config parity.")

    llm_backend: str = Field(
        default="openai",
        description="'openai' for any OpenAI-compatible endpoint, 'transformers' for a local HF model.",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name or path.")
    llm_base_url: Optional[str] = Field(default=None, description="OpenAI-compatible base URL.")
    llm_api_key: Optional[str] = Field(default=None, description="API key (any string works for a local server).")
    llm_max_tokens: int = Field(default=1024, description="Max tokens per completion.")
    llm_temperature: float = Field(default=0.0, description="Sampling temperature.")

    # 缓冲区满足两个条件任意一个后即可触发记忆构建：1. 对话轮数超过阈值；2. token数超过阈值
    buffer_turn_threshold: int = Field(default=10)
    buffer_token_threshold: int = Field(default=4096)
    theta_cluster: float = Field(default=0.75)
    theta_dup: float = Field(default=0.92)
    theta_confirm: float = Field(default=0.90)
    theta_retrieval: float = Field(default=0.6)
    K1: int = Field(default=5)
    K2: int = Field(default=5)
    K3: int = Field(default=3)
    top_k: int = Field(default=10)
    collection_name: Optional[str] = Field(default=None)
    on_disk: bool = Field(default=True)


class LayerMemLayer(BaseMemoryLayer):
    layer_type: str = "LayerMem"

    def __init__(self, config: LayerMemConfig) -> None:
        from layermem import LayerMemory
        from layermem.configs.config import EmbedderConfig, LayerMemConfig as CoreConfig, LLMConfig

        self.config = config
        os.makedirs(config.save_dir, exist_ok=True)

        core = CoreConfig(
            embedder=EmbedderConfig(
                model_name="huggingface",
                model=config.retriever_name_or_path,
                embedding_dims=config.embedding_model_dims,
            ),
            llm=LLMConfig(
                model_name=config.llm_backend,
                model=config.llm_model,
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
            ),
            buffer_turn_threshold=config.buffer_turn_threshold,
            buffer_token_threshold=config.buffer_token_threshold,
            theta_cluster=config.theta_cluster,
            theta_dup=config.theta_dup,
            theta_confirm=config.theta_confirm,
            theta_retrieval=config.theta_retrieval,
            K1=config.K1,
            K2=config.K2,
            K3=config.K3,
            top_k=config.top_k,
            qdrant_path=os.path.join(config.save_dir, "qdrant"),
            collection_name=config.collection_name or config.user_id, # 用于构建qdrant的存储标识
            on_disk=config.on_disk,
        )
        self.memory = LayerMemory(core)
        logger.info(
            "LayerMemLayer ready for user=%s (store=%s, embedder=%s, llm=%s)",
            config.user_id,
            core.qdrant_path,
            config.retriever_name_or_path,
            config.llm_model,
        )

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        payload = dict(message)
        timestamp = kwargs.get("timestamp")
        if timestamp is not None:
            payload.setdefault("timestamp", timestamp)
        self.memory.add([payload])

    # 用于处理对话中的消息，每次处理均需要判断是否已经超过dialog_buffer的阈值，若超过则需要flush
    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        timestamp = kwargs.get("timestamp")
        for message in messages:
            self.add_message(message, timestamp=timestamp)

    def save_memory(self) -> None:
        counts = self.memory.flush() # 收尾清空，结束时必须flush一次
        if any(counts.values()):
            logger.info("flushed tail batch: %s", counts)
        logger.info("memory stats: %s", self.memory.stats())

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        return self.memory.store.count() > 0

    def retrieve(
        self, query: str, k: int = 10, **kwargs
    ) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        memories = self.memory.retrieve(query, k=k)
        outputs: List[Dict[str, Union[str, Dict[str, Any]]]] = []
        for text in memories:
            outputs.append(
                {
                    "content": text,
                    "metadata": {"source": "layermem"},
                    "used_content": text,
                }
            )
        return outputs

    def delete(self, memory_id: str) -> bool:
        try:
            self.memory.store.delete([memory_id])
            return True
        except Exception as exc:
            logger.warning("delete failed for %s: %s", memory_id, exc)
            return False

    def update(self, memory_id: str, **kwargs) -> bool:
        try:
            self.memory.store.update_payload(memory_id, dict(kwargs))
            return True
        except Exception as exc:
            logger.warning("update failed for %s: %s", memory_id, exc)
            return False

    def close(self) -> None:
        self.memory.close()

# msg处理后的完整内容
def locomo_message_preprocessor(message, session) -> Dict[str, Any]:
    metadata = dict(getattr(message, "metadata", {}) or {})
    return {
        "role": getattr(message, "role", "user"),
        "content": getattr(message, "content", ""),
        "timestamp": getattr(message, "timestamp", None),
        "speaker_name": metadata.get("name") or metadata.get("speaker_name"),
        "session_id": getattr(session, "id", None) or getattr(session, "session_id", None),
    }
