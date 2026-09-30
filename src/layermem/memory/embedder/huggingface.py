"""HuggingFace embedding backend.

Without ``base_url`` this loads a local SentenceTransformer. With
``base_url`` it calls a HuggingFace TEI service through its
OpenAI-compatible embeddings endpoint.
"""

import dataclasses
import logging
from typing import List, Optional, Union

import httpx
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from layermem.configs.config import EmbedderConfig

logger = logging.getLogger(__name__)


class TextEmbedderHuggingface:
    def __init__(self, config: Optional[EmbedderConfig] = None):
        # Resolve defaults on a private copy: the same config instance is
        # shared with the clusterer and the vector store, so filling in
        # ``model`` / ``embedding_dims`` here must not leak back to them.
        self.config = dataclasses.replace(config) if config is not None else EmbedderConfig()
        self.total_calls = 0
        self.total_tokens = 0

        if self.config.base_url:
            self.use_api = True
            self.model_name = self.config.model or "tei"
            client_kwargs: dict = {
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
            }
            if not self.config.verify_ssl:
                client_kwargs["http_client"] = httpx.Client(verify=False)
            self.client = OpenAI(**client_kwargs)
        else:
            self.use_api = False
            self.model_name = self.config.model or "all-MiniLM-L6-v2"
            self.model = self._load_local(self.model_name)
            self.config.model = self.model_name
            self.config.embedding_dims = (
                self.config.embedding_dims
                or self.model.get_sentence_embedding_dimension()
            )

    def _load_local(self, model_name: str) -> SentenceTransformer:
        """Load a sentence transformer, preferring whatever is already cached.

        A cached model must not require the Hub: sentence-transformers probes
        for optional side files such as ``adapter_config.json``, and on a
        restricted network that probe is a connection error that would
        otherwise take down a run whose weights are sitting right there in the
        cache. So the online attempt is a first try, not a requirement.
        """
        kwargs = self.config.model_kwargs or {}
        try:
            return SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            logger.warning(
                "online load of %r failed (%s); retrying from the local cache",
                model_name,
                exc,
            )
            return SentenceTransformer(model_name, local_files_only=True, **kwargs)

    @classmethod
    def from_config(cls, config: EmbedderConfig) -> "TextEmbedderHuggingface":
        return cls(config)

    def embed(self, text: Union[str, List[str]]):
        """Embed a single string or a batch of strings.

        Returns one vector for a string input and a list of vectors for a
        list input, mirroring ``TextEmbedderOpenAI.embed``.
        """
        self.total_calls += 1

        if self.use_api:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            self.total_tokens += getattr(response.usage, "total_tokens", 0)
            vectors = [item.embedding for item in response.data]
            self._remember_dims(vectors)
            return vectors[0] if isinstance(text, str) else vectors

        if isinstance(text, list) and not text:
            return []
        result = self.model.encode(text, convert_to_numpy=True)
        vectors = result.tolist() if isinstance(result, np.ndarray) else result
        self._remember_dims(vectors if isinstance(text, list) else [vectors])
        return vectors

    def _remember_dims(self, vectors: List[List[float]]) -> None:
        """Record the vector size reported by the backend.

        Creating the Qdrant collection needs the dimension, and TEI does not
        expose it up front — but every response carries it implicitly.
        """
        if vectors and self.config.embedding_dims is None:
            self.config.embedding_dims = len(vectors[0])

    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }
