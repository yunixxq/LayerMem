"""OpenAI-compatible embedding backend, including remote providers."""

import dataclasses
from typing import List, Optional, Union

import httpx
from openai import OpenAI

from layermem.configs.config import EmbedderConfig


class TextEmbedderOpenAI:
    def __init__(self, config: Optional[EmbedderConfig] = None):
        # See TextEmbedderHuggingface: keep the caller's config untouched.
        self.config = (
            dataclasses.replace(config)
            if config is not None
            else EmbedderConfig(model_name="openai")
        )
        self.model = self.config.model or "text-embedding-3-small"
        self.total_calls = 0
        self.total_tokens = 0

        client_kwargs: dict = {
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
        }
        if not self.config.verify_ssl:
            client_kwargs["http_client"] = httpx.Client(verify=False)
        self.client = OpenAI(**client_kwargs)

    @classmethod
    def from_config(cls, config: EmbedderConfig) -> "TextEmbedderOpenAI":
        return cls(config)

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        def preprocess(t):
            return str(t).replace("\n", " ")

        api_params = {"model": self.model}
        if self.config.pass_dimensions and self.config.embedding_dims is not None:
            api_params["dimensions"] = self.config.embedding_dims

        if isinstance(text, list):
            if len(text) == 0:
                return []
            inputs = [preprocess(x) for x in text]
            resp = self.client.embeddings.create(input=inputs, **api_params)
            self.total_calls += 1
            self.total_tokens += resp.usage.total_tokens
            vectors = [item.embedding for item in resp.data]
            self._remember_dims(vectors)
            return vectors

        preprocessed = preprocess(text)
        resp = self.client.embeddings.create(input=[preprocessed], **api_params)
        self.total_calls += 1
        self.total_tokens += resp.usage.total_tokens
        vectors = [item.embedding for item in resp.data]
        self._remember_dims(vectors)
        return vectors[0]

    def _remember_dims(self, vectors: List[List[float]]) -> None:
        """Record the vector size so the vector store can size the collection."""
        if vectors and self.config.embedding_dims is None:
            self.config.embedding_dims = len(vectors[0])

    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }
