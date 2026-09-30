from typing import Any, Dict, Mapping

from layermem.configs.config import EmbedderConfig

from .huggingface import TextEmbedderHuggingface
from .openai import TextEmbedderOpenAI


class EmbedderFactory:
    _MODEL_MAPPING = {
        "huggingface": TextEmbedderHuggingface,
        "openai": TextEmbedderOpenAI,
    }

    @classmethod
    def from_config(cls, config: EmbedderConfig | Mapping[str, Any] | None = None) -> Any:
        """Instantiate the embedder named by ``config.model_name``.

        Args:
            config: an ``EmbedderConfig``, a plain mapping in the LightMem
                ``{"model_name": ..., "configs": {...}}`` shape, or ``None``
                for the defaults.

        Raises:
            ValueError: if the backend name is not supported.
        """
        if config is None:
            config = EmbedderConfig()
        elif isinstance(config, Mapping):
            values = dict(config)
            model_name = values.pop("model_name", "huggingface")
            nested = values.pop("configs", None)
            if nested is not None:
                values = {**nested, **values}
            config = EmbedderConfig(model_name=model_name, **values)

        model_name = config.model_name.lower()
        if model_name not in cls._MODEL_MAPPING:
            raise ValueError(
                f"Unsupported embedder model: {model_name}. "
                f"Supported models are: {list(cls._MODEL_MAPPING.keys())}"
            )

        return cls._MODEL_MAPPING[model_name](config)


# Backward-compatible name for callers that copied LightMem's API.
TextEmbedderFactory = EmbedderFactory
