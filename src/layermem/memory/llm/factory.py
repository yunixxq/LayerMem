from typing import Any, Dict, Mapping

from layermem.configs.config import LLMConfig

from .openai import OpenAILLM
from .transformers import TransformersLLM


class LLMFactory:
    _MODEL_MAPPING = {
        "openai": OpenAILLM,
        "transformers": TransformersLLM,
    }

    @classmethod
    def from_config(cls, config: LLMConfig | Mapping[str, Any] | None = None) -> Any:
        """Instantiate the backend named by ``config.model_name``.

        Args:
            config: an ``LLMConfig``, a plain mapping in the LightMem
                ``{"model_name": ..., "configs": {...}}`` shape, or ``None``
                for the defaults.

        Raises:
            ValueError: if the backend name is not supported.
        """
        if config is None:
            config = LLMConfig()
        elif isinstance(config, Mapping):
            values = dict(config)
            model_name = values.pop("model_name", "openai")
            nested = values.pop("configs", None)
            if nested is not None:
                values = {**nested, **values}
            config = LLMConfig(model_name=model_name, **values)

        model_name = config.model_name.lower()
        if model_name not in cls._MODEL_MAPPING:
            raise ValueError(
                f"Unsupported LLM backend: {model_name}. "
                f"Supported backends are: {list(cls._MODEL_MAPPING.keys())}"
            )

        return cls._MODEL_MAPPING[model_name](config)
