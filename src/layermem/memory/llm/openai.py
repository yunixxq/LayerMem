"""OpenAI-compatible LLM backend for LayerMem memory extraction."""

import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from layermem.configs.config import LLMConfig
from layermem.memory.llm.utils import (
    Conversation,
    Message,
    conversation_text,
    latest_timestamp,
    normalize_extraction,
    parse_json_object,
)
from layermem.prompts.extraction import render_extraction_prompt


class OpenAILLM:
    """Extract structured memories through an OpenAI-compatible chat API."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = self.config.model or "gpt-4o-mini"
        self.total_calls = 0
        self.total_tokens = 0

        client_kwargs: Dict[str, Any] = {
            "api_key": self.config.api_key or os.getenv("OPENAI_API_KEY"),
            "base_url": (
                self.config.base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            ),
        }
        if not self.config.verify_ssl:
            client_kwargs["http_client"] = httpx.Client(verify=False)
        self.client = OpenAI(**client_kwargs)

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OpenAILLM":
        return cls(config)

    def generate_response(
        self,
        messages: List[Message],
        response_format: Optional[Dict[str, str]] = None,
    ) -> tuple[str, Dict[str, int]]:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if response_format is not None:
            params["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**params)
        except Exception:
            # Not every OpenAI-compatible server implements JSON mode. The
            # parser tolerates fenced output, so retry once without it; if the
            # retry fails its own error surfaces.
            if response_format is None:
                raise
            params.pop("response_format", None)
            response = self.client.chat.completions.create(**params)

        usage = getattr(response, "usage", None)
        usage_info = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        self.total_calls += 1
        self.total_tokens += usage_info["total_tokens"]
        return response.choices[0].message.content or "", usage_info

    def extract_memories(
        self,
        conversation: Conversation,
        system_prompt: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract factual, relational, and state memories from raw dialogue.

        ``current_date`` anchors relative time expressions; it defaults to the
        newest turn timestamp in the conversation.
        """
        anchor = current_date or latest_timestamp(conversation)
        messages = [
            {
                "role": "system",
                "content": system_prompt or render_extraction_prompt(anchor),
            },
            {"role": "user", "content": conversation_text(conversation)},
        ]
        raw_response, usage = self.generate_response(
            messages,
            response_format={"type": "json_object"},
        )
        # Keep usage and the raw response out of the three memory lists, so
        # callers can iterate the result without special-casing metadata.
        result = normalize_extraction(parse_json_object(raw_response))
        result["meta"] = {
            "usage": usage,
            "raw_response": raw_response,
            "current_date": anchor,
        }
        return result

    def get_stats(self) -> Dict[str, int]:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens}
