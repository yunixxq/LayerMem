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
    normalize_factual_relational,
    normalize_state,
    parse_json_object,
    parse_json_value,
)
from layermem.prompts.extraction import (
    FACTUAL_RELATIONAL_EXTRACTION_PROMPT,
    STATE_EXTRACTION_PROMPT,
)


class OpenAILLM:
    """Extract memories through the OpenAI Responses API."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        if self.config.api_type != "responses":
            raise ValueError(f"Unsupported LLM API type: {self.config.api_type}")
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
    ) -> tuple[str, Dict[str, int]]:
        prompt = "\n\n".join(message.get("content", "") for message in messages)
        response = self.client.responses.create(model=self.model, input=prompt)

        status = getattr(response, "status", None)
        if status not in (None, "completed") or getattr(response, "error", None):
            raise ValueError(f"Responses request did not complete (status={status})")
        
        text = response.output_text
        if not text or not text.strip():
            raise ValueError("Responses endpoint returned no output text")
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        usage_info = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        self.total_calls += 1
        self.total_tokens += usage_info["total_tokens"]
        return text, usage_info

    def extract_memories(
        self,
        conversation: Conversation,
    ) -> Dict[str, Any]:
        """Run independent factual/relational and State extraction passes."""
        rendered = conversation_text(conversation)
        factual_raw, factual_usage = self.generate_response(
            [{
                "role": "user",
                "content": FACTUAL_RELATIONAL_EXTRACTION_PROMPT + "\n\n" + rendered,
            }]
        )
        state_raw, state_usage = self.generate_response(
            [{
                "role": "user",
                "content": STATE_EXTRACTION_PROMPT + "\n\n" + rendered,
            }]
        )
        factual_result = normalize_factual_relational(parse_json_object(factual_raw))
        state_result = normalize_state(parse_json_value(state_raw))
        result = {
            "factual": factual_result["factual"],
            "relational": factual_result["relational"],
            "state": state_result,
        }
        usage = {
            key: factual_usage.get(key, 0) + state_usage.get(key, 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        result["meta"] = {
            "usage": usage,
            "calls": 2,
        }
        return result

    def get_stats(self) -> Dict[str, int]:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens}
