"""Local HuggingFace Transformers backend for LayerMem memory extraction."""

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


class TransformersLLM:
    """Extract structured memories with a locally loaded causal language model."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        if not self.config.model:
            raise ValueError("A local Transformers model path or model id is required.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Without an explicit device_map the previous version silently fell
        # back to CPU/float32, which makes a local model unusable on Apple
        # silicon. Prefer CUDA, then MPS, then CPU.
        if torch.cuda.is_available():
            default_device: Any = "auto"
            default_dtype = torch.float16
        elif _mps_available():
            default_device = {"": "mps"}
            default_dtype = torch.float16
        else:
            default_device = {"": "cpu"}
            default_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            device_map=self.config.device_map or default_device,
            # ``dtype`` replaced ``torch_dtype`` in transformers 4.56.
            dtype=default_dtype,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.total_calls = 0
        self.total_tokens = 0

    @classmethod
    def from_config(cls, config: LLMConfig) -> "TransformersLLM":
        return cls(config)

    def _render_prompt(self, messages: List[Message]) -> str:
        """Apply the chat template, disabling reasoning mode where supported.

        Qwen3 and friends wrap their answer in a `` thinking`` block unless told
        otherwise; on an extraction prompt that consumes the whole completion
        budget and returns no JSON at all. Templates that do not know the flag
        raise TypeError, so fall back to the plain call.
        """
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.config.enable_thinking is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, enable_thinking=self.config.enable_thinking, **kwargs
                )
            except TypeError:
                pass
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_response(self, messages: List[Message]) -> tuple[str, Dict[str, int]]:
        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_params: Dict[str, Any] = {
            "do_sample": self.config.do_sample,
            "max_new_tokens": self.config.max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            generation_params.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_params)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(generated.shape[0])
        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.total_calls += 1
        self.total_tokens += usage_info["total_tokens"]
        return text, usage_info

    def extract_memories(
        self,
        conversation: Conversation,
        system_prompt: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        anchor = current_date or latest_timestamp(conversation)
        messages = [
            {
                "role": "system",
                "content": system_prompt or render_extraction_prompt(anchor),
            },
            {"role": "user", "content": conversation_text(conversation)},
        ]
        raw_response, usage = self.generate_response(messages)
        result = normalize_extraction(parse_json_object(raw_response))
        result["meta"] = {
            "usage": usage,
            "raw_response": raw_response,
            "current_date": anchor,
        }
        return result

    def get_stats(self) -> Dict[str, int]:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens}
