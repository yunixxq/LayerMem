"""Shared helpers for the LLM extraction backends."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

Message = Dict[str, Any]
Conversation = Union[str, List[Message]]

EXTRACTION_KEYS = ("factual", "relational", "state")

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
# Buffer entries use "timestamp"; datasets and LightMem-style loaders use the
# other two spellings.
_TIME_KEYS = ("timestamp", "time_stamp", "float_time_stamp")
_SPEAKER_KEYS = ("speaker_name", "speaker")


def _turn_timestamp(message: Message) -> str:
    """Return the turn's timestamp as an ISO-8601 string, or an empty string."""
    for key in _TIME_KEYS:
        value = message.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (int, float)):
            # Rendered in the host's local timezone, so "yesterday" resolves
            # against the user's wall clock. Benchmark fixtures should carry
            # ISO strings instead if they need to be reproducible across
            # machines.
            try:
                return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
            except (OverflowError, OSError, ValueError):
                continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _turn_speaker(message: Message) -> str:
    for key in _SPEAKER_KEYS:
        value = message.get(key)
        if value:
            return str(value)
    return str(message.get("role", "user"))


def conversation_text(conversation: Conversation) -> str:
    """Render a conversation for the extraction prompt.

    Every turn is prefixed with its timestamp so the model can resolve
    relative time expressions and emit absolute double timestamps. Without
    that prefix the extraction prompt has no time anchor at all.
    """
    if isinstance(conversation, str):
        return conversation

    lines: List[str] = []
    for message in conversation:
        content = message.get("content", "")
        if not content:
            continue
        timestamp = _turn_timestamp(message)
        prefix = f"[{timestamp}] " if timestamp else ""
        lines.append(f"{prefix}{_turn_speaker(message)}: {content}")
    return "\n".join(lines)


def latest_timestamp(conversation: Conversation) -> Optional[str]:
    """Return the newest turn timestamp, used as the "today" anchor."""
    if isinstance(conversation, str):
        return None
    timestamps = [ts for ts in (_turn_timestamp(m) for m in conversation) if ts]
    return timestamps[-1] if timestamps else None


def parse_json_object(response_text: str) -> Dict[str, Any]:
    """Parse a JSON object out of a raw model response.

    Tolerates markdown fences and surrounding prose, both of which show up
    with providers that do not honour ``response_format``.
    """
    text = (response_text or "").strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The extraction response does not contain a JSON object.")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The extraction response must be a JSON object.")
    return parsed


def normalize_extraction(result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Coerce a parsed response into the three top-level memory lists.

    Missing categories become empty lists and non-dict entries are dropped, so
    one malformed item does not discard the whole batch.
    """
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for key in EXTRACTION_KEYS:
        value = result.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ValueError(f"Extraction field '{key}' must be a list.")
        normalized[key] = [item for item in value if isinstance(item, dict)]
    return normalized
