"""Shared helpers for the LLM extraction backends."""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

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
            # Unix timestamps are absolute instants; never use the host timezone.
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
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
    for index, message in enumerate(conversation, 1):
        content = message.get("content", "")
        if not content:
            continue
        timestamp = _turn_timestamp(message)
        prefix = f"[{timestamp}] " if timestamp else ""
        lines.append(f"[message {index}] {prefix}{_turn_speaker(message)}: {content}")
    return "\n".join(lines)


def parse_json_object(response_text: str) -> Dict[str, Any]:
    """Parse a JSON object out of a raw model response. """
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


def parse_json_value(response_text: str) -> Any:
    """Parse a JSON object or array from a provider response."""
    text = (response_text or "").strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("The extraction response does not contain a JSON value.")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("The extraction response contains incomplete JSON.")
    return json.loads(text[start : end + 1])


def normalize_extraction(result: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Coerce a parsed response into the three top-level memory lists.

    Missing categories become empty lists. Factual and relational strings are
    valid prompt outputs and must be preserved for the downstream Extractor;
    malformed scalar values are dropped, so one bad item does not discard the
    whole batch.
    """
    normalized: Dict[str, List[Any]] = {}
    for key in EXTRACTION_KEYS:
        value = result.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ValueError(f"Extraction field '{key}' must be a list.")
        if key in ("factual", "relational"):
            normalized[key] = [
                item for item in value
                if isinstance(item, (str, dict))
            ]
        else:
            normalized[key] = [item for item in value if isinstance(item, dict)]
    return normalized


def normalize_factual_relational(result: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Normalize the response from the factual/relational extraction pass."""
    normalized: Dict[str, List[Any]] = {}
    for key in ("factual", "relational"):
        value = result.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ValueError(f"Extraction field '{key}' must be a list.")
        normalized[key] = [
            item for item in value if isinstance(item, (str, dict))
        ]
    return normalized


def normalize_state(result: Any) -> List[Dict[str, Any]]:
    """Normalize the response from the independent State extraction pass."""
    if isinstance(result, dict):
        value = result.get("state", [])
    elif isinstance(result, list):
        value = result
    else:
        raise ValueError("State extraction response must be an object or list.")
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError("Extraction field 'state' must be a list.")
    return [item for item in value if isinstance(item, dict)]
