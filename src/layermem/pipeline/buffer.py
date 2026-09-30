"""Dialog buffer: accumulate turns, flush a batch to the extractor.

LayerMem.md §2.3 — batching is what keeps the LLM cost sane. Extracting on
every turn would call the model once per message; flushing every
``turn_threshold`` turns (or ``token_threshold`` tokens) collapses that into a
single call for the whole batch.

The buffer's message contract is ``{"role", "content", "timestamp"}``; the
LLM backends render the timestamp into the extraction prompt, and without it
neither ``event_time`` nor ``mention_time`` can be produced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

Message = Dict[str, Any]

try:  # tiktoken is a declared dependency, but never let counting break a run
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - defensive
    _ENCODER = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:  # pragma: no cover - defensive
            pass
    return max(1, len(text) // 4)


class DialogBuffer:
    """A bounded staging area in front of the memory store."""

    def __init__(self, turn_threshold: int = 10, token_threshold: int = 4096) -> None:
        self.turn_threshold = turn_threshold
        self.token_threshold = token_threshold
        self._messages: List[Message] = []
        self._tokens = 0

    # ---------------------------------------------------------------- write

    def add(self, message: Message) -> None:
        content = message.get("content") or ""
        if not str(content).strip():
            return
        self._messages.append(dict(message))
        self._tokens += count_tokens(str(content))

    def add_messages(self, messages: List[Message]) -> None:
        for message in messages:
            self.add(message)

    # ---------------------------------------------------------------- state

    @property
    def pending_turns(self) -> int:
        return len(self._messages)

    @property
    def pending_tokens(self) -> int:
        return self._tokens

    def should_flush(self) -> bool:
        if not self._messages:
            return False
        return (
            len(self._messages) >= self.turn_threshold
            or self._tokens >= self.token_threshold
        )

    def drain(self) -> List[Message]:
        """Return the buffered batch and reset."""
        batch = self._messages
        self._messages = []
        self._tokens = 0
        return batch

    def peek(self) -> List[Message]:
        return list(self._messages)


def session_to_messages(session: Any) -> List[Message]:
    """Convert a toolkit ``Session`` into buffer messages.

    Kept here (rather than in the integration layer) because the timestamp
    field name is part of the buffer's contract.
    """
    messages: List[Message] = []
    for message in getattr(session, "messages", []):
        metadata: Dict[str, Any] = dict(getattr(message, "metadata", {}) or {})
        messages.append(
            {
                "role": getattr(message, "role", "user"),
                "content": getattr(message, "content", ""),
                "timestamp": getattr(message, "timestamp", None),
                "speaker_name": metadata.get("name"),
                "session_id": getattr(session, "id", None) or getattr(session, "session_id", None),
            }
        )
    return messages


def messages_within(messages: List[Message], since: Optional[float]) -> List[Message]:
    """Filter a batch by mention time — used when only part of a batch is new."""
    if since is None:
        return list(messages)
    from layermem.core.schema import iso_to_ts

    kept = []
    for message in messages:
        ts = iso_to_ts(message.get("timestamp"))
        if ts is None or ts > since:
            kept.append(message)
    return kept
