from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# The three heterogeneous perspectives of LayerMem.md §2.1.
ENTRY_TYPES = ("factual", "relational", "state")
# The hierarchical compression levels of LayerMem.md §2.2.
LAYERS = ("L1", "L2", "L3", "L4")


def now_ts() -> float:
    """Current wall-clock time as a unix timestamp."""
    return datetime.now(tz=timezone.utc).timestamp()


def iso_to_ts(value: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 string into a unix timestamp.

    Returns ``None`` for missing or unparsable input rather than raising:
    a memory with a broken timestamp must still be storable, it just cannot
    participate in ordering.
    """
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def ts_to_iso(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class Entry:
    """A single memory item."""

    # --- identity ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = "factual"          # factual | relational | state
    layer: str = "L1"                    # L1 | L2 | L3 | L4

    # --- content ---
    memory: str = ""

    # --- double timestamps (LayerMem.md §2.2 note 2) ---
    float_mention_time: float = field(default_factory=now_ts)
    float_event_time: Optional[float] = None

    # --- semantic-axis membership ---
    topic_id: Optional[str] = None

    # --- state-chain fields ---
    subject: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[str] = None
    status: Optional[str] = None         # current | superseded

    # --- bookkeeping ---
    mention_count: int = 1
    source_dialog_ids: List[str] = field(default_factory=list)
    # Watermark for the compression trigger (LayerMem_imple.md §七.2). Only
    # meaningful on topic nodes; absent elsewhere.
    compressed_until: Optional[float] = None
    updated_at: float = field(default_factory=now_ts)

    def __post_init__(self) -> None:
        if self.entry_type not in ENTRY_TYPES:
            raise ValueError(f"Unknown entry_type {self.entry_type!r}; expected one of {ENTRY_TYPES}")
        if self.layer not in LAYERS:
            raise ValueError(f"Unknown layer {self.layer!r}; expected one of {LAYERS}")

    @property
    def event_key(self) -> float:
        """Sort key for every temporal operation.

        ``float_event_time`` is optional, so ordering falls back to the
        mention time. Never compare the raw ``float_event_time``.
        """
        return self.float_event_time if self.float_event_time is not None else self.float_mention_time

    def payload(self) -> Dict[str, Any]:
        """Payload stored in Qdrant (the vector itself is kept separately)."""
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "Entry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
