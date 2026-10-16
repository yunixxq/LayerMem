from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypeAlias, Union

ENTRY_TYPES = ("factual", "relational", "state")
LAYERS = ("L1", "L2", "L3", "L4")


def normalize_time(value: str | float | int | datetime | None) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def now_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def now_iso() -> str:
    return normalize_time(datetime.now(tz=timezone.utc))


def iso_to_ts(value: str | float | None) -> Optional[float]:
    try:
        text = normalize_time(value)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() if text else None
    except (ValueError, OverflowError, OSError):
        return None


def ts_to_iso(value: str | float | None) -> Optional[str]:
    return normalize_time(value)


@dataclass
class ClusteredEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = "factual"
    layer: str = "L1"
    memory: str = ""
    topic_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.entry_type not in ("factual", "relational"):
            raise ValueError(f"ClusteredEntry cannot have type {self.entry_type!r}")
        if self.layer not in LAYERS:
            raise ValueError(f"Unknown layer {self.layer!r}; expected one of {LAYERS}")

    def payload(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}


@dataclass
class StateEntry:
    """A version in a subject/attribute state chain."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = "state"
    memory: str = ""
    mention_time: str = field(default_factory=now_iso)
    event_time: Optional[str] = None
    subject: str = ""
    attribute: str = ""
    value: str = ""

    def __post_init__(self) -> None:
        if self.entry_type != "state":
            raise ValueError(f"StateEntry must have type 'state', got {self.entry_type!r}")
        self.mention_time = normalize_time(self.mention_time)
        self.event_time = normalize_time(self.event_time) or self.mention_time
        if self.mention_time is None:
            raise ValueError("mention_time is required")

    @property
    def event_key(self) -> str:
        return self.event_time or self.mention_time

    def payload(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}


Entry: TypeAlias = Union[ClusteredEntry, StateEntry]


def entry_from_payload(payload: Dict[str, Any]) -> Entry:
    payload = dict(payload)
    payload["id"] = str(payload.get("id"))
    if payload.get("entry_type") == "state":
        fields = {f.name for f in StateEntry.__dataclass_fields__.values()}
        return StateEntry(**{k: v for k, v in payload.items() if k in fields})
    fields = {f.name for f in ClusteredEntry.__dataclass_fields__.values()}
    return ClusteredEntry(**{k: v for k, v in payload.items() if k in fields})
