from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import ClusteredEntry, Entry, StateEntry, normalize_time
from layermem.pipeline.buffer import Message

logger = logging.getLogger(__name__)

# Values a model emits instead of an actual timestamp.
_NULLISH = {"", "null", "none", "unknown", "n/a", "na", "unspecified", "unclear"}


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _NULLISH:
        return None
    return text


class Extractor:
    def __init__(self, llm: Any, config: Optional[LayerMemConfig] = None) -> None:
        self.llm = llm
        self.config = config or LayerMemConfig()

    def extract(
        self,
        batch: List[Message],
    ) -> Dict[str, List[Entry]]:
        if not batch:
            return {"factual": [], "relational": [], "state": []}

        raw = self._extract_with_retry(batch)

        result = {
            "factual": self._factual_atoms(raw.get("factual") or []),
            "relational": self._relational_atoms(raw.get("relational") or []),
            "state": self._state_atoms(raw.get("state") or []),
        }
        logger.debug(
            "extracted factual=%d relational=%d state=%d",
            len(result["factual"]),
            len(result["relational"]),
            len(result["state"]),
        )
        return result

    def _extract_with_retry(self, batch: List[Message]) -> Dict[str, Any]:
        for attempt in (1, 2):
            try:
                return self.llm.extract_memories(batch)
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError("LLM memory extraction failed after two attempts") from exc
                logger.warning("extraction attempt %d failed: %s", attempt, exc)

    @staticmethod
    def _parse_time(value: Any) -> Optional[str]:
        try:
            return normalize_time(value)
        except (ValueError, OverflowError, OSError):
            return None

    def _time_metadata(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        mention = self._parse_time(item.get("mention_time"))
        if mention is None:
            return None
        event_value = _clean(item.get("event_time"))
        event = self._parse_time(event_value)
        if event_value is not None and event is None:
            raise ValueError("Invalid event_time; use ISO-8601 or null")
        metadata = {"event_time": event or mention}
        if mention is not None:
            metadata["mention_time"] = mention
        return metadata

    def _clustered_atoms(
        self,
        items: List[str],
        entry_type: Literal["factual", "relational"],
    ) -> List[ClusteredEntry]:
        atoms: List[ClusteredEntry] = []
        for item in items:
            if not isinstance(item, str):
                continue
            memory = _clean(item)
            if memory is None:
                continue
            atoms.append(
                ClusteredEntry(
                    entry_type=entry_type,
                    layer="L1",
                    memory=memory,
                )
            )
        return atoms

    def _factual_atoms(self, items: List[str]) -> List[ClusteredEntry]:
        return self._clustered_atoms(items, "factual")

    def _relational_atoms(self, items: List[str]) -> List[ClusteredEntry]:
        return self._clustered_atoms(items, "relational")

    def _state_atoms(self, items: List[Dict[str, str]]) -> List[StateEntry]:
        """Build state entries from the six-field state objects in the prompt."""
        atoms: List[StateEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            memory = _clean(item.get("memory"))
            subject = _clean(item.get("subject"))
            attribute = _clean(item.get("attribute"))
            value = _clean(item.get("value"))
            if not (memory and subject and attribute and value):
                continue
            time_metadata = self._time_metadata(item)
            if time_metadata is None:
                continue
            atoms.append(
                StateEntry(
                    entry_type="state",
                    memory=memory,
                    subject=subject,
                    attribute=attribute.lower(),
                    value=value,
                    **time_metadata,
                )
            )
        return atoms
