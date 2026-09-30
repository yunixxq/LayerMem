from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import Entry, iso_to_ts, now_ts
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
        fallback_time: Optional[float] = None,
    ) -> Dict[str, List[Entry]]:
        if not batch:
            return {"factual": [], "relational": [], "state": []}

        mention_ts = fallback_time or self._batch_time(batch) or now_ts()
        session_id = self._session_id(batch)
        raw = self._extract_with_retry(batch)

        if raw is None:
            logger.warning("extraction failed twice; storing the batch verbatim")
            return self._verbatim_fallback(batch, mention_ts, session_id)

        result = {
            "factual": self._factual_atoms(raw.get("factual") or [], mention_ts, session_id),
            "relational": self._relational_atoms(raw.get("relational") or [], mention_ts, session_id),
            "state": self._state_atoms(raw.get("state") or [], mention_ts, session_id),
        }
        logger.debug(
            "extracted factual=%d relational=%d state=%d",
            len(result["factual"]),
            len(result["relational"]),
            len(result["state"]),
        )
        return result

    def _extract_with_retry(self, batch: List[Message]) -> Optional[Dict[str, Any]]:
        """One retry, because a small local model intermittently emits
        malformed JSON on an otherwise fine prompt."""
        for attempt in (1, 2):
            try:
                return self.llm.extract_memories(batch)
            except Exception as exc:
                logger.warning("extraction attempt %d failed: %s", attempt, exc)
        return None

    def _verbatim_fallback(
        self, batch: List[Message], mention_ts: float, session_id: Optional[str]
    ) -> Dict[str, List[Entry]]:
        text = " ".join(
            f"{m.get('speaker_name') or m.get('role', 'user')}: {m.get('content', '')}".strip()
            for m in batch
            if str(m.get("content", "")).strip()
        )
        entry = Entry(
            entry_type="factual",
            layer="L1",
            memory=text[:4000],
            float_mention_time=mention_ts,
            float_event_time=mention_ts,
            source_dialog_ids=self._provenance(session_id),
        )
        return {"factual": [entry], "relational": [], "state": []}

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _batch_time(batch: List[Message]) -> Optional[float]:
        stamps = [iso_to_ts(m.get("timestamp")) for m in batch]
        stamps = [s for s in stamps if s is not None]
        return max(stamps) if stamps else None

    @staticmethod
    def _session_id(batch: List[Message]) -> Optional[str]:
        for message in batch:
            session_id = message.get("session_id")
            if session_id:
                return str(session_id)
        return None

    def _times(self, item: Dict[str, Any], mention_ts: float) -> Dict[str, Any]:
        event = iso_to_ts(_clean(item.get("event_time"))) or mention_ts
        return {"float_mention_time": mention_ts, "float_event_time": event}

    def _provenance(self, session_id: Optional[str]) -> List[str]:
        return [session_id] if session_id else []

    def _factual_atoms(
        self, items: List[Any], mention_ts: float, session_id: Optional[str]
    ) -> List[Entry]:
        atoms = []
        for item in items:
            if not isinstance(item, dict):
                continue
            memory = _clean(item.get("memory"))
            if not memory:
                continue
            atoms.append(
                Entry(
                    entry_type="factual",
                    layer="L1",
                    memory=memory,
                    source_dialog_ids=self._provenance(session_id),
                    **self._times(item, mention_ts),
                )
            )
        return atoms

    def _relational_atoms(
        self, items: List[Any], mention_ts: float, session_id: Optional[str]
    ) -> List[Entry]:
        atoms = []
        for item in items:
            if not isinstance(item, dict):
                continue
            memory = _clean(item.get("memory"))
            if not memory:
                continue
            atoms.append(
                Entry(
                    entry_type="relational",
                    layer="L1",
                    memory=memory,   # 保持原始自然语言，不拼 entities
                    source_dialog_ids=self._provenance(session_id),
                    **self._times(item, mention_ts),
                )
            )
        return atoms

    def _state_atoms(
        self, items: List[Any], mention_ts: float, session_id: Optional[str]
    ) -> List[Entry]:
        atoms = []
        for item in items:
            if not isinstance(item, dict):
                continue
            subject = _clean(item.get("subject"))
            attribute = _clean(item.get("attribute"))
            value = _clean(item.get("value"))
            if not (subject and attribute and value):
                continue
            atoms.append(
                Entry(
                    entry_type="state",
                    layer="L1",
                    memory=f"{subject} — {attribute}: {value}",
                    subject=subject,
                    attribute=attribute.lower(),
                    value=value,
                    status="current",
                    source_dialog_ids=self._provenance(session_id),
                    **self._times(item, mention_ts),
                )
            )
        return atoms
