"""State version chain (LayerMem.md §2.2, the state track).

A subject's attribute holds exactly one *current* value; older values stay
reachable for historical questions. The chain is ordered by ``event_time``,
never by insertion order, because a later turn can describe an earlier
period ("I used to live in Shanghai last year").

Four transitions, from the design doc:

============ ==========================================================
ADD          no history for this key — write it as ``current``
CONFIRM      the same value restated at the same-or-later time
UPDATE       a different value at the same-or-later time — supersede the old
HISTORICAL   a different value at an *earlier* time — insert, do not touch
             what is currently believed
============ ==========================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from layermem.configs.config import LayerMemConfig
from layermem.core.schema import Entry
from layermem.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

ADD = "ADD"
CONFIRM = "CONFIRM"
UPDATE = "UPDATE"
HISTORICAL_INSERT = "HISTORICAL_INSERT"


@dataclass
class StateTransition:
    action: str
    entry: Entry
    superseded: Optional[Entry] = None


class StateChain:
    """Maintains one version chain per ``(subject, attribute)``."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Any,
        config: Optional[LayerMemConfig] = None,
    ) -> None:
        self.store = store
        # Kept for the offline paths that still embed superseded entries.
        self.embedder = embedder
        self.config = config or LayerMemConfig()

    # ------------------------------------------------------------------ api

    def handle(self, entry: Entry) -> StateTransition:
        """Apply one state atom to its chain and persist the outcome."""
        versions = self.versions(entry.subject, entry.attribute)
        current = self._current(versions)

        if current is None:
            transition = StateTransition(ADD, entry)
        elif entry.event_key >= current.event_key:
            transition = self._at_or_after_current(entry, current)
        else:
            transition = self._before_current(entry)

        self._persist(transition)
        return transition

    def get_current(self, subject: str, attribute: str) -> Optional[Entry]:
        """The value believed now — the only non-superseded version."""
        versions = [
            v for v in self.versions(subject, attribute) if v.status == "current"
        ]
        if versions:
            return max(versions, key=lambda e: e.event_key)
        # Defensive: a chain whose current flag was never set (e.g. all
        # entries arrived as historical inserts) still has a latest version.
        all_versions = self.versions(subject, attribute)
        return max(all_versions, key=lambda e: e.event_key) if all_versions else None

    def get_at_time(
        self, subject: str, attribute: str, target_ts: float
    ) -> Optional[Entry]:
        """The value in force at ``target_ts``: the newest version at or before it."""
        eligible = [
            v for v in self.versions(subject, attribute) if v.event_key <= target_ts
        ]
        return max(eligible, key=lambda e: e.event_key) if eligible else None

    def versions(self, subject: Optional[str], attribute: Optional[str]) -> List[Entry]:
        if not subject or not attribute:
            return []
        entries = self.store.scroll(
            limit=1000, entry_type="state", subject=subject, attribute=attribute
        )
        return sorted(entries, key=lambda e: e.event_key)

    def current_attributes(self, limit: int = 500) -> List[Entry]:
        """Every attribute currently believed, for the L4 profile prompt."""
        return sorted(
            self.store.scroll(limit=limit, entry_type="state", status="current"),
            key=lambda e: (e.subject or "", e.attribute or ""),
        )

    # ------------------------------------------------------------- internals

    @staticmethod
    def _current(versions: Sequence[Entry]) -> Optional[Entry]:
        flagged = [v for v in versions if v.status == "current"]
        if flagged:
            return max(flagged, key=lambda e: e.event_key)
        return max(versions, key=lambda e: e.event_key) if versions else None

    def _at_or_after_current(self, entry: Entry, current: Entry) -> StateTransition:
        if self._same_value(entry.value, current.value):
            # Restating a fact does not create a version, but it is evidence
            # the value still held at the later time — so the event time moves
            # forward, which is what keeps get_at_time() honest.
            current.float_event_time = max(current.event_key, entry.event_key)
            current.mention_count += 1
            current.float_mention_time = max(
                current.float_mention_time, entry.float_mention_time
            )
            return StateTransition(CONFIRM, current)
        return StateTransition(UPDATE, entry, superseded=current)

    def _before_current(self, entry: Entry) -> StateTransition:
        # An older value fills in history; it never displaces what is current.
        entry.status = "superseded"
        return StateTransition(HISTORICAL_INSERT, entry)

    def _same_value(self, left: Optional[str], right: Optional[str]) -> bool:
        """Is the new value a restatement of the current one?

        Deliberately *not* an embedding-similarity test, despite
        ``theta_confirm`` existing for that purpose. The two mistakes are not
        symmetric: a false UPDATE keeps the old value as history and makes the
        new value current — usually still the right answer, and always
        recoverable. A false CONFIRM discards a real change, so the system
        keeps answering with a value that is no longer true. City names, job
        titles and colours all embed close enough to cross a 0.9 threshold
        while being genuinely different, so the cheap mistake would be the
        common one.

        Restatements in practice are near-identical strings or one containing
        the other ("Helsinki" / "Helsinki, Finland"), which this catches.
        """
        if not left or not right:
            return False
        left_norm = " ".join(left.strip().lower().split())
        right_norm = " ".join(right.strip().lower().split())
        if left_norm == right_norm:
            return True
        shorter, longer = sorted((left_norm, right_norm), key=len)
        return len(shorter) >= 3 and shorter in longer

    def _persist(self, transition: StateTransition) -> None:
        action, entry = transition.action, transition.entry

        if action == CONFIRM:
            # Nothing new to insert — only the bookkeeping moved.
            self.store.update_payload(
                entry.id,
                {
                    "float_event_time": entry.float_event_time,
                    "float_mention_time": entry.float_mention_time,
                    "mention_count": entry.mention_count,
                },
                entry_type="state",
            )
            return

        if action == UPDATE and transition.superseded is not None:
            self.store.update_payload(
                transition.superseded.id, {"status": "superseded"}, entry_type="state"
            )

        vector = self.embedder.embed(entry.memory)
        entry.status = "current" if action in (ADD, UPDATE) else "superseded"
        self.store.upsert(entry, vector)
        logger.debug("state %s: %s/%s", action, entry.subject, entry.attribute)
