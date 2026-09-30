"""Retrieval planner: one LLM call decides *which stores to consult*.

Two departures from the usual "classify query difficulty" router:

* It splits the query across the three **perspectives**, not across layers.
  LayerMem_imple.md §三 argues that the L1-L4 abstraction axis already lives
  in the embedding space, so the ANN distance performs granularity alignment
  for free; a router that guesses at difficulty only adds a way to be wrong.
* It resolves ``time_ref`` into an absolute timestamp here, because the state
  chain needs a number, not the phrase "last year".

The planner is best-effort: if the model returns junk, the router falls back
to querying every track with the query itself as the only keyword.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from layermem.core.schema import iso_to_ts
from layermem.prompts.retrieval import render_planner_prompt

logger = logging.getLogger(__name__)

ALL_TRACKS = ("factual", "relational", "state")

# Placeholders a model emits instead of a real value, plus present-tense words
# that mean "no time constraint" — a state lookup with time_ref="now" must
# resolve to the current value, not to a parsed timestamp.
_NULLISH_VALUES = {
    "", "null", "none", "n/a", "na", "unknown", "unspecified",
    "now", "current", "currently", "present", "today", "the present",
}


@dataclass
class RetrievalPlan:
    tracks: List[str] = field(default_factory=lambda: list(ALL_TRACKS))
    keywords: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    state_attr: Optional[str] = None
    time_ref: Optional[str] = None
    target_ts: Optional[float] = None

    @property
    def wants_state(self) -> bool:
        return "state" in self.tracks


class RetrievalRouter:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def plan(self, query: str, now: Optional[float] = None) -> RetrievalPlan:
        now = now or datetime.now(tz=timezone.utc).timestamp()
        try:
            text, _ = self.llm.generate_response(
                [{"role": "user", "content": render_planner_prompt(query)}]
            )
            payload = _extract_json(text)
        except Exception as exc:  # pragma: no cover - provider dependent
            logger.warning("retrieval planner failed (%s); querying all tracks", exc)
            payload = {}

        plan = _to_plan(payload, query)
        plan.target_ts = resolve_time_reference(plan.time_ref, now)
        return plan


# --------------------------------------------------------------------- parse


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_plan(payload: dict, query: str) -> RetrievalPlan:
    tracks = payload.get("tracks")
    if isinstance(tracks, str):
        tracks = [tracks]
    if not isinstance(tracks, list):
        tracks = list(ALL_TRACKS)
    tracks = [t for t in tracks if t in ALL_TRACKS] or list(ALL_TRACKS)

    keywords = payload.get("keywords")
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    if not keywords:
        keywords = [query]

    def text_or_none(value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if not value or value.lower() in _NULLISH_VALUES:
            return None
        return value

    subject = text_or_none(payload.get("subject"))
    state_attr = text_or_none(payload.get("state_attr"))
    # A model that filled in state_attr clearly intends a state lookup even
    # when it forgot to list "state" in tracks; honour the stronger signal
    # rather than dropping the track.
    if state_attr and "state" not in tracks:
        tracks = [*tracks, "state"]

    return RetrievalPlan(
        tracks=tracks,
        keywords=keywords,
        subject=subject,
        state_attr=state_attr,
        time_ref=text_or_none(payload.get("time_ref")),
    )


# ------------------------------------------------------------ time resolution

_RELATIVE_PATTERNS = (
    (re.compile(r"\b(\d+)\s*(?:day|days)\s+ago\b", re.I), lambda m: timedelta(days=int(m.group(1)))),
    (re.compile(r"\b(\d+)\s*(?:week|weeks)\s+ago\b", re.I), lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"\b(\d+)\s*(?:month|months)\s+ago\b", re.I), lambda m: timedelta(days=30 * int(m.group(1)))),
    (re.compile(r"\b(\d+)\s*(?:year|years)\s+ago\b", re.I), lambda m: timedelta(days=365 * int(m.group(1)))),
)

_SIMPLE_OFFSETS = {
    "yesterday": timedelta(days=1),
    "last week": timedelta(weeks=1),
    "last month": timedelta(days=30),
    "last year": timedelta(days=365),
    "a week ago": timedelta(weeks=1),
    "a month ago": timedelta(days=30),
    "a year ago": timedelta(days=365),
}


def resolve_time_reference(time_ref: Optional[str], now: float) -> Optional[float]:
    """Turn a time expression into a timestamp.

    Returns ``None`` when nothing can be resolved, which the retriever treats
    as "no time constraint" — i.e. answer with the current value. That is the
    safe direction: a wrong timestamp silently answers a different question,
    while no timestamp answers the present-tense one.
    """
    if not time_ref:
        return None
    text = time_ref.strip().lower()

    absolute = iso_to_ts(text)
    if absolute is not None:
        return absolute

    for pattern in (r"%Y-%m-%d", r"%d %B %Y", r"%B %Y", r"%Y"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue

    if text in _SIMPLE_OFFSETS:
        return now - _SIMPLE_OFFSETS[text].total_seconds()

    for pattern, delta in _RELATIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            return now - delta(match).total_seconds()

    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        year = int(match.group(0))
        try:
            return datetime(year, 12, 31, tzinfo=timezone.utc).timestamp()
        except ValueError:  # pragma: no cover - defensive
            return None

    return None
