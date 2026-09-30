"""Canonicalise ``(subject, attribute)`` before anything touches storage.

Why this matters more than it looks (LayerMem_imple.md §七.1): the state
chain is keyed on an *exact* ``(subject, attribute)`` match, and the retrieval
planner emits ``state_attr`` that has to land in the same key space. If
"residence" and "home_address" become separate chains, ``get_current`` does
not fail loudly — it returns nothing, and the question quietly falls back to
the other tracks.

This first version is deliberately rule-based: the extraction prompt already
asks for short lowercase English attribute names, so a synonym table plus
light string normalisation covers the common drift. Anything the table does
not know is registered as-is, which keeps the chain working even when the
name is new.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

# Attribute synonyms seen in practice, mapped to one canonical name.
_ATTRIBUTE_ALIASES: Dict[str, str] = {
    "address": "residence",
    "home": "residence",
    "home_address": "residence",
    "city": "residence",
    "location": "residence",
    "living_place": "residence",
    "live_in": "residence",
    "move_to": "residence",
    "housing": "residence",
    "job": "occupation",
    "work": "occupation",
    "profession": "occupation",
    "career": "occupation",
    "employment": "occupation",
    "employer": "employer",
    "company": "employer",
    "workplace": "employer",
    "spouse": "marital_status",
    "marriage": "marital_status",
    "partner": "marital_status",
    "relationship_status": "marital_status",
    "pet": "pet_name",
    "pets": "pet_name",
    "dog": "pet_name",
    "cat": "pet_name",
    "age": "age",
    "birthday": "date_of_birth",
    "birthdate": "date_of_birth",
    "dob": "date_of_birth",
    "hobby": "hobbies",
    "hobbies_and_interests": "hobbies",
    "interests": "hobbies",
    "education": "education",
    "school": "education",
    "university": "education",
    "degree": "education",
}

# Pronouns and placeholders that should resolve to the speaker, not a subject.
_SELF_REFERENCES = {
    "i", "me", "my", "myself", "user", "the user", "the speaker",
    "speaker", "speaker_a", "speaker_b", "self",
}

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_attribute(attribute: Optional[str]) -> Optional[str]:
    """Fold an attribute name into its canonical form."""
    if not attribute:
        return None
    key = _NON_WORD.sub("_", str(attribute).strip().lower()).strip("_")
    if not key:
        return None
    return _ATTRIBUTE_ALIASES.get(key, key)


def normalize_subject(
    subject: Optional[str],
    speaker_names: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Resolve a subject to a concrete person when one can be identified.

    ``speaker_names`` are the people actually present in the current session;
    a self-reference maps to the first, which is the common case for
    single-user dialogue such as LoCoMo.
    """
    if not subject:
        return None
    text = str(subject).strip()
    if text.lower() in _SELF_REFERENCES:
        names = [n for n in (speaker_names or []) if n]
        if names:
            return names[0]
    return text


class Normalizer:
    """Stateful normaliser: remembers which names it has already seen."""

    def __init__(self) -> None:
        self._speaker_names: list[str] = []

    def observe_speakers(self, messages: Iterable[dict]) -> None:
        """Record the people present so self-references can resolve to them."""
        for message in messages:
            name = message.get("speaker_name")
            if name and name not in self._speaker_names:
                self._speaker_names.append(str(name))

    @property
    def speakers(self) -> list[str]:
        return list(self._speaker_names)

    def subject(self, subject: Optional[str]) -> Optional[str]:
        return normalize_subject(subject, self._speaker_names)

    def attribute(self, attribute: Optional[str]) -> Optional[str]:
        return normalize_attribute(attribute)
