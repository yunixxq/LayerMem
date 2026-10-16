"""SQLite is the authoritative store for State nodes (LayerMem.md §2.4)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from layermem.core.schema import StateEntry, iso_to_ts


def attribute_key(attribute: str) -> str:
    """Keep natural-language attributes, including non-Latin names."""
    return " ".join(attribute.strip().casefold().split())


def timestamp(value: str | float | None) -> Optional[float]:
    if value is None:
        return None
    parsed = iso_to_ts(value)
    if parsed is None:
        raise ValueError(f"Invalid state timestamp: {value!r}")
    return parsed


class StateStore:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS state_chain (
                id INTEGER PRIMARY KEY,
                entry_id TEXT NOT NULL UNIQUE,
                subject TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time REAL NOT NULL,
                mention_time REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chain
                ON state_chain (subject, attribute, event_time);
            CREATE TABLE IF NOT EXISTS state_alias (
                subject TEXT NOT NULL,
                alias TEXT NOT NULL,
                attribute TEXT NOT NULL,
                PRIMARY KEY (subject, alias)
            );
            CREATE TABLE IF NOT EXISTS state_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    def close(self) -> None:
        self.connection.close()

    def upsert(self, entry: StateEntry, alias: str) -> None:
        event, mention = timestamp(entry.event_time), timestamp(entry.mention_time)
        if event is None or mention is None:
            raise ValueError("State nodes require event_time and mention_time")
        with self.connection:
            # Integer id is the insertion-order tie breaker; entry_id preserves
            # the public UUID and makes retrying the same write idempotent.
            self.connection.execute("""
                INSERT INTO state_chain
                    (entry_id, subject, attribute, value, content, event_time, mention_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    subject=excluded.subject, attribute=excluded.attribute,
                    value=excluded.value, content=excluded.content,
                    event_time=excluded.event_time, mention_time=excluded.mention_time
            """, (entry.id, entry.subject, entry.attribute, entry.value,
                  entry.memory, event, mention))
            self.connection.executemany("""
                INSERT INTO state_alias (subject, alias, attribute) VALUES (?, ?, ?)
                ON CONFLICT(subject, alias) DO UPDATE SET attribute=excluded.attribute
            """, [(entry.subject, name, entry.attribute) for name in {alias, entry.attribute}])

    def resolve_alias(self, subject: str, attribute: str) -> str:
        key = attribute_key(attribute)
        row = self.connection.execute(
            "SELECT attribute FROM state_alias WHERE subject=? AND alias=?",
            (subject, key),
        ).fetchone()
        return row[0] if row else key

    def has_chain(self, subject: str, attribute: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM state_chain WHERE subject=? AND attribute=? LIMIT 1",
            (subject, attribute),
        ).fetchone() is not None

    def query(
        self, subject: str, attribute: str, query_type: str = "current",
        time_lower: str | float | None = None,
        time_upper: str | float | None = None,
    ) -> list[StateEntry]:
        if query_type not in ("current", "range", "all"):
            raise ValueError(f"Unknown state query_type: {query_type!r}")
        if not subject or not attribute:
            raise ValueError("State queries require subject and attribute")
        lower, upper = timestamp(time_lower), timestamp(time_upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("time_lower must not be later than time_upper")
        # Only the server-selected ordering is interpolated; model output is
        # always bound as data. LIMIT -1 is SQLite's unlimited form.
        order = "DESC" if query_type == "current" else "ASC"
        sql = f"""
            SELECT * FROM state_chain WHERE subject=? AND attribute=?
                AND event_time >= ? AND event_time <= ?
            ORDER BY event_time {order}, mention_time {order}, id {order}
            LIMIT ?
        """
        rows = self.connection.execute(sql, (
            subject, self.resolve_alias(subject, attribute),
            lower if lower is not None else -float("inf"),
            upper if upper is not None else float("inf"),
            1 if query_type == "current" else -1,
        )).fetchall()
        return [self._entry(row) for row in rows]

    def current_attributes(self, limit: int = 500) -> list[StateEntry]:
        rows = self.connection.execute("""
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY subject, attribute
                    ORDER BY event_time DESC, mention_time DESC, id DESC
                ) AS position FROM state_chain
            ) WHERE position=1 ORDER BY subject, attribute LIMIT ?
        """, (limit,)).fetchall()
        return [self._entry(row) for row in rows]

    def attributes(self) -> list[dict[str, str]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT DISTINCT subject, attribute FROM state_chain ORDER BY subject, attribute"
        )]

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM state_chain").fetchone()[0]

    def get_by_id(self, entry_id: str) -> Optional[StateEntry]:
        row = self.connection.execute(
            "SELECT * FROM state_chain WHERE entry_id=?", (entry_id,),
        ).fetchone()
        return self._entry(row) if row else None

    def delete(self, entry_id: str) -> bool:
        with self.connection:
            return self.connection.execute(
                "DELETE FROM state_chain WHERE entry_id=?", (entry_id,),
            ).rowcount > 0

    def has_metadata(self, key: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM state_metadata WHERE key=?", (key,),
        ).fetchone() is not None

    def set_metadata(self, key: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO state_metadata (key, value) VALUES (?, '1')", (key,),
            )

    @staticmethod
    def _entry(row: sqlite3.Row) -> StateEntry:
        return StateEntry(
            id=row["entry_id"], subject=row["subject"], attribute=row["attribute"],
            value=row["value"], memory=row["content"],
            event_time=row["event_time"], mention_time=row["mention_time"],
        )
