#!/usr/bin/env python
"""Report what LayerMem actually built, for the LoCoMo results write-up.

Usage::

    python scripts/layermem_stats.py <qdrant_dir> <collection_name>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layermem.storage.vector_store import VectorStore  # noqa: E402
from layermem.storage.state_chain import StateChain  # noqa: E402
from layermem.storage.state_store import StateStore  # noqa: E402


def main() -> int:
    qdrant_dir, collection = sys.argv[1], sys.argv[2]
    store = VectorStore(qdrant_dir, collection, embedding_dims=384).connect()

    entries = store.scroll(limit=10000)
    by_layer = Counter(e.layer for e in entries)
    by_type = Counter(e.entry_type for e in entries)
    topics = {e.topic_id for e in entries if e.topic_id}

    state_path = sys.argv[3] if len(sys.argv) > 3 else StateChain.default_db_path(qdrant_dir, collection)
    nodes = StateStore(state_path)
    attributes = {
        f"{item['subject']}/{item['attribute']}": len(nodes.query(**item, query_type="all"))
        for item in nodes.attributes()
    }
    state_count = nodes.count()
    by_type["state"] = state_count

    report = {
        "total_entries": len(entries) + state_count,
        "by_layer": dict(sorted(by_layer.items())),
        "by_type": dict(sorted(by_type.items())),
        "topics": len(topics),
        "state": {
            "total": state_count,
            "current": len(attributes),
            "historical": state_count - len(attributes),
            "distinct_attributes": len(attributes),
            "attributes_with_history": sorted(
                key for key, count in attributes.items() if count > 1
            ),
        },
        "compression": {
            "L1_per_L2": round(by_layer.get("L1", 0) / by_layer["L2"], 2) if by_layer.get("L2") else None,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    nodes.close()
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
