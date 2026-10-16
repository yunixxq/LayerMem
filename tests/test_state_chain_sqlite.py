import json
import math
import runpy
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from layermem.api import LayerMemory
from layermem.configs.config import EmbedderConfig, LayerMemConfig
from layermem.core.schema import ClusteredEntry, StateEntry
from layermem.retrieval.retriever import Retriever
from layermem.retrieval.router import RetrievalPlan, RetrievalRouter
from layermem.storage.state_chain import StateChain
from layermem.storage.vector_store import VectorStore


class Embedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return {
            "residence": [1., 0., 0.],
            "居住地": [1., 0., 0.],
            "home city": [.95, math.sqrt(1 - .95**2), 0.],
            "living place": [.8, .6, 0.],
            "work location": [.8, 0., .6],
            "occupation": [0., 1., 0.],
        }.get(text, [0., 0., 1.])

    def get_stats(self):
        return {"calls": len(self.calls)}


class LLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate_response(self, messages):
        self.prompts.append(messages[0]["content"])
        return json.dumps(self.response), {}


def state(value="Helsinki", event="2024-01-01", mention="2024-02-01", **kwargs):
    return StateEntry(
        subject=kwargs.pop("subject", "Alice"), attribute=kwargs.pop("attribute", "residence"),
        value=value, event_time=event, mention_time=mention,
        memory=kwargs.pop("memory", f"Alice lived in {value} last year."), **kwargs,
    )


@pytest.fixture
def chain(tmp_path):
    client = QdrantClient(":memory:")
    store = VectorStore(":memory:", "test", 3, client=client)
    store.ensure_collections()
    config = LayerMemConfig(state_db_path=str(tmp_path / "state.sqlite3"))
    result = StateChain(store, Embedder(), config)
    yield result
    result.close()
    store.close()


def test_out_of_order_versions_and_as_of_queries(chain):
    assert chain.handle(state("Helsinki", "2024-08-01")).action == "ADD"
    assert chain.handle(state("Shanghai", "2023-06-01", "2025-01-01")).action == "HISTORICAL_INSERT"
    assert chain.handle(state("Beijing", "2025-01-01")).action == "UPDATE"
    assert [e.value for e in chain.versions("Alice", "residence")] == ["Shanghai", "Helsinki", "Beijing"]
    assert chain.get_current("Alice", "residence").value == "Beijing"
    assert chain.get_at_time("Alice", "residence", "2024-05-01").value == "Shanghai"
    assert chain.get_at_time("Alice", "residence", "2024-08-01").value == "Helsinki"
    assert chain.get_at_time("Alice", "residence", "2020-01-01") is None


def test_restatement_keeps_both_audit_records_and_original_event_time(chain):
    first = state("Helsinki", "2024-01-01")
    chain.handle(first)
    chain.handle(state("Helsinki", "2024-05-01", "2024-05-02"))
    chain.handle(first)  # same node retry is idempotent
    assert chain.count() == 2
    assert chain.get_at_time("Alice", "residence", "2024-03-01").id == first.id
    assert chain.get_current("Alice", "residence").mention_time.startswith("2024-05-02")
    assert chain.embedder.calls == ["residence"]


def test_times_are_ordered_by_instant_then_mention_time(chain):
    chain.handle(state("early", "2024-01-01T09:00:00+08:00", "2024-01-02"))
    chain.handle(state("later", "2024-01-01T02:00:00Z", "2024-01-02"))
    chain.handle(state("correction", "2024-01-01T02:00:00Z", "2024-01-03"))
    chain.handle(state("stale correction", "2024-01-01T02:00:00Z", "2024-01-01"))
    assert chain.get_current("Alice", "residence").value == "correction"
    assert chain.get_at_time("Alice", "residence", "2024-01-01T01:30:00Z").value == "early"


def test_same_timestamps_have_deterministic_insertion_order(chain):
    first = state("A")
    chain.handle(first)
    chain.handle(state("B"))
    chain.handle(first)
    assert chain.get_current("Alice", "residence").value == "B"


def test_range_all_null_bounds_and_epoch(chain):
    for index, when in enumerate([0, "2023-01-01", "2024-01-01"]):
        chain.handle(state(str(index), when))
    assert chain.get_at_time("Alice", "residence", 0).value == "0"
    assert [e.value for e in chain.query("Alice", "residence", "range", "2023-01-01", "2024-01-01")] == ["1", "2"]
    assert len(chain.query("Alice", "residence", "all")) == 3
    assert len(chain.query("Alice", "residence", "range", time_upper=0)) == 1
    with pytest.raises(ValueError):
        chain.query("Alice", "residence", "all", "2025-01-01", "2020-01-01")
    with pytest.raises(ValueError):
        chain.query("Alice", "residence", "invalid")
    assert chain.query("Alice' OR 1=1 --", "residence", "all") == []


def test_high_similarity_merges_attributes_not_values(chain):
    chain.handle(state("Hangzhou"))
    result = chain.handle(state("Helsinki", "2025-01-01", attribute="Home City"))
    assert result.entry.attribute == "residence"
    assert chain.get_current("Alice", "home city").value == "Helsinki"
    assert chain.count() == 2
    assert chain.embedder.calls == ["residence", "home city"]
    points, _ = chain.store.client.scroll(chain.index.collection, limit=10)
    assert [p.payload for p in points] == [{"subject": "Alice", "attribute": "residence"}]


def test_low_similarity_creates_chain_and_subject_filter_prevents_cross_merge(chain):
    chain.handle(state())
    chain.handle(state("teacher", attribute="occupation"))
    chain.handle(state("Beijing", subject="Bob", attribute="home city"))
    assert chain.get_current("Bob", "residence") is None
    assert chain.get_current("Bob", "home city").value == "Beijing"
    assert len(chain.nodes.attributes()) == 3


@pytest.mark.parametrize("decision,expected", [("residence", "residence"), (None, "living place")])
def test_intermediate_similarity_asks_llm(chain, decision, expected):
    chain.handle(state())
    chain.llm = LLM({"attribute": decision})
    result = chain.handle(state("Beijing", "2025-01-01", attribute="living place"))
    assert result.entry.attribute == expected
    assert len(chain.llm.prompts) == 1
    assert '"candidates": ["residence"]' in chain.llm.prompts[0]


def test_invalid_llm_candidate_is_rejected_without_writing(chain):
    chain.handle(state())
    chain.llm = LLM({"attribute": "not a candidate"})
    with pytest.raises(ValueError, match="candidate"):
        chain.handle(state(attribute="living place"))
    assert chain.count() == 1


def test_chinese_attributes_and_full_memory_are_preserved(chain):
    original = state(subject="李彦学", attribute="居住地", memory="李彦学去年住在杭州。")
    chain.handle(original)
    loaded = chain.get_current("李彦学", "居住地")
    assert loaded.memory == original.memory
    assert loaded.attribute == "居住地"


def test_all_read_paths_work_after_qdrant_is_closed_and_sqlite_reopened(chain):
    chain.handle(state())
    chain.close()
    chain.store.close()
    reopened = StateChain(chain.store, None, chain.config)
    try:
        assert reopened.get_current("Alice", "residence").value == "Helsinki"
        assert len(reopened.versions("Alice", "residence")) == 1
        assert len(reopened.current_attributes()) == 1
        result = Retriever(chain.store, None, reopened).retrieve("history", RetrievalPlan(
            tracks=["state"], subject="Alice", state_attr="residence", query_type="all",
        ))
        assert len(result.state_entries) == 1
        assert "last year" in result.memories[0]
    finally:
        reopened.close()


def test_missing_index_is_rebuilt_from_sqlite_on_next_writer(chain):
    chain.handle(state())
    chain.store.client.delete_collection(chain.index.collection)
    chain.close()
    reopened = StateChain(chain.store, chain.embedder, chain.config)
    try:
        reopened.handle(state("Beijing", attribute="home city"))
        assert reopened.nodes.attributes() == [{"subject": "Alice", "attribute": "residence"}]
    finally:
        reopened.close()


def test_full_history_is_not_truncated_at_old_scroll_limit(chain):
    for index in range(1005):
        entry = state(str(index), index)
        chain.nodes.upsert(entry, "residence")
    assert len(chain.versions("Alice", "residence")) == 1005
    assert chain.get_current("Alice", "residence").value == "1004"


def test_current_attributes_limit_is_applied_after_grouping(chain):
    for index in range(510):
        chain.nodes.upsert(state(str(index), index), "residence")
    chain.nodes.upsert(state("teacher", attribute="occupation"), "occupation")
    assert len(chain.current_attributes(limit=2)) == 2


def test_state_update_delete_and_schema(chain):
    entry = state()
    chain.handle(entry)
    assert chain.update(entry.id, {"value": "Beijing", "memory": "Alice lives in Beijing."})
    assert chain.get_current("Alice", "residence").value == "Beijing"
    row = chain.nodes.connection.execute("SELECT id, typeof(event_time), content FROM state_chain").fetchone()
    assert isinstance(row[0], int)
    assert row[1] == "real"
    assert row[2] == "Alice lives in Beijing."
    assert chain.nodes.delete(entry.id)
    assert chain.get_current("Alice", "residence") is None


def test_legacy_migration_is_non_destructive_and_idempotent(chain):
    name = "test_state"
    entry = state()
    chain.store.client.create_collection(name, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    chain.store.client.upsert(name, points=[models.PointStruct(
        id=entry.id, vector=[1., 0., 0.], payload=entry.payload(),
    )])
    assert chain.migrate_legacy() == 1
    assert chain.migrate_legacy() == 0
    assert chain.get_current("Alice", "residence").id == entry.id
    assert chain.store.client.count(name).count == 1
    assert chain.embedder.calls == []


def test_legacy_migration_reads_every_page_and_legacy_float_times(chain):
    name = "test_state"
    chain.store.client.create_collection(name, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    points = []
    for index in range(270):
        entry = state(str(index), index)
        payload = entry.payload()
        payload.pop("event_time")
        payload.pop("mention_time")
        payload.update(float_event_time=index, float_mention_time=300)
        points.append(models.PointStruct(id=entry.id, vector=[1., 0., 0.], payload=payload))
    chain.store.client.upsert(name, points=points)
    assert chain.migrate_legacy() == 270
    assert chain.count() == 270
    assert chain.get_current("Alice", "residence").value == "269"


def test_range_retrieval_returns_multiple_states_without_vector_access(chain):
    for when in ("2022-01-01", "2023-01-01", "2023-06-01", "2024-01-01"):
        chain.handle(state(when, when))
    chain.store.close()
    result = Retriever(None, None, chain).retrieve("history", RetrievalPlan(
        tracks=["state"], subject="Alice", state_attr="residence", query_type="range",
        time_lower="2023-01-01", time_upper="2023-12-31",
    ))
    assert [e.value for e in result.state_entries] == ["2023-01-01", "2023-06-01"]
    assert result.tracks_used == ["state"]


def test_sqlite_failure_does_not_make_an_orphan_attribute_authoritative(chain, monkeypatch):
    original_upsert = chain.nodes.upsert

    def failing_upsert(*args):
        raise RuntimeError("SQLite unavailable")

    monkeypatch.setattr(chain.nodes, "upsert", failing_upsert)
    with pytest.raises(RuntimeError, match="SQLite unavailable"):
        chain.handle(state())
    monkeypatch.setattr(chain.nodes, "upsert", original_upsert)
    result = chain.handle(state(attribute="home city"))
    assert result.entry.attribute == "home city"
    assert chain.get_current("Alice", "residence") is None
    assert chain.count() == 1


def test_planner_outputs_sql_parameters_and_gets_canonical_attributes():
    llm = LLM({"tracks": ["state"], "subject": "Alice", "attribute": "residence",
               "query_type": "range", "time_lower": "2023-01-01", "time_upper": "2023-12-31"})
    plan = RetrievalRouter(llm).plan("Where did Alice live last year?", now=1704067200,
                                    state_attributes=[{"subject": "Alice", "attribute": "residence"}])
    assert plan.query_type == "range"
    assert plan.time_lower == "2023-01-01T00:00:00.000000Z"
    assert "2024-01-01" in llm.prompts[0]
    assert '"attribute": "residence"' in llm.prompts[0]


def test_invalid_planner_bounds_do_not_turn_a_historical_query_into_current():
    llm = LLM({"subject": "Alice", "attribute": "residence", "time_upper": "not-a-time"})
    plan = RetrievalRouter(llm).plan("past residence")
    assert plan.subject is None
    assert plan.state_attr is None


def test_layer_memory_writes_sqlite_and_stats_and_retrieval_include_states(tmp_path, monkeypatch, capsys):
    class ExtractionLLM(LLM):
        def extract_memories(self, batch):
            entry = state(memory="Alice lived in Helsinki last year.")
            return {"state": [entry.payload()], "factual": [], "relational": []}

    llm = ExtractionLLM({"tracks": ["state"], "subject": "Alice", "attribute": "residence",
                         "query_type": "all", "time_lower": None, "time_upper": None})
    config = LayerMemConfig(qdrant_path=str(tmp_path / "qdrant"),
                            embedder=EmbedderConfig(embedding_dims=3))
    memory = LayerMemory(config, embedder=Embedder(), llm=llm)
    try:
        memory.add([{"content": "I lived in Helsinki last year", "speaker_name": "Alice"}])
        assert memory.flush() == {"factual": 0, "relational": 0, "state": 1}
        assert memory.stats()["total"] == memory.stats()["state_entries"] == 1
        assert memory.store.count() == 0
        assert not memory.store.client.collection_exists("layermem_state")
        entry = memory.state_chain.get_current("Alice", "residence")
        assert entry.memory == "Alice lived in Helsinki last year."
        assert "last year" in memory.retrieve("all residences")[0]
        memory._write_clustered(ClusteredEntry(memory="Alice likes Python"))
        memory._write_clustered(ClusteredEntry(entry_type="relational", memory="Alice thanked Bob"))
        llm.response["tracks"] = ["factual", "relational", "state"]
        result = memory.retrieve_detailed("Alice")
        assert len(result.entries) == 3
        assert memory.stats()["by_type"] == {"factual": 1, "relational": 1, "state": 1}
        assert memory.update(entry.id, {"value": "Beijing"})
        assert memory.delete(entry.id)
        assert memory.stats()["total"] == 2
        memory._write_state(state())
    finally:
        memory.close()
    script = Path(__file__).resolve().parents[1] / "scripts" / "layermem_stats.py"
    monkeypatch.setattr(sys, "argv", [str(script), config.qdrant_path, config.collection_name])
    namespace = runpy.run_path(str(script))
    assert namespace["main"]() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total_entries"] == 3
    assert report["state"]["current"] == 1
