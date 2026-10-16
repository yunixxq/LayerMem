"""Local review probes: failures capture current defects, not a passing regression suite.

Run from LayerMem with .venv/bin/python -m pytest -q others/reviews/2026-10-13/test_logic_review.py.
Only in-memory or pytest temporary databases; no provider requests.
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "memory_toolkits"))

import pytest
from qdrant_client import QdrantClient
from layermem.api import LayerMemory
from layermem.compression.layer import LayerCompressor
from layermem.configs.config import EmbedderConfig, LayerMemConfig, LLMConfig
from layermem.core.schema import ClusteredEntry, StateEntry
from layermem.memory.llm.openai import OpenAILLM
from layermem.pipeline.clusterer import Clusterer
from layermem.pipeline.extractor import Extractor
from layermem.pipeline.normalizer import Normalizer
from layermem.retrieval.retriever import _render_state
from layermem.storage.state_chain import StateChain
from layermem.storage.vector_store import VectorStore


class Embedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]

    def get_stats(self):
        return {}


class LLM:
    def __init__(self):
        self.prompts = []

    def generate_response(self, messages):
        self.prompts.append(messages[0]["content"])
        return "topic summary", {}

    def extract_memories(self, batch):
        return {"factual": ["Alice likes hiking"], "relational": [], "state": []}


@pytest.fixture
def store():
    result = VectorStore(":memory:", "review", 3, client=QdrantClient(":memory:"))
    result.ensure_collections()
    yield result
    result.close()


@pytest.fixture
def memory():
    result = LayerMemory(
        LayerMemConfig(qdrant_path=":memory:", embedder=EmbedderConfig(embedding_dims=3),
                       K1=2, K2=2, K3=2), embedder=Embedder(), llm=LLM(),
    )
    yield result
    result.close()


@pytest.mark.parametrize("layer,capacity", [("L1", 10), ("L2", 7), ("L3", 5)])
def test_full_topics_must_not_hide_an_eligible_topic(store, layer, capacity):
    for _ in range(capacity):
        store.upsert(ClusteredEntry(layer=layer, memory="old hiking", topic_id="full"), [1., 0., 0.])
    store.upsert(ClusteredEntry(layer=layer, memory="recent hiking", topic_id="open"), [.98, math.sqrt(1 - .98**2), 0.])
    assignment = Clusterer(store, Embedder()).assign(ClusteredEntry(layer=layer, memory="new hiking"))
    assert assignment.topic_id == "open", "an available topic with cosine .98 is hidden by five full-topic hits"


@pytest.mark.parametrize("failure", ["extraction", "embedding"])
def test_failed_flush_keeps_retryable_batch(memory, monkeypatch, failure):
    memory.add([{"content": "Alice likes hiking", "timestamp": "2024-01-01"}])
    target = memory.llm if failure == "extraction" else memory.embedder
    method = "extract_memories" if failure == "extraction" else "embed"
    monkeypatch.setattr(target, method, Mock(side_effect=RuntimeError("service unavailable")))
    with pytest.raises(RuntimeError):
        memory.flush()
    assert memory.buffer.pending_turns == 1, "flush drained the batch before the failing operation"


def test_repeating_compression_does_not_duplicate_summary(store):
    for name in ("hiking", "climbing"):
        store.upsert(ClusteredEntry(memory=name, topic_id="topic"), [1., 0., 0.])
    compressor = LayerCompressor(store, Embedder(), LLM(), config=LayerMemConfig(K1=2))
    compressor.compress_topic("topic", "factual")
    compressor.compress_topic("topic", "factual")
    assert store.count(layer="L2") == 1, "same input topic produced two unrelated summary IDs"


def test_retry_after_failed_summary_persist_does_not_lose_promotion(memory, monkeypatch):
    original = memory.store.upsert
    failed = False

    def upsert(entry, vector):
        nonlocal failed
        if entry.layer == "L2" and not failed:
            failed = True
            raise RuntimeError("summary store temporarily unavailable")
        original(entry, vector)

    monkeypatch.setattr(memory.store, "upsert", upsert)
    memory._write_clustered(ClusteredEntry(memory="first hiking fact"))
    with pytest.raises(RuntimeError):
        memory._write_clustered(ClusteredEntry(memory="second hiking fact"))
    memory._write_clustered(ClusteredEntry(memory="third hiking fact"))
    memory._write_clustered(ClusteredEntry(memory="fourth hiking fact"))
    assert memory.store.count(layer="L2") == 2, "first full topic was abandoned after summary persistence failed"


def test_text_update_refreshes_vectors(memory, monkeypatch):
    monkeypatch.setattr(memory.embedder, "embed", lambda text: [1., 0., 0.] if "hiking" in text else [0., 1., 0.])
    entry = ClusteredEntry(memory="Alice likes hiking")
    memory._write_clustered(entry)
    assert memory.update(entry.id, {"memory": "Alice plays chess"})
    point = memory.store.client.retrieve(memory.store.collection_for("factual"), [entry.id], with_vectors=True)[0]
    assert point.vector["dense"] == [0., 1., 0.], "payload changes but both retrieval indexes retain the old text"


def test_injected_connected_store_is_preserved(store):
    entry = ClusteredEntry(memory="already stored")
    store.upsert(entry, [1., 0., 0.])
    old_client = store.client
    result = LayerMemory(LayerMemConfig(embedder=EmbedderConfig(embedding_dims=3)),
                         embedder=Embedder(), llm=LLM(), store=store)
    try:
        assert result.store.get_by_id(entry.id) is not None, "LayerMemory reconnect replaced the injected in-memory database"
    finally:
        result.close()
        old_client.close()


def test_state_render_keeps_mention_time_for_relative_memory():
    entry = StateEntry(subject="Alice", attribute="residence", value="Helsinki",
                       event_time="2023-01-01", mention_time="2024-01-07",
                       memory="Alice lived in Helsinki last year.")
    rendered = _render_state(entry)
    assert "2024-01-07" in rendered, rendered


def test_period_validity_differs_from_change_nodes(store):
    chain = StateChain(store, Embedder())
    try:
        for when, city in [("2022-01-01", "Beijing"), ("2024-01-01", "Shanghai")]:
            chain.handle(StateEntry(subject="Alice", attribute="residence", value=city,
                                   memory=f"Alice lives in {city}.", event_time=when, mention_time=when))
        assert [e.value for e in chain.query("Alice", "residence", "range", "2023-01-01", "2023-12-31")] == ["Beijing"]
    finally:
        chain.close()


def test_second_speaker_placeholder_does_not_map_to_first_person():
    normalizer = Normalizer()
    normalizer.observe_speakers([{"speaker_name": "Alice"}, {"speaker_name": "Bob"}])
    assert normalizer.subject("speaker_b") == "Bob"


def test_unknown_historical_state_time_is_not_promoted_to_present():
    llm = LLM()
    llm.extract_memories = lambda batch: {"state": [{
        "subject": "Alice", "attribute": "residence", "value": "Beijing",
        "memory": "Alice lived in Beijing at an unknown past time.",
        "event_time": None, "mention_time": "2024-01-01",
    }]}
    result = Extractor(llm).extract([{"content": "I used to live in Beijing", "timestamp": "2024-01-01"}])
    assert result["state"] == [], "missing historical event time is silently replaced by mention time"


def test_deleted_attribute_points_do_not_hide_valid_candidate(store):
    chain = StateChain(store, Embedder())
    try:
        valid = StateEntry(subject="Alice", attribute="residence", value="Beijing",
                           memory="Alice lives in Beijing.", event_time="2024-01-01")
        chain.nodes.upsert(valid, "residence")
        chain.index.ensure_collection()
        chain.index.upsert("Alice", "residence", [.95, math.sqrt(1 - .95**2), 0.])
        for index in range(5):
            chain.index.upsert("Alice", f"deleted property {index}", [1., 0., 0.])
        result = chain.handle(StateEntry(subject="Alice", attribute="home city", value="Shanghai",
                                        memory="Alice lives in Shanghai.", event_time="2025-01-01"))
        assert result.entry.attribute == "residence", "orphan points consume top-k before SQLite existence filtering"
    finally:
        chain.close()


@pytest.mark.parametrize("decision", ["INCORRECT", "No, the answer is not correct."])
def test_negative_judgments_are_false(monkeypatch, decision):
    import memory_evaluation as evaluation

    class Judge:
        def __init__(self, **kwargs):
            pass

        def set_prompt(self, name):
            pass

        def __call__(self, *args, **kwargs):
            return [{"processed_content": decision}]

    monkeypatch.setattr(evaluation, "LLMExactMatch", Judge)
    question = SimpleNamespace(question="Where?", answer_list=["Beijing"], metadata={})
    result = evaluation.evaluate_answers([{"qa_pair": question}], [{"processed_content": "Paris"}], "fake")
    assert not result[0]["is_correct"], decision


def test_llm_configuration_is_forwarded():
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.config = LLMConfig(model="gpt-4o-mini", temperature=0., max_tokens=123)
    llm.model = llm.config.model
    llm.total_calls = llm.total_tokens = 0
    create = Mock(return_value=SimpleNamespace(status="completed", error=None, output_text="ok", usage=None))
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    llm.generate_response([{"role": "user", "content": "test"}])
    assert create.call_args.kwargs.get("temperature") == 0.
    assert create.call_args.kwargs.get("max_output_tokens") == 123


def test_rerun_rebuilds_instead_of_appending(tmp_path, monkeypatch):
    import memory_construction as construction
    from memories.layers.layermem import LayerMemLayer, LayerMemConfig as AdapterConfig

    opened = []

    class ReviewLayer(LayerMemLayer):
        def __init__(self, config):
            self.config = config
            self.memory = LayerMemory(LayerMemConfig(
                qdrant_path=str(tmp_path / "qdrant"),
                embedder=EmbedderConfig(embedding_dims=3),
            ), embedder=Embedder(), llm=LLM())
            opened.append(self)

    class Session(list):
        def get_string_timestamp(self):
            return "2024-01-01"

    monkeypatch.setattr(construction, "CONFIG_MAPPING", {"LayerMem": AdapterConfig})
    monkeypatch.setattr(construction, "MEMORY_LAYERS_MAPPING", {"LayerMem": ReviewLayer})
    monkeypatch.setattr(construction.time, "sleep", lambda seconds: None)
    messages = Session([SimpleNamespace(role="user", content="Alice likes hiking")])
    counts = []
    for _ in range(2):
        try:
            construction.memory_construction("LayerMem", "review-user", [messages],
                                             config={"llm_model": "review"}, rerun=True,
                                             dataset_type="LoCoMo")
            counts.append(opened[-1].memory.store.count())
        finally:
            opened[-1].close()
    assert counts == [1, 1], f"two rebuilds on the same input left {counts} entries"


@pytest.mark.parametrize("entry_type", ["factual", "relational"])
def test_control_one_topic_cascades_without_state(store, entry_type):
    config = LayerMemConfig(K1=1, K2=1, K3=1)
    llm = LLM()
    compressor = LayerCompressor(store, Embedder(), llm, config=config)
    entry = ClusteredEntry(entry_type=entry_type, memory="Alice thanked Bob for hiking advice", topic_id="topic")
    store.upsert(entry, [1., 0., 0.])
    created = compressor.on_write("topic", entry_type)
    assert [e.layer for e in created] == ["L2", "L3", "L4"]
    assert all(e.entry_type == entry_type for e in created)
    assert store.count(entry_type=entry_type) == 4
    assert len(llm.prompts) == 3
    assert entry_type in llm.prompts[0]
    assert "Current attributes" not in llm.prompts[-1]
    assert not hasattr(compressor, "state_chain")


def test_control_state_reads_require_only_sqlite(store):
    chain = StateChain(store, Embedder())
    try:
        for date, value in [("2025-01-01", "Beijing"), ("2023-01-01", "Helsinki")]:
            chain.handle(StateEntry(subject="Alice", attribute="residence", value=value,
                                    memory=f"Alice lived in {value}.", event_time=date))
        store.close()
        assert chain.get_current("Alice", "residence").value == "Beijing"
        assert chain.get_at_time("Alice", "residence", "2024-01-01").value == "Helsinki"
    finally:
        chain.close()
