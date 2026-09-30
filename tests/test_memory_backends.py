"""Tests for the embedder and LLM backends.

Run with::

    .venv/bin/pip install -e ".[dev]"
    .venv/bin/pytest tests/ -v                     # fast tests only
    .venv/bin/pytest tests/ -v -m integration      # adds model/download tests

The integration tests download a small model from HuggingFace. Behind a
restricted network, point ``HF_ENDPOINT`` at a mirror first::

    export HF_ENDPOINT=https://hf-mirror.com
"""

import sys
import types
from unittest.mock import patch

import pytest

from layermem.configs.config import EmbedderConfig, LLMConfig
from layermem.memory.embedder import EmbedderFactory, TextEmbedderHuggingface
from layermem.memory.llm import LLMFactory, OpenAILLM
from layermem.memory.llm.utils import (
    conversation_text,
    latest_timestamp,
    normalize_extraction,
    parse_json_object,
)
from layermem.prompts.extraction import (
    EXTRACTION_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    render_extraction_prompt,
)

# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def test_prompt_fills_the_date_anchor():
    rendered = render_extraction_prompt("2024-03-20")
    assert "2024-03-20" in rendered
    assert "{current_date}" not in rendered


def test_prompt_keeps_literal_json_braces():
    """str.replace (not str.format) must leave the JSON examples intact."""
    rendered = render_extraction_prompt("2024-03-20")
    assert '"factual": [' in rendered
    assert "{{" not in rendered


def test_prompt_asks_only_for_event_time():
    assert "event_time" in EXTRACTION_PROMPT
    assert "ISO-8601" in EXTRACTION_PROMPT
    assert "mention_time" not in EXTRACTION_PROMPT
    assert MEMORY_EXTRACTION_PROMPT is EXTRACTION_PROMPT


def test_prompt_without_a_date_still_renders():
    assert "{current_date}" not in render_extraction_prompt(None)


# --------------------------------------------------------------------------
# conversation rendering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("time_key", ["timestamp", "time_stamp", "float_time_stamp"])
def test_conversation_text_accepts_every_timestamp_spelling(time_key):
    conversation = [{"role": "user", "content": "hello", time_key: "2024-03-15T14:30:00"}]
    assert "[2024-03-15T14:30:00]" in conversation_text(conversation)


def test_conversation_text_renders_unix_timestamps_as_iso():
    conversation = [{"role": "user", "content": "hi", "timestamp": 1710500000.0}]
    rendered = conversation_text(conversation)
    assert rendered.startswith("[2024-03-15T")
    assert "hi" in rendered


def test_conversation_text_prefers_speaker_name_over_role():
    conversation = [{"role": "user", "speaker_name": "Alice", "content": "hi"}]
    assert conversation_text(conversation) == "Alice: hi"


def test_conversation_text_skips_empty_turns():
    conversation = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "two"},
    ]
    assert len(conversation_text(conversation).splitlines()) == 2


def test_conversation_text_passes_raw_strings_through():
    assert conversation_text("raw dialogue") == "raw dialogue"


def test_latest_timestamp_is_the_anchor():
    conversation = [
        {"role": "user", "content": "a", "timestamp": "2024-03-15T14:30:00"},
        {"role": "user", "content": "b", "timestamp": "2024-03-16T09:00:00"},
    ]
    assert latest_timestamp(conversation) == "2024-03-16T09:00:00"
    assert latest_timestamp([]) is None
    assert latest_timestamp("raw") is None


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------


def test_parse_json_object_unwraps_markdown_fences():
    assert parse_json_object('```json\n{"factual": []}\n```') == {"factual": []}


def test_parse_json_object_tolerates_surrounding_prose():
    raw = 'Sure, here you go:\n{"factual": []}\nLet me know if that helps.'
    assert parse_json_object(raw) == {"factual": []}


def test_parse_json_object_rejects_non_json():
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_normalize_extraction_fills_missing_categories():
    assert normalize_extraction({}) == {"factual": [], "relational": [], "state": []}


def test_normalize_extraction_drops_malformed_items():
    result = normalize_extraction(
        {"factual": ["oops", {"memory": "ok"}], "relational": None, "state": []}
    )
    assert result["factual"] == [{"memory": "ok"}]
    assert result["relational"] == []


def test_normalize_extraction_rejects_a_non_list_category():
    with pytest.raises(ValueError):
        normalize_extraction({"factual": "not a list"})


# --------------------------------------------------------------------------
# embedder
# --------------------------------------------------------------------------


def test_embedder_factory_returns_the_named_backend():
    embedder = EmbedderFactory.from_config(
        {"model_name": "openai", "model": "text-embedding-3-small", "api_key": "test"}
    )
    assert type(embedder).__name__ == "TextEmbedderOpenAI"


def test_embedder_factory_rejects_unknown_backends():
    with pytest.raises(ValueError, match="Unsupported embedder model"):
        EmbedderFactory.from_config({"model_name": "does-not-exist"})


def test_embedder_does_not_mutate_the_callers_config():
    """The config is shared with the clusterer and vector store."""
    caller_config = EmbedderConfig(model="all-MiniLM-L6-v2")
    TextEmbedderHuggingface(caller_config)
    assert caller_config.embedding_dims is None


def test_openai_embedder_omits_dimensions_by_default():
    """vLLM / TEI reject the ``dimensions`` parameter with a 400."""
    from layermem.memory.embedder.openai import TextEmbedderOpenAI

    embedder = TextEmbedderOpenAI(
        EmbedderConfig(model_name="openai", model="m", api_key="k", embedding_dims=8)
    )
    with patch.object(embedder, "client") as client:
        client.embeddings.create.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.0] * 8)],
            usage=types.SimpleNamespace(total_tokens=1),
        )
        embedder.embed("hello")
    assert "dimensions" not in client.embeddings.create.call_args.kwargs


def test_openai_embedder_sends_dimensions_when_opted_in():
    from layermem.memory.embedder.openai import TextEmbedderOpenAI

    embedder = TextEmbedderOpenAI(
        EmbedderConfig(
            model_name="openai",
            model="m",
            api_key="k",
            embedding_dims=8,
            pass_dimensions=True,
        )
    )
    with patch.object(embedder, "client") as client:
        client.embeddings.create.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.0] * 8)],
            usage=types.SimpleNamespace(total_tokens=1),
        )
        embedder.embed("hello")
    assert client.embeddings.create.call_args.kwargs["dimensions"] == 8


@pytest.mark.integration
def test_local_embedder_embeds_single_and_batch():
    """Downloads all-MiniLM-L6-v2 on first run (~90MB)."""
    embedder = EmbedderFactory.from_config(
        {"model_name": "huggingface", "model": "all-MiniLM-L6-v2"}
    )
    single = embedder.embed("hello world")
    batch = embedder.embed(["a", "b"])

    assert isinstance(single, list) and len(single) == 384
    assert len(batch) == 2 and len(batch[0]) == 384
    assert embedder.config.embedding_dims == 384
    assert embedder.get_stats() == {"total_calls": 2, "total_tokens": 0}


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


def _fake_completion(payload: str, total_tokens: int = 100):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=payload))],
        usage=types.SimpleNamespace(
            prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens
        ),
    )


VALID_PAYLOAD = (
    '{"factual": [{"memory": "Alice moved to Helsinki", '
    '"event_time": "2024-03-15T14:30:00", "mention_time": "2024-03-15T14:30:00"}],'
    ' "relational": [],'
    ' "state": [{"subject": "Alice", "attribute": "residence", "value": "Helsinki",'
    ' "event_time": "2024-03-15T14:30:00", "mention_time": "2024-03-15T14:30:00"}]}'
)

CONVERSATION = [
    {
        "role": "user",
        "content": "I moved to Helsinki",
        "timestamp": "2024-03-15T14:30:00",
        "speaker_name": "Alice",
    }
]


def test_llm_factory_selects_the_named_backend():
    config = {"model_name": "openai", "model": "gpt-4o-mini", "api_key": "test"}
    assert isinstance(LLMFactory.from_config(config), OpenAILLM)


def test_llm_factory_rejects_unknown_backends():
    with pytest.raises(ValueError, match="Unsupported LLM backend"):
        LLMFactory.from_config({"model_name": "does-not-exist"})


def test_extract_memories_returns_three_lists_plus_meta():
    llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="test"))
    with patch.object(llm, "client") as client:
        client.chat.completions.create.return_value = _fake_completion(VALID_PAYLOAD)
        result = llm.extract_memories(CONVERSATION)

    assert set(result) == {"factual", "relational", "state", "meta"}
    assert result["state"][0]["value"] == "Helsinki"
    assert result["meta"]["usage"]["total_tokens"] == 100
    assert result["meta"]["current_date"] == "2024-03-15T14:30:00"


def test_extract_memories_sends_timestamps_and_date_anchor_to_the_model():
    llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="test"))
    with patch.object(llm, "client") as client:
        client.chat.completions.create.return_value = _fake_completion(VALID_PAYLOAD)
        llm.extract_memories(CONVERSATION)

    sent = client.chat.completions.create.call_args.kwargs
    assert "2024-03-15T14:30:00" in sent["messages"][0]["content"]
    assert "[2024-03-15T14:30:00] Alice: I moved to Helsinki" in sent["messages"][1]["content"]


def test_extract_memories_retries_without_json_mode():
    """Some OpenAI-compatible servers reject response_format."""
    llm = OpenAILLM(LLMConfig(model="m", api_key="test"))
    calls = []

    def create(**params):
        calls.append(params)
        if "response_format" in params:
            raise RuntimeError("response_format is not supported")
        return _fake_completion(VALID_PAYLOAD)

    with patch.object(llm, "client") as client:
        client.chat.completions.create.side_effect = create
        result = llm.extract_memories(CONVERSATION)

    assert len(calls) == 2
    assert "response_format" not in calls[1]
    assert result["factual"]


def test_llm_tracks_token_stats():
    llm = OpenAILLM(LLMConfig(model="m", api_key="test"))
    assert llm.get_stats() == {"total_calls": 0, "total_tokens": 0}
    with patch.object(llm, "client") as client:
        client.chat.completions.create.return_value = _fake_completion(VALID_PAYLOAD, 42)
        llm.extract_memories(CONVERSATION)
    assert llm.get_stats() == {"total_calls": 1, "total_tokens": 42}


def test_llm_backend_imports_are_eager():
    """The package exposes the backend classes without lazy __getattr__ tricks."""
    import layermem.memory as memory

    assert memory.OpenAILLM is OpenAILLM
    assert memory.TextEmbedderHuggingface is TextEmbedderHuggingface
    assert not hasattr(sys.modules["layermem.memory.llm"], "_LAZY_BACKENDS")


def test_transformers_backend_uses_dtype_and_a_real_device():
    """transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``.

    Loading is stubbed out — this only checks the arguments we send and the
    device selection, not the model itself.
    """
    import torch

    from layermem.memory.llm import transformers as backend

    captured = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return object()

    with patch.object(backend.AutoTokenizer, "from_pretrained", return_value=object()), patch.object(
        backend.AutoModelForCausalLM, "from_pretrained", side_effect=fake_from_pretrained
    ):
        backend.TransformersLLM(LLMConfig(model="fake/model"))

    assert "torch_dtype" not in captured, "torch_dtype is deprecated in transformers>=4.56"
    assert "dtype" in captured

    if torch.cuda.is_available():
        assert captured["device_map"] == "auto"
    elif backend._mps_available():
        assert captured["device_map"] == {"": "mps"}
    else:
        assert captured["device_map"] == {"": "cpu"}


# --------------------------------------------------------------------------
# LayerMem pipeline regressions
# --------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    from layermem.storage.vector_store import VectorStore

    s = VectorStore(str(tmp_path / "qdrant"), "test", embedding_dims=4)
    s.connect()
    yield s
    s.close()


def test_entry_id_survives_a_round_trip_through_the_store(store):
    """Regression: a rebuilt Entry used to get a fresh UUID.

    Entry's id has a default_factory, and the payload copy used to drop the
    id field — so every read returned a different id and any later
    update_payload/delete targeted a row that did not exist.
    """
    from layermem.core.schema import Entry

    entry = Entry(entry_type="state", memory="m", subject="A", attribute="x", value="v")
    store.upsert(entry, [1.0, 0.0, 0.0, 0.0])

    read_back = store.scroll()[0]
    assert read_back.id == entry.id

    store.update_payload(read_back.id, {"status": "superseded"})
    assert store.scroll()[0].status == "superseded"

    store.delete([read_back.id])
    assert store.count() == 0


def test_excluding_status_also_drops_entries_without_the_field(store):
    """Documents why the semantic tracks must not filter on status.

    A Qdrant MatchExcept does not match documents missing the key, so
    ``exclude_status="superseded"`` would silently drop every factual and
    relational row.
    """
    from layermem.core.schema import Entry

    store.upsert(Entry(entry_type="factual", memory="no status here"), [1.0, 0.0, 0.0, 0.0])
    store.upsert(
        Entry(entry_type="state", memory="has status", status="current",
              subject="A", attribute="x", value="v"),
        [0.9, 0.1, 0.0, 0.0],
    )
    assert store.count() == 2
    assert store.count(exclude_status="superseded") == 1


def test_duplicate_mentions_bump_the_counter_instead_of_adding_rows(store):
    """A restatement at >= theta_dup is a mention, not a new memory."""
    from layermem.configs.config import LayerMemConfig
    from layermem.core.schema import Entry
    from layermem.pipeline.clusterer import Clusterer

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    clusterer = Clusterer(store, FixedEmbedder(), LayerMemConfig(theta_dup=0.9, theta_cluster=0.5))
    first = Entry(entry_type="factual", memory="Alice lives in Helsinki")
    store.upsert(first, [1.0, 0.0, 0.0, 0.0])
    first.topic_id = "t1"
    store.update_payload(first.id, {"topic_id": "t1"})

    second = Entry(entry_type="factual", memory="Alice lives in Helsinki")
    assignment = clusterer.assign(second, [1.0, 0.0, 0.0, 0.0])
    assert assignment.duplicate_of is not None
    assert assignment.topic_id == "t1"


def test_planner_promotes_state_when_it_names_an_attribute():
    """A model that fills state_attr but omits the track still meant state."""
    from layermem.retrieval.router import _to_plan

    plan = _to_plan({"tracks": ["factual"], "state_attr": "residence", "subject": "Alice"}, "q")
    assert "state" in plan.tracks
    assert plan.state_attr == "residence"


@pytest.mark.parametrize("present", ["now", "current", "present", "n/a", "null"])
def test_planner_treats_present_tense_as_no_time_constraint(present):
    from layermem.retrieval.router import _to_plan

    assert _to_plan({"time_ref": present}, "q").time_ref is None


def test_time_reference_resolution():
    from layermem.retrieval.router import resolve_time_reference

    now = 1710500000.0
    assert resolve_time_reference(None, now) is None
    assert resolve_time_reference("now", now) is None
    assert resolve_time_reference("yesterday", now) == pytest.approx(now - 86400)
    assert resolve_time_reference("2 years ago", now) == pytest.approx(now - 2 * 365 * 86400)
    assert resolve_time_reference("2022-05-01", now) is not None
    assert resolve_time_reference("2021", now) is not None


def test_extractor_falls_back_to_verbatim_text_when_the_model_breaks():
    """Losing a batch to a malformed JSON response is not acceptable."""
    from layermem.pipeline.extractor import Extractor

    class BrokenLLM:
        def extract_memories(self, conversation, **_):
            raise ValueError("no JSON object in response")

    batch = [
        {"role": "user", "content": "I moved to Helsinki",
         "timestamp": "2024-03-15T14:30:00", "speaker_name": "Alice"},
    ]
    atoms = Extractor(BrokenLLM()).extract(batch)
    assert len(atoms["factual"]) == 1
    assert "Helsinki" in atoms["factual"][0].memory


def test_mention_time_comes_from_the_turn_not_the_model():
    from layermem.core.schema import iso_to_ts
    from layermem.pipeline.extractor import Extractor

    class LLM:
        def extract_memories(self, conversation, **_):
            return {
                "factual": [
                    {"memory": "event in the past", "event_time": "2024-03-15T14:30:00"},
                    {"memory": "no date given"},
                ],
                "relational": [],
                "state": [
                    {"subject": "Alice", "attribute": "Residence", "value": "Helsinki",
                     "event_time": "unknown"},
                ],
            }

    batch = [
        {"role": "user", "content": "x", "timestamp": "2024-01-01T00:00:00"},
        {"role": "user", "content": "y", "timestamp": "2024-08-02T09:00:00"},
    ]
    atoms = Extractor(LLM()).extract(batch)
    session_ts = iso_to_ts("2024-08-02T09:00:00")

    for entry in atoms["factual"] + atoms["state"]:
        assert entry.float_mention_time == pytest.approx(session_ts)

    assert atoms["factual"][0].float_event_time == pytest.approx(
        iso_to_ts("2024-03-15T14:30:00")
    )
    assert atoms["factual"][1].float_event_time == pytest.approx(session_ts)
    assert atoms["state"][0].float_event_time == pytest.approx(session_ts)
    assert atoms["state"][0].attribute == "residence"


def test_normalizer_maps_common_attribute_synonyms():
    from layermem.pipeline.normalizer import Normalizer, normalize_attribute

    assert normalize_attribute("Home Address") == "residence"
    assert normalize_attribute("City") == "residence"
    assert normalize_attribute("Job") == "occupation"
    assert normalize_attribute("Pets") == "pet_name"
    assert normalize_attribute("favourite_colour") == "favourite_colour"

    normalizer = Normalizer()
    normalizer.observe_speakers([{"speaker_name": "Caroline"}])
    assert normalizer.subject("user") == "Caroline"
    assert normalizer.subject("Mel") == "Mel"


def test_state_chain_supersedes_and_answers_historically(store):
    """The core promise: one valid value now, older values still reachable."""
    from layermem.configs.config import LayerMemConfig
    from layermem.core.schema import Entry, iso_to_ts
    from layermem.storage.state_chain import StateChain

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    chain = StateChain(store, FixedEmbedder(), LayerMemConfig())

    def make(value, when):
        ts = iso_to_ts(when)
        return Entry(entry_type="state", memory=f"Alice — residence: {value}",
                     subject="Alice", attribute="residence", value=value,
                     float_event_time=ts, float_mention_time=ts)

    jan = chain.handle(make("Hangzhou", "2024-01-10T10:00:00"))
    aug = chain.handle(make("Helsinki", "2024-08-02T09:00:00"))

    assert jan.action == "ADD"
    assert aug.action == "UPDATE"
    assert chain.get_current("Alice", "residence").value == "Helsinki"
    assert chain.get_at_time("Alice", "residence", iso_to_ts("2024-03-01T00:00:00")).value == "Hangzhou"
    assert store.count(entry_type="state", status="superseded") == 1


def test_state_chain_keeps_a_later_mentioned_earlier_period_as_history(store):
    """'I used to live in Shanghai' must not overwrite what is current."""
    from layermem.configs.config import LayerMemConfig
    from layermem.core.schema import Entry, iso_to_ts
    from layermem.storage.state_chain import StateChain

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    chain = StateChain(store, FixedEmbedder(), LayerMemConfig())

    def make(value, event, mention):
        return Entry(entry_type="state", memory=f"Alice — residence: {value}",
                     subject="Alice", attribute="residence", value=value,
                     float_event_time=iso_to_ts(event), float_mention_time=iso_to_ts(mention))

    chain.handle(make("Helsinki", "2024-08-01T00:00:00", "2024-08-02T09:00:00"))
    # Mentioned later, but happened earlier.
    late = chain.handle(make("Shanghai", "2023-06-01T00:00:00", "2024-09-01T09:00:00"))

    assert late.action == "HISTORICAL_INSERT"
    assert chain.get_current("Alice", "residence").value == "Helsinki"
    assert chain.get_at_time("Alice", "residence", iso_to_ts("2023-12-01T00:00:00")).value == "Shanghai"


def test_compression_does_not_re_summarise_the_same_atoms(store):
    """The watermark regression: L1 is never deleted, so a naive count fires forever."""
    from layermem.configs.config import LayerMemConfig
    from layermem.compression.layer import LayerCompressor
    from layermem.core.schema import Entry, iso_to_ts

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    class CountingLLM:
        def __init__(self):
            self.calls = 0

        def generate_response(self, messages, **kwargs):
            self.calls += 1
            return "summary", {}

    store.llm = CountingLLM()
    compressor = LayerCompressor(store, FixedEmbedder(), store.llm, None, LayerMemConfig(K1=3, K2=99, K3=99))

    for index in range(3):
        store.upsert(
            Entry(entry_type="factual", layer="L1", topic_id="t1", memory=f"fact {index}",
                  float_event_time=iso_to_ts(f"2024-01-0{index + 1}T00:00:00"),
                  float_mention_time=iso_to_ts(f"2024-01-0{index + 1}T00:00:00")),
            [1.0, 0.0, 0.0, 0.0],
        )

    assert compressor.compress_topic("t1", "factual") is not None
    assert store.llm.calls == 1
    # Same atoms, second pass: the watermark must make this a no-op.
    assert compressor.compress_topic("t1", "factual") is None
    assert store.llm.calls == 1


def test_state_chain_prefers_update_over_confirm_for_similar_values(store):
    """A false CONFIRM is much worse than a false UPDATE.

    'Hangzhou' and 'Helsinki' are both cities and embed close together; an
    embedding-based comparison at theta_confirm=0.9 would call that a
    restatement and silently keep the stale value forever.
    """
    from layermem.configs.config import LayerMemConfig
    from layermem.core.schema import Entry, iso_to_ts
    from layermem.storage.state_chain import StateChain

    class EverythingLooksAlike:
        """Adversarial embedder: every value maps to the same vector."""

        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    chain = StateChain(store, EverythingLooksAlike(), LayerMemConfig(theta_confirm=0.9))

    def make(value, when):
        ts = iso_to_ts(when)
        return Entry(entry_type="state", memory=f"Alice — residence: {value}",
                     subject="Alice", attribute="residence", value=value,
                     float_event_time=ts, float_mention_time=ts)

    chain.handle(make("Hangzhou", "2024-01-10T10:00:00"))
    assert chain.handle(make("Helsinki", "2024-08-02T09:00:00")).action == "UPDATE"
    assert chain.get_current("Alice", "residence").value == "Helsinki"


def test_state_chain_confirms_a_restatement(store):
    from layermem.configs.config import LayerMemConfig
    from layermem.core.schema import Entry, iso_to_ts
    from layermem.storage.state_chain import StateChain

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    chain = StateChain(store, FixedEmbedder(), LayerMemConfig())

    def make(value, when):
        ts = iso_to_ts(when)
        return Entry(entry_type="state", memory=f"Alice — residence: {value}",
                     subject="Alice", attribute="residence", value=value,
                     float_event_time=ts, float_mention_time=ts)

    chain.handle(make("Helsinki", "2024-08-02T09:00:00"))
    transition = chain.handle(make("Helsinki, Finland", "2024-09-02T09:00:00"))

    assert transition.action == "CONFIRM"
    assert chain.get_current("Alice", "residence").value == "Helsinki"
    # Confirming is evidence the value still held later, so its event time moves.
    assert chain.get_current("Alice", "residence").float_event_time == pytest.approx(
        iso_to_ts("2024-09-02T09:00:00")
    )
    assert store.count(entry_type="state") == 1


# --------------------------------------------------------------------------
# one collection per perspective
# --------------------------------------------------------------------------


def test_each_perspective_gets_its_own_collection(store):
    from layermem.core.schema import Entry

    store.upsert(Entry(entry_type="factual", memory="f"), [1.0, 0.0, 0.0, 0.0])
    store.upsert(Entry(entry_type="relational", memory="r"), [0.0, 1.0, 0.0, 0.0])
    store.upsert(
        Entry(entry_type="state", memory="s", subject="A", attribute="x", value="v"),
        [0.0, 0.0, 1.0, 0.0],
    )

    for entry_type in ("factual", "relational", "state"):
        name = store.collection_for(entry_type)
        assert store.client.collection_exists(name)
        assert store.client.count(name, exact=True).count == 1

    assert store.count() == 3
    assert store.count(entry_type="factual") == 1
    assert store.count(entry_type="state") == 1


def test_unknown_perspective_is_rejected(store):
    with pytest.raises(ValueError, match="Unknown entry_type"):
        store.collection_for("opinions")


def test_search_routes_to_a_single_collection(store):
    """A factual query must not surface state rows, and vice versa."""
    from layermem.core.schema import Entry

    store.upsert(Entry(entry_type="factual", memory="Alice moved to Helsinki"), [1.0, 0.0, 0.0, 0.0])
    store.upsert(
        Entry(entry_type="state", memory="Alice residence Helsinki",
              subject="Alice", attribute="residence", value="Helsinki", status="current"),
        [1.0, 0.0, 0.0, 0.0],
    )

    factual = store.search([1.0, 0.0, 0.0, 0.0], limit=10, entry_type="factual")
    state = store.search([1.0, 0.0, 0.0, 0.0], limit=10, entry_type="state")
    assert [e.entry_type for e, _ in factual] == ["factual"]
    assert [e.entry_type for e, _ in state] == ["state"]

    # No perspective named: fan out and merge by score.
    everything = store.search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert {e.entry_type for e, _ in everything} == {"factual", "state"}


def test_hybrid_search_fans_out_round_robin(store):
    """Fusion scores are per-collection ranks, so interleave rather than merge."""
    from layermem.core.schema import Entry

    for i in range(3):
        store.upsert(Entry(entry_type="factual", memory=f"Helsinki fact {i}"), [1.0, 0.0, 0.0, 0.0])
        store.upsert(Entry(entry_type="relational", memory=f"Helsinki relation {i}"), [1.0, 0.0, 0.0, 0.0])

    hits = store.hybrid_search([1.0, 0.0, 0.0, 0.0], "Helsinki", limit=4)
    types = [e.entry_type for e, _ in hits]
    assert len(hits) == 4
    # Both perspectives must appear in the first two slots.
    assert set(types[:2]) == {"factual", "relational"}


def test_update_payload_finds_the_right_collection_without_a_hint(store):
    from layermem.core.schema import Entry

    entry = Entry(entry_type="relational", memory="Bob is Alice's friend")
    store.upsert(entry, [1.0, 0.0, 0.0, 0.0])

    assert store.update_payload(entry.id, {"topic_id": "t9"}) is True
    assert store.get_by_id(entry.id).topic_id == "t9"


def test_update_payload_reports_an_unknown_id(store):
    assert store.update_payload("no-such-id", {"status": "current"}) is False


def test_scroll_limit_is_per_collection(store):
    """One perspective must not be able to starve the others."""
    from layermem.core.schema import Entry

    for i in range(5):
        store.upsert(Entry(entry_type="factual", memory=f"f{i}"), [1.0, 0.0, 0.0, 0.0])
    for i in range(5):
        store.upsert(Entry(entry_type="state", memory=f"s{i}", subject="A",
                           attribute=f"a{i}", value="v"), [0.0, 1.0, 0.0, 0.0])

    assert len(store.scroll(limit=3)) == 6          # 3 per collection, not 3 total
    assert len(store.scroll(limit=3, entry_type="state")) == 3


def test_state_chain_and_clusterer_survive_the_split(store):
    """End-to-end: the two writers that fan out must still work."""
    from layermem.configs.config import LayerMemConfig
    from layermem.core.schema import Entry, iso_to_ts
    from layermem.pipeline.clusterer import Clusterer
    from layermem.storage.state_chain import StateChain

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0] if isinstance(text, str) else [[1.0, 0.0, 0.0, 0.0]] * len(text)

    config = LayerMemConfig(theta_cluster=0.5, theta_dup=0.9)
    embedder = FixedEmbedder()

    clusterer = Clusterer(store, embedder, config)
    first = Entry(entry_type="factual", memory="Alice lives in Helsinki")
    store.upsert(first, [1.0, 0.0, 0.0, 0.0])
    store.update_payload(first.id, {"topic_id": "t1"})

    assignment = clusterer.assign(Entry(entry_type="factual", memory="Alice lives in Helsinki"),
                                  [1.0, 0.0, 0.0, 0.0])
    assert assignment.duplicate_of is not None

    chain = StateChain(store, embedder, config)
    early = Entry(entry_type="state", memory="Alice — residence: Hangzhou", subject="Alice",
                  attribute="residence", value="Hangzhou",
                  float_event_time=iso_to_ts("2024-01-01T00:00:00"),
                  float_mention_time=iso_to_ts("2024-01-01T00:00:00"))
    assert chain.handle(early).action == "ADD"

    late = Entry(entry_type="state", memory="Alice — residence: Helsinki", subject="Alice",
                 attribute="residence", value="Helsinki",
                 float_event_time=iso_to_ts("2024-08-01T00:00:00"),
                 float_mention_time=iso_to_ts("2024-08-01T00:00:00"))
    assert chain.handle(late).action == "UPDATE"
    assert chain.get_current("Alice", "residence").value == "Helsinki"
    # The superseded row is still in the state collection, flagged not deleted.
    assert store.count(entry_type="state", status="superseded") == 1
