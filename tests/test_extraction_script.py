import io
import json
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory_toolkits"))
from memory_extraction import extract_batches, write_review, save_batches


class Sequence:
    def __init__(self, items, identifier):
        self.items = items
        self.metadata = {"id": identifier}

    def __iter__(self):
        return iter(self.items)


def trajectory(identifier, count):
    messages = [SimpleNamespace(role="user", content=f"Fact {i}", timestamp="2024-01-01T12:00:00",
                               metadata={"name": "Alice"}) for i in range(count)]
    return Sequence([Sequence(messages, "session-1")], identifier)


def test_batches_tail_and_trajectory_isolation():
    class LLM:
        calls = 0

        def extract_memories(self, batch):
            self.calls += 1
            return {"factual": [{"memory": "Alice has a fact", "source_message_id": 1}], "relational": [], "state": [],
                    "meta": {"raw_response": "original response"}}

    llm = LLM()
    records = list(extract_batches([trajectory("one", 3), trajectory("two", 1)], llm, 2, 10000))
    assert [len(r["messages"]) for r in records] == [2, 1, 1]
    assert [r["trigger"] for r in records] == ["threshold", "trajectory_end", "trajectory_end"]
    assert llm.calls == 3
    assert all(not {"fallback", "model", "api_type", "buffer_thresholds"}.intersection(r) for r in records)
    assert records[0]["attempts"][0]["result"]["meta"]["raw_response"] == "original response"
    atom = records[0]["atoms"]["factual"][0]
    assert atom["layer"] == "L1"
    assert atom["topic_id"] is None
    assert atom["source_dialog_ids"] == ["session-1"]


def test_failure_stops_extraction_with_batch_context():
    class FailingLLM:
        def extract_memories(self, batch):
            raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="one / batch 1: 无法调用 LLM"):
        next(extract_batches([trajectory("one", 1)], FailingLLM()))


def test_token_threshold_and_successful_retry():
    class RetryLLM:
        calls = 0

        def extract_memories(self, batch):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("invalid JSON")
            return {"factual": [], "relational": [], "state": []}

    record = next(extract_batches([trajectory("one", 1)], RetryLLM(), 10, 1))
    assert record["trigger"] == "threshold"
    assert [a["success"] for a in record["attempts"]] == [False, True]
    output = io.StringIO()
    write_review(record, output)
    assert "抽取成功" in output.getvalue()


def test_json_snapshot_survives_interrupted_write(tmp_path, monkeypatch):
    path = tmp_path / "batches.json"
    save_batches(path, [{"batch_index": 1}])
    assert json.loads(path.read_text()) == [{"batch_index": 1}]

    def interrupted_dump(records, output, **kwargs):
        output.write("[")
        raise KeyboardInterrupt

    monkeypatch.setattr("memory_extraction.json.dump", interrupted_dump)
    with pytest.raises(KeyboardInterrupt):
        save_batches(path, [{"batch_index": 1}, {"batch_index": 2}])
    assert json.loads(path.read_text()) == [{"batch_index": 1}]
    assert list(tmp_path.iterdir()) == [path]
