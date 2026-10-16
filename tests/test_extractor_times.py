import pytest

from layermem.memory.llm.utils import conversation_text
from layermem.pipeline.extractor import Extractor


class FixedLLM:
    def __init__(self, item):
        self.item = item
        self.calls = 0

    def extract_memories(self, batch):
        self.calls += 1
        return {"relational": [self.item]}


@pytest.mark.parametrize("event_time", [None, "null", "unknown", ""])
def test_unknown_event_uses_source_mention_not_batch_latest(event_time):
    llm = FixedLLM({"memory": "Alice met Bob at an unknown past date",
                    "source_message_id": 1, "event_time": event_time,
                    "mention_time": "2099-01-01T00:00:00Z"})
    batch = [
        {"content": "I met Bob", "timestamp": "2023-05-25T14:00:00+08:00", "session_id": "a"},
        {"content": "Later message", "timestamp": "2023-06-01T00:00:00Z", "session_id": "b"},
    ]
    entry = Extractor(llm).extract(batch)["relational"][0]
    assert entry.event_time == "2023-05-25T06:00:00.000000Z"
    assert entry.mention_time == "2023-05-25T06:00:00.000000Z"
    assert entry.source_dialog_ids == ["a"]
    assert entry.event_key == entry.mention_time


@pytest.mark.parametrize("source_id", [None, 0, -1, 2, True, "1"])
def test_invalid_source_is_retried_then_rejected(source_id):
    llm = FixedLLM({"memory": "fact", "source_message_id": source_id})
    with pytest.raises(RuntimeError, match="failed after two attempts") as error:
        Extractor(llm).extract([{"content": "fact", "timestamp": "2023-05-25"}])
    assert "source_message_id" in str(error.value.__cause__)
    assert llm.calls == 2


@pytest.mark.parametrize("timestamp,event", [(None, None), ("bad", None), ("2023-05-25", "2023-02-30")])
def test_missing_source_time_or_invalid_event_is_not_replaced_by_now(timestamp, event):
    llm = FixedLLM({"memory": "fact", "source_message_id": 1, "event_time": event})
    with pytest.raises(RuntimeError):
        Extractor(llm).extract([{"content": "fact", "timestamp": timestamp}])
    assert llm.calls == 2


def test_message_numbers_match_original_batch_even_with_empty_messages():
    batch = [{"content": ""}, {"content": "fact", "timestamp": "2023-05-25"}]
    assert conversation_text(batch) == "[message 2] [2023-05-25] user: fact"
    entry = Extractor(FixedLLM({"memory": "fact", "source_message_id": 2})).extract(batch)["relational"][0]
    assert entry.mention_time == "2023-05-25T00:00:00.000000Z"
