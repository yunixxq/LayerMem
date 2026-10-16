import pytest

from layermem.core.schema import Entry, normalize_time


def test_offsets_and_fractional_precision_sort_chronologically():
    earlier = Entry(event_time="2023-05-25T14:00:00+08:00")
    later = Entry(event_time="2023-05-25T07:00:00Z")
    assert earlier.event_time == "2023-05-25T06:00:00.000000Z"
    assert earlier.event_key < later.event_key
    assert normalize_time("2023-05-25T07:00:00.1Z") == "2023-05-25T07:00:00.100000Z"


def test_legacy_payload_migrates_all_stored_times():
    entry = Entry.from_payload({"float_event_time": 1685020440.0,
                                "float_mention_time": 1685020440.0,
                                "updated_at": 1685020440.0,
                                "compressed_until": 1685020440.0})
    payload = entry.payload()
    for key in ("event_time", "mention_time", "updated_at", "compressed_until"):
        assert payload[key] == "2023-05-25T13:14:00.000000Z"
    assert not any(key.startswith("float_") for key in payload)
    assert Entry.from_payload(payload) == entry


def test_new_fields_override_legacy_values_after_patch():
    entry = Entry.from_payload({"float_event_time": 0, "event_time": "2024-01-01T00:00:00Z"})
    assert entry.event_time == "2024-01-01T00:00:00.000000Z"


def test_epoch_naive_and_invalid_dates():
    assert normalize_time(0) == "1970-01-01T00:00:00.000000Z"
    assert normalize_time("2023-05-25T13:14:00") == "2023-05-25T13:14:00.000000Z"
    with pytest.raises(ValueError):
        Entry(event_time="not-a-date")
    assert Entry(event_time=None).event_time is None
