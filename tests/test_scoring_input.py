"""One bad message must go to the dead-letter topic, not stall the partition."""
import importlib
import sys

import pytest


@pytest.fixture
def normalise():
    sys.path.insert(0, "services/scoring")
    sys.modules.pop("app", None)
    mod = importlib.import_module("app")
    yield mod.normalise
    sys.modules.pop("app", None)
    sys.path.remove("services/scoring")


GOOD = {"transaction_id": "1", "event_ts": "2018-05-07T00:00:00.001000+00:00", "amount": 5, "user_id": "u", "card_id": "c"}


@pytest.mark.parametrize("bad", [
    [1, 2], {"transaction_id": "1"}, {**GOOD, "amount": "lots"}, {**GOOD, "amount": -1}, {**GOOD, "amount": float("inf")},
    {**GOOD, "event_ts": "garbage"}])
def test_rejects_what_would_crash_the_scorer(normalise, bad):
    with pytest.raises((ValueError, TypeError)):
        normalise(bad)


def test_accepts_a_minimal_event_and_fills_defaults(normalise):
    e = normalise(dict(GOOD))
    assert e["device_id"] == "unknown" and e["amount"] == 5.0 and "C1" in e


def test_naive_timestamp_becomes_utc(normalise):
    assert normalise({**GOOD, "event_ts": "2018-05-07 10:00:00"})["event_ts"].endswith("+00:00")
