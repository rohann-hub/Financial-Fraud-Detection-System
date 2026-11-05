"""Behaviour of the shared feature engine, including the rules that prevent subtle bugs."""
import pandas as pd
import pytest

from conftest import txn
from fraudlib.features.engine import FeatureEngine, to_rows


def run(events):
    e = FeatureEngine()
    return [e.process(t["ts"], t["user_id"], t["card_id"], t["device_id"], t["location_id"], t["amount"]) for t in events]


def test_velocity_window_expires():
    out = run([txn(1, 0), txn(2, 1800), txn(3, 3600), txn(4, 3601)])
    assert out[1][0]["user_txn_count_1h"] == 2
    assert out[2][0]["user_txn_count_1h"] == 2          # the t=0 event is exactly 1h old and has left
    assert out[3][0]["user_txn_count_24h"] == 4


def test_amount_versus_average_excludes_current_transaction():
    out = run([txn(1, 0, amount=100), txn(2, 10, amount=100), txn(3, 20, amount=500)])
    assert out[0][0]["user_amount_vs_avg"] == 1.0       # no history yet
    assert out[2][0]["user_amount_vs_avg"] == pytest.approx(5.0)
    assert out[2][0]["user_amount_mean"] == pytest.approx(100.0)


def test_device_and_location_change():
    out = run([txn(1, 0), txn(2, 10), txn(3, 20, device="d2", location="l2"), txn(4, 30)])
    assert out[0][0]["user_new_device"] == 0            # the first device is not a "change"
    assert out[2][0]["user_new_device"] == 1 and out[2][0]["user_device_changed"] == 1
    assert out[2][0]["user_new_location"] == 1 and out[2][0]["user_location_changed"] == 1
    assert out[3][0]["user_new_device"] == 0 and out[3][0]["user_device_changed"] == 1


def test_unknown_device_is_not_treated_as_a_shared_device():
    out = run([txn(1, 0, user="a", device="unknown"), txn(2, 1, user="b", device="unknown")])
    assert out[0][2] is None and out[1][2] is None
    assert out[1][0]["user_new_device"] == 0


def test_device_sees_many_accounts():
    out = run([txn(i, i, user=f"u{i}", card=f"c{i}", device="ring") for i in range(5)])
    assert out[4][2]["device_distinct_users_7d"] == 5
    assert out[4][2]["device_distinct_cards_7d"] == 5


def test_distinct_counts_forget_after_seven_days():
    day = 86400
    out = run([txn(1, 0, user="a"), txn(2, 8 * day, user="b")])
    assert out[1][2]["device_distinct_users_7d"] == 1
    assert out[1][2]["device_user_count_total"] == 2


def test_state_snapshot_resumes_identically(tmp_path, make_events):
    ev = make_events([txn(i, i * 300, amount=10 + i) for i in range(20)])
    whole = to_rows(FeatureEngine(), ev)["user_features"]
    e = FeatureEngine()
    first = to_rows(e, ev.iloc[:10])["user_features"]
    e.save(tmp_path / "s.pkl")
    second = to_rows(FeatureEngine.load(tmp_path / "s.pkl"), ev.iloc[10:])["user_features"]
    pd.testing.assert_frame_equal(whole.reset_index(drop=True), pd.concat([first, second], ignore_index=True))


def test_timestamps_work_for_any_pandas_datetime_unit(make_events):
    ev = make_events([txn(1, 0), txn(2, 100)])
    a = to_rows(FeatureEngine(), ev.assign(event_ts=ev["event_ts"].dt.as_unit("ns")))["user_features"]
    b = to_rows(FeatureEngine(), ev.assign(event_ts=ev["event_ts"].dt.as_unit("ms")))["user_features"]
    assert a["user_secs_since_last"].iloc[1] == b["user_secs_since_last"].iloc[1] == 100
