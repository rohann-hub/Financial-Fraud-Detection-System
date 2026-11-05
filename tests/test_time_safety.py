"""The leakage rules. These are the tests an evaluator should care about most."""
import numpy as np
import pandas as pd
import pytest

from fraudlib.data.ieee import time_split, split_mask
from fraudlib.models.metrics import decisions, evaluate, thresholds
from fraudlib.serving.pipeline import split_without_repeats


def _tx(n=1000):
    return pd.DataFrame({"event_ts": pd.date_range("2018-01-01", periods=n, freq="min", tz="UTC")})


def test_split_is_ordered_in_time():
    df = _tx()
    t1, t2 = time_split(df, .6, .2)
    s = split_mask(df, t1, t2)
    assert (s == "train").sum() == 600 and (s == "val").sum() == 200 and (s == "test").sum() == 200
    assert df.event_ts[s == "train"].max() < df.event_ts[s == "val"].min()
    assert df.event_ts[s == "val"].max() < df.event_ts[s == "test"].min()


def test_accuracy_trap_is_visible_in_metrics():
    y = np.array([0] * 990 + [1] * 10)
    m = evaluate(y, np.zeros(1000) + .01)      # a useless constant model
    assert m["pr_auc"] == pytest.approx(.01)    # no better than the base rate
    assert m["recall_at_precision_90"] == 0


def test_thresholds_give_analysts_their_own_band_below_block():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < .03).astype(int)
    p = np.clip(rng.normal(.1 + .6 * y, .15), 0, 1)
    th = thresholds(y, p, alert_rate=.02, block_precision=.9)
    assert th["block"] > th["review"]
    assert ((p >= th["review"]) & (p < th["block"])).mean() == pytest.approx(.02, abs=.002)
    assert y[p >= th["block"]].mean() >= .9


def test_unreachable_precision_never_auto_blocks():
    y = np.array([0, 1] * 500)
    th = thresholds(y, np.full(1000, .5) + np.random.default_rng(0).normal(0, .01, 1000), .02, .99)
    assert th["block"] > 1


def test_decisions_do_not_count_a_transaction_twice():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([.9, .8, .6, .3, .2, .1])
    d = decisions(y, p, {"review": .5, "block": .75})
    assert d["block_rate"] == pytest.approx(2 / 6) and d["review_rate"] == pytest.approx(1 / 6)
    assert d["block_precision"] == 1 and d["review_precision"] == 0
    assert d["fraud_caught_share"] == pytest.approx(2 / 3)


def test_unknown_device_does_not_break_up_a_batch():
    ev = [{"user_id": str(i), "card_id": f"c{i}", "device_id": "unknown"} for i in range(50)]
    assert len(split_without_repeats(ev)) == 1


def test_batches_never_contain_the_same_entity_twice():
    ev = [{"user_id": u, "card_id": c, "device_id": d} for u, c, d in
          [("a", "1", "x"), ("b", "2", "y"), ("a", "3", "z"), ("c", "2", "w"), ("d", "4", "x")]]
    batches = split_without_repeats(ev)
    assert sum(len(b) for b in batches) == 5
    for b in batches:
        for key in ("user_id", "card_id", "device_id"):
            assert len({e[key] for e in b}) == len(b)


def test_batches_keep_arrival_order():
    ev = [{"user_id": str(i % 3), "card_id": "c" + str(i), "device_id": "d" + str(i), "i": i} for i in range(9)]
    assert [e["i"] for b in split_without_repeats(ev) for e in b] == list(range(9))
