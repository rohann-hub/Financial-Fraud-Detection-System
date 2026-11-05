"""Retraining must learn from live traffic, and must never see rows that have not happened yet."""
import json

import numpy as np
import pandas as pd
import pytest

import fraudlib.pipeline as P
from fraudlib.models import frame as F
from fraudlib.monitor.drift import has_drifted


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "path", lambda name: tmp_path)
    return tmp_path


def _rows(start, n, prefix, fraud=0):
    return pd.DataFrame({"transaction_id": [f"{prefix}{i}" for i in range(n)],
                         "event_ts": pd.date_range(start, periods=n, freq="h", tz="UTC"), "is_fraud": fraud, "amount": 10.0})


def _history(d, history, future):
    both = pd.concat([history, future], ignore_index=True)
    t2 = future["event_ts"].iloc[0]
    P._write_split(both, history["event_ts"].iloc[60], t2, "raw", t2)


def test_unreplayed_rows_never_reach_retraining(dirs):
    history, future = _rows("2018-01-01", 100, "h"), _rows("2018-03-01", 60, "f", fraud=1)
    _history(dirs, history, future)
    P.roll_forward(future.iloc[:30])        # only the first 30 replayed rows have labels
    out = pd.read_parquet(dirs / "transactions.parquet")
    assert not set(future["transaction_id"].iloc[30:]) & set(out["transaction_id"])
    assert len(out) == 130


def test_live_window_splits_into_train_val_and_gate(dirs):
    history, live = _rows("2018-01-01", 100, "h"), _rows("2018-03-01", 30, "f")
    _history(dirs, history, live)
    P.roll_forward(live)
    df, t1, t2 = P._load()
    assert ((df["event_ts"] >= t1) & (df["event_ts"] < t2)).sum() == 10     # validation: middle third
    assert (df["event_ts"] >= t2).sum() == 10                               # gate: newest third
    assert (df["event_ts"] < t1).sum() == 110                               # train gained the oldest third


def test_rolling_forward_twice_does_not_duplicate_live_rows(dirs):
    history, live = _rows("2018-01-01", 100, "h"), _rows("2018-03-01", 30, "f")
    _history(dirs, history, live)
    P.roll_forward(live.iloc[:15])
    P.roll_forward(live)
    out = pd.read_parquet(dirs / "transactions.parquet")
    assert out["transaction_id"].is_unique and len(out) == 130


def test_blend_treats_absence_from_the_graph_as_unknown_not_safe():
    stacker = {"intercept": 0.0, "xgb_weight": 1.0, "gnn_weight": 1.0, "unseen_weight": 0.0}
    xgb_p = np.array([.5, .5])
    p = F.blend(xgb_p, np.array([0.0, .01]), stacker)
    assert p[0] == pytest.approx(.5)         # unseen: graph has no opinion
    assert p[1] < .05                        # seen and clean: graph lowers the score


def test_drift_direction_depends_on_the_test_evidently_chose():
    distance = "ValueDrift(column=amount,method=Wasserstein distance (normed),threshold=0.1)"
    pvalue = "ValueDrift(column=amount,method=K-S p_value,threshold=0.05)"
    assert has_drifted(distance, .5) and not has_drifted(distance, .01)
    assert has_drifted(pvalue, .01) and not has_drifted(pvalue, .5)


def test_serving_booster_matches_the_model_that_was_evaluated(tmp_path):
    import xgboost as xgb
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(600, 5)).astype("float32"), rng.integers(0, 2, 600)
    model = xgb.XGBClassifier(n_estimators=60, max_depth=3, early_stopping_rounds=3, eval_metric="logloss")
    model.fit(X[:400], y[:400], eval_set=[(X[400:], y[400:])], verbose=False)
    assert model.best_iteration < 59             # labels are noise, so stopping must have kicked in
    P.save_best_trees(model, tmp_path / "m.json")
    booster = xgb.Booster()
    booster.load_model(tmp_path / "m.json")
    assert np.abs(booster.inplace_predict(X) - model.predict_proba(X)[:, 1]).max() < 1e-6
