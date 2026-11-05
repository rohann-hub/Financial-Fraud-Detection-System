"""The graph must never see the future."""
import numpy as np
import pandas as pd
import pytest
import torch

from fraudlib.models import gnn as G


def _df():
    ts = pd.date_range("2018-01-01", periods=120, freq="D", tz="UTC")
    return pd.DataFrame({
        "event_ts": ts, "amount": 50.0, "is_fraud": ([0] * 60 + [1] * 60),
        "user_id": [f"u{i % 10}" for i in range(120)], "card_id": [f"c{i % 4}" for i in range(120)],
        "device_id": ["unknown" if i % 3 == 0 else f"d{i % 5}" for i in range(120)]})


def test_graph_rejects_rows_after_cutoff():
    df = _df()
    with pytest.raises(AssertionError):
        G.build_snapshot(df, df.event_ts[50], 7)


def test_recent_fraud_is_hidden_until_label_arrives():
    df = _df()
    cutoff = df.event_ts[70]
    hist = df[df.event_ts < cutoff]
    late = G._node_stats(hist, "user_id", cutoff, pd.Timedelta(days=30))
    early = G._node_stats(hist, "user_id", cutoff, pd.Timedelta(days=0))
    assert late["known_fraud"].sum() == 0        # fraud started on day 60; a 30-day delay hides all of it
    assert early["known_fraud"].sum() > 0


def test_unknown_devices_create_no_node():
    df = _df()
    cutoff = df.event_ts[100]
    snap = G.build_snapshot(df[df.event_ts < cutoff], cutoff, 7)
    assert "unknown" not in snap.index["device"]


def test_edges_are_valid_and_mirrored():
    df = _df()
    cutoff = df.event_ts[100]
    snap = G.build_snapshot(df[df.event_ts < cutoff], cutoff, 7)
    for rel in G.EDGES:
        e = snap.data[rel].edge_index
        assert e.min() >= 0
    fwd = snap.data["user", "uses", "card"].edge_index
    rev = snap.data["card", "used_by", "user"].edge_index
    assert torch.equal(fwd, rev.flip(0))


def test_model_runs_inductively_on_a_larger_graph():
    df = _df()
    a, b = df.event_ts[50], df.event_ts[100]
    small = G.build_snapshot(df[df.event_ts < a], a, 7, future=df[df.event_ts.between(a, b, inclusive="left")])
    big = G.build_snapshot(df[df.event_ts < b], b, 7, future=df[df.event_ts >= b])
    model, curve = G.train([small], big, {"hidden": 8, "layers": 2, "dropout": 0, "lr": .01, "weight_decay": 0, "epochs": 3, "patience": 3})
    risk = G.node_risk(model, big)
    assert set(risk) == set(G.NODE_KEYS) and np.isfinite(risk["user"]).all()
