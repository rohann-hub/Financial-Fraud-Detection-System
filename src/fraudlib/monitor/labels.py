"""Gather live transactions with their best-known label, for retraining.

Order of trust: an analyst's decision, then a chargeback feed if one exists.
Transactions with neither are left out rather than assumed legitimate, except those old enough that a
chargeback would have arrived. In this project the replay's own labels stand in for that feed.
"""
from __future__ import annotations

import glob
import os

import httpx
import pandas as pd

from fraudlib.config import path


def live_events() -> pd.DataFrame:
    files = sorted(glob.glob(str(path("live_dir") / "scored-*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("transaction_id", keep="last")


def analyst_labels() -> dict[str, int]:
    url = os.environ.get("ANALYST_API_URL")
    if not url:
        return {}
    token = os.environ.get("INTERNAL_TOKEN", "")
    r = httpx.get(f"{url}/api/labels/export", headers={"x-internal-token": token}, timeout=30)
    r.raise_for_status()
    return {x["transaction_id"]: int(x["label"]) for x in r.json()}


def _labels() -> pd.Series:
    """Analyst decisions override the chargeback feed."""
    labels = pd.Series(analyst_labels(), dtype="float64")
    truth_file = path("live_dir") / "chargebacks.parquet"
    if truth_file.exists():
        cb = pd.read_parquet(truth_file).set_index("transaction_id")["is_fraud"].astype("float64")
        labels = pd.concat([cb[~cb.index.isin(labels.index)], labels])
    return labels


def _base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, format="ISO8601")
    drop = ("score", "xgb_score", "gnn_score", "decision", "model_version")
    return df[[c for c in df.columns if not c.startswith("feat_") and c not in drop]]


def labelled_live_events() -> pd.DataFrame:
    """Rows retraining may learn from: confirmed labels only."""
    df = live_events()
    if df.empty:
        return df
    df["is_fraud"] = df["transaction_id"].map(_labels())
    df = df[df["is_fraud"].notna()].copy()
    df["is_fraud"] = df["is_fraud"].astype("int8")
    return _base_columns(df)


def known_label_events() -> pd.DataFrame:
    """All live rows for the graph, which needs the structure even where no label has arrived.
    A missing label counts as "no known fraud", which is exactly what a graph node feature means."""
    df = live_events()
    if df.empty:
        return df
    df["is_fraud"] = df["transaction_id"].map(_labels()).fillna(0).astype("int8")
    return _base_columns(df)
