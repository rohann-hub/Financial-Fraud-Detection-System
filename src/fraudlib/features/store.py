"""Thin wrapper around Feast: one way to open the store, write rows and read them back."""
from __future__ import annotations

import os

import pandas as pd
from feast import FeatureStore

from fraudlib.config import ROOT, path
from fraudlib.features.engine import CARD_FEATURES, DEVICE_FEATURES, USER_FEATURES

BEHAVIOUR_REFS = ([f"user_features:{f}" for f in USER_FEATURES] + [f"card_features:{f}" for f in CARD_FEATURES]
                  + [f"device_features:{f}" for f in DEVICE_FEATURES])
GRAPH_REFS = ["user_graph:user_graph_risk", "card_graph:card_graph_risk", "device_graph:device_graph_risk"]
ALL_REFS = BEHAVIOUR_REFS + GRAPH_REFS


def get_store() -> FeatureStore:
    feast_dir = ROOT / "data/feast"
    feast_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FEATURE_DATA_DIR", path("feature_dir").as_posix())
    os.environ.setdefault("FEAST_REGISTRY", (feast_dir / "registry.db").as_posix())
    os.environ.setdefault("FEAST_ONLINE_PATH", (feast_dir / "online.db").as_posix())
    # Local runs use SQLite; Compose and Kubernetes set REDIS_CONNECTION and share one Redis.
    name = "feature_store.redis.yaml" if os.environ.get("REDIS_CONNECTION") else "feature_store.yaml"
    return FeatureStore(fs_yaml_file=ROOT / "feature_repo" / name)


def apply(store: FeatureStore) -> None:
    """Register the definitions from feature_repo/definitions.py."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("fraud_feature_defs", ROOT / "feature_repo/definitions.py")
    defs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = defs
    spec.loader.exec_module(defs)
    store.apply([defs.user, defs.card, defs.device, defs.user_features, defs.card_features, defs.device_features,
                 defs.user_graph, defs.card_graph, defs.device_graph, defs.fraud_scoring])


def historical(store: FeatureStore, events: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    """Point-in-time join: each transaction gets each entity's newest row at or before its own time."""
    entity_df = events[["transaction_id", "user_id", "card_id", "device_id"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(events["event_ts"], utc=True).to_numpy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)
    out = store.get_historical_features(entity_df=entity_df, features=refs).to_df()
    names = [r.split(":")[1] for r in refs]
    return out.set_index("transaction_id").loc[events["transaction_id"], names].reset_index()


def write_online(store: FeatureStore, view: str, rows: pd.DataFrame) -> None:
    if len(rows):
        store.write_to_online_store(view, rows)


def online(store: FeatureStore, entity_rows: list[dict], refs: list[str] = ALL_REFS) -> pd.DataFrame:
    values = store.get_online_features(features=refs, entity_rows=entity_rows).to_dict()
    return pd.DataFrame(values)
