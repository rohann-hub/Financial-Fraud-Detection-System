"""Assemble the model input. Called with the same arguments in training and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlib.data.ieee import CATEGORICAL, RAW_NUMERIC
from fraudlib.features.engine import CARD_FEATURES, DEVICE_FEATURES, USER_FEATURES

ENGINEERED = USER_FEATURES + CARD_FEATURES + DEVICE_FEATURES
GRAPH = ["user_graph_risk", "card_graph_risk", "device_graph_risk"]
# Raw columns arrive in the event itself; everything in ENGINEERED and GRAPH comes from Feast.
EVENT_COLUMNS = ["amount", "event_ts", *RAW_NUMERIC, *CATEGORICAL]
MODEL_COLUMNS = ["amount", "log_amount", "hour", "day_of_week", *RAW_NUMERIC,
                 *[f"{c}_code" for c in CATEGORICAL], *ENGINEERED]


def fit_vocab(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {c: {v: i for i, v in enumerate(sorted(train[c].dropna().astype(str).unique()))} for c in CATEGORICAL}


def build_frame(events: pd.DataFrame, feast_values: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """events: one row per transaction. feast_values: aligned engineered columns from Feast."""
    ts = pd.to_datetime(events["event_ts"], utc=True, format="ISO8601")
    x = pd.DataFrame(index=events.index)
    x["amount"] = pd.to_numeric(events["amount"]).astype("float32")
    x["log_amount"] = np.log1p(x["amount"])
    x["hour"] = ts.dt.hour.astype("float32")
    x["day_of_week"] = ts.dt.dayofweek.astype("float32")
    for c in RAW_NUMERIC:
        x[c] = pd.to_numeric(events[c], errors="coerce").astype("float32")
    for c in CATEGORICAL:
        x[f"{c}_code"] = events[c].map(vocab[c]).fillna(-1).astype("float32")
    for c in ENGINEERED:
        x[c] = pd.to_numeric(feast_values[c], errors="coerce").astype("float32").to_numpy()
    return x[MODEL_COLUMNS]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def graph_score(values: pd.DataFrame) -> np.ndarray:
    """A transaction is as risky as its riskiest connected node. Never-seen nodes count as 0."""
    return np.nan_to_num(values[GRAPH].to_numpy(dtype="float64"), nan=0.0).max(axis=1)


def blend_inputs(xgb_p: np.ndarray, gnn_p: np.ndarray) -> np.ndarray:
    """Stacker inputs. Absence from the graph gets its own term: 0 means "unknown", not "very safe",
    and pushing it through a logit would wrongly punish or forgive every new account."""
    seen = gnn_p > 0
    graph = np.where(seen, logit(np.clip(gnn_p, .01, .99)), 0.0)
    return np.c_[logit(xgb_p), graph, (~seen).astype("float64")]


def blend(xgb_p: np.ndarray, gnn_p: np.ndarray, stacker: dict) -> np.ndarray:
    """Calibrated blend: logistic regression over the XGBoost and graph logits."""
    a = blend_inputs(xgb_p, gnn_p)
    z = stacker["intercept"] + stacker["xgb_weight"] * a[:, 0] + stacker["gnn_weight"] * a[:, 1] + stacker["unseen_weight"] * a[:, 2]
    return 1 / (1 + np.exp(-z))
