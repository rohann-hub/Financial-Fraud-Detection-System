"""Heterogeneous graph model for connected fraud.

Graph:  user --uses--> card,  user --uses--> device,  device --sees--> card (each with a reverse edge).

Time discipline, which is where fraud GNNs most often leak:
    graph(T)      contains only transactions before T
    node features use only fraud labels that were already known at T (see label_delay_days)
    node target   "does this node show fraud AFTER T"
The model therefore learns to predict future fraud from past structure. GraphSAGE is inductive, so the
same weights work on a later, larger graph, and we train on one cut and score on another.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from fraudlib.features.engine import UNKNOWN

NODE_KEYS = {"user": "user_id", "card": "card_id", "device": "device_id"}
EDGES = [("user", "uses", "card"), ("card", "used_by", "user"), ("user", "on", "device"),
         ("device", "hosts", "user"), ("device", "sees", "card"), ("card", "seen_on", "device")]


@dataclass
class Snapshot:
    data: HeteroData
    index: dict[str, pd.Index]     # node id per position, per node type


def _node_stats(df: pd.DataFrame, key: str, cutoff: pd.Timestamp, label_delay: pd.Timedelta) -> pd.DataFrame:
    known = df["event_ts"] < cutoff - label_delay     # labels newer than this have not arrived yet
    g = df.assign(known_fraud=df["is_fraud"].where(known, 0), known_amt=df["amount"].where(known & (df["is_fraud"] == 1), 0)).groupby(key)
    s = pd.DataFrame({
        "txn_count": np.log1p(g.size()),
        "amount_mean": np.log1p(g["amount"].mean()),
        "amount_std": np.log1p(g["amount"].std().fillna(0)),
        "known_fraud": np.log1p(g["known_fraud"].sum()),
        "known_fraud_rate": g["known_fraud"].mean(),
        "days_since_last": (cutoff - g["event_ts"].max()).dt.total_seconds() / 86400 / 30,
        "active_days": (g["event_ts"].max() - g["event_ts"].min()).dt.total_seconds() / 86400 / 30,
    })
    return s


def build_snapshot(history: pd.DataFrame, cutoff: pd.Timestamp, label_delay_days: int,
                   future: pd.DataFrame | None = None) -> Snapshot:
    """`history` must only contain rows before `cutoff`. `future` supplies targets when training."""
    assert (history["event_ts"] < cutoff).all(), "graph must not contain anything after the cutoff"
    delay = pd.Timedelta(days=label_delay_days)
    data, index = HeteroData(), {}
    for ntype, key in NODE_KEYS.items():
        rows = history[history[key] != UNKNOWN]
        stats = _node_stats(rows, key, cutoff, delay)
        index[ntype] = stats.index
        data[ntype].x = torch.tensor(stats.to_numpy(dtype="float32"))
        if future is not None:
            bad = set(future.loc[(future["is_fraud"] == 1) & (future[key] != UNKNOWN), key])
            seen = set(future[key])
            data[ntype].y = torch.tensor(stats.index.isin(bad), dtype=torch.float32)
            # Only nodes that transact again teach us anything about future fraud.
            data[ntype].mask = torch.tensor(stats.index.isin(seen))

    def edge(a: str, b: str) -> torch.Tensor:
        ka, kb = NODE_KEYS[a], NODE_KEYS[b]
        rows = history.loc[(history[ka] != UNKNOWN) & (history[kb] != UNKNOWN), [ka, kb]].drop_duplicates()
        src = index[a].get_indexer(rows[ka])
        dst = index[b].get_indexer(rows[kb])
        return torch.tensor(np.stack([src, dst]), dtype=torch.long)

    for a, rel, b in EDGES[0::2]:
        e = edge(a, b)
        data[a, rel, b].edge_index = e
    for (a, _, b), (ra, rrel, rb) in zip(EDGES[0::2], EDGES[1::2]):
        data[ra, rrel, rb].edge_index = data[a, _, b].edge_index.flip(0)
    return Snapshot(data, index)


class FraudGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.encode = nn.ModuleDict({t: nn.Linear(in_dim, hidden) for t in NODE_KEYS})
        self.convs = nn.ModuleList(
            HeteroConv({e: SAGEConv((hidden, hidden), hidden) for e in EDGES}, aggr="sum") for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.ModuleDict({t: nn.Linear(hidden, 1) for t in NODE_KEYS})

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        h = {t: torch.relu(self.encode[t](data[t].x)) for t in NODE_KEYS}
        for conv in self.convs:
            out = conv(h, data.edge_index_dict)
            # skip connection: isolated nodes keep their own features
            h = {t: self.dropout(torch.relu(out.get(t, 0) + h[t])) for t in h}
        return {t: self.head[t](h[t]).squeeze(-1) for t in h}


def _loss(logits: dict, data: HeteroData) -> torch.Tensor:
    total = 0.0
    for t in NODE_KEYS:
        m = data[t].mask
        if not m.any():
            continue
        y = data[t].y[m]
        pos_weight = (len(y) - y.sum()).clamp(min=1) / y.sum().clamp(min=1)
        total = total + nn.functional.binary_cross_entropy_with_logits(logits[t][m], y, pos_weight=pos_weight)
    return total


def _pr_auc(logits: dict, data: HeteroData) -> float:
    from sklearn.metrics import average_precision_score
    ys, ps = [], []
    for t in NODE_KEYS:
        m = data[t].mask
        ys.append(data[t].y[m].numpy())
        ps.append(logits[t][m].numpy())
    y = np.concatenate(ys)
    return float(average_precision_score(y, np.concatenate(ps))) if y.sum() else 0.0


def train(fits: list[Snapshot], val: Snapshot, cfg: dict, seed: int = 7) -> tuple[FraudGNN, list[float]]:
    """Each fit snapshot is the graph as it stood at one past date, so the model sees rings at many ages."""
    torch.manual_seed(seed)
    model = FraudGNN(val.data["user"].x.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, stale, curve = -1.0, None, 0, []
    for _ in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        loss = sum(_loss(model(f.data), f.data) for f in fits) / len(fits)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            score = _pr_auc(model(val.data), val.data)
        curve.append(score)
        if score > best:
            best, stale = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, curve


@torch.no_grad()
def node_risk(model: FraudGNN, snap: Snapshot) -> dict[str, pd.Series]:
    model.eval()
    logits = model(snap.data)
    return {t: pd.Series(torch.sigmoid(logits[t]).numpy(), index=snap.index[t], dtype="float32") for t in NODE_KEYS}
