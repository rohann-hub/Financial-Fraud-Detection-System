"""Real-time scoring: event in, decision out. Used by the Kafka consumer and by POST /score."""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudlib.config import ROOT, params, path
from fraudlib.features import store as fs
from fraudlib.features.engine import UNKNOWN, FeatureEngine, to_rows
from fraudlib.models import frame as F

log = logging.getLogger("scoring")
ALIAS = "champion"


def _client():
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'data/mlflow.db').as_posix()}"))
    return mlflow, mlflow.MlflowClient()


def champion_version() -> str:
    _, client = _client()
    return client.get_model_version_by_alias(params()["mlflow"]["model_name"], ALIAS).version


def fetch_bundle() -> Path:
    mlflow, client = _client()
    mv = client.get_model_version_by_alias(params()["mlflow"]["model_name"], ALIAS)
    return Path(mlflow.artifacts.download_artifacts(mv.source))


class Bundle:
    def __init__(self, folder: Path, version: str):
        self.version = version
        self.meta = json.loads((folder / "bundle.json").read_text())
        self.model = xgb.Booster()
        self.model.load_model(folder / "xgb_model.json")
        self.columns = self.meta["columns"]
        self.review = self.meta["thresholds"]["review"]
        self.block = self.meta["thresholds"]["block"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.inplace_predict(X[self.columns].to_numpy(dtype="float32"), missing=np.nan)

    def reasons(self, X: pd.DataFrame, top: int = 5) -> list[list[dict]]:
        """Per-transaction SHAP contributions: what pushed this score up, for the analyst."""
        d = xgb.DMatrix(X[self.columns].to_numpy(dtype="float32"), feature_names=self.columns, missing=np.nan)
        contrib = self.model.predict(d, pred_contribs=True)[:, :-1]
        out = []
        for row, values in zip(contrib, X[self.columns].to_numpy()):
            idx = np.argsort(-row)[:top]
            out.append([{"feature": self.columns[i], "value": None if np.isnan(values[i]) else round(float(values[i]), 3),
                         "impact": round(float(row[i]), 3)} for i in idx if row[i] > 0])
        return out


def split_without_repeats(events: list[dict]) -> list[list[dict]]:
    """Feast keeps only the latest row per entity, so two events for one user cannot share a read.
    "unknown" is the absence of a device, not a device: it has no row, so it never forces a split."""
    batches, cur, seen = [], [], set()
    for e in events:
        keys = {("u", e["user_id"]), ("c", e["card_id"])}
        if e["device_id"] != UNKNOWN:
            keys.add(("d", e["device_id"]))
        if keys & seen:
            batches.append(cur)
            cur, seen = [], set()
        cur.append(e)
        seen |= keys
    if cur:
        batches.append(cur)
    return batches


class ScoringPipeline:
    def __init__(self, engine_path: Path | None = None):
        self.store = fs.get_store()
        snapshot = path("live_dir") / "engine_state.pkl"
        chosen = engine_path or (snapshot if snapshot.exists() else path("artifact_dir") / "engine_state.pkl")
        self.engine = FeatureEngine.load(chosen)
        self.bundle = Bundle(fetch_bundle(), champion_version())
        self.vocab = self.bundle.meta["vocab"]
        self.recent: OrderedDict[str, dict] = OrderedDict()     # redelivered events must not count twice
        log.info("ready: model v%s, %d users in state", self.bundle.version, len(self.engine.users))

    def maybe_reload(self) -> bool:
        version = champion_version()
        if version == self.bundle.version:
            return False
        self.bundle = Bundle(fetch_bundle(), version)
        self.vocab = self.bundle.meta["vocab"]
        log.info("hot-reloaded model v%s", version)
        return True

    def snapshot(self) -> None:
        tmp = path("live_dir") / "engine_state.tmp"
        self.engine.save(tmp)
        os.replace(tmp, path("live_dir") / "engine_state.pkl")

    def score(self, events: list[dict]) -> list[dict]:
        out: dict[str, dict] = {}
        fresh = []
        for e in events:
            tid = e["transaction_id"]
            if tid in self.recent:
                out[tid] = self.recent[tid]
            else:
                fresh.append(e)
        for chunk in split_without_repeats(fresh):
            for r in self._score_chunk(chunk):
                out[r["transaction_id"]] = r
                self.recent[r["transaction_id"]] = r
        while len(self.recent) > 50_000:
            self.recent.popitem(last=False)
        return [out[e["transaction_id"]] for e in events]

    def _score_chunk(self, events: list[dict]) -> list[dict]:
        t0 = time.perf_counter()
        ev = pd.DataFrame(events)
        ev["event_ts"] = pd.to_datetime(ev["event_ts"], utc=True, format="ISO8601")
        rows = to_rows(self.engine, ev)
        written_at = pd.Timestamp.now(tz="UTC")     # online expiry runs on wall-clock time, replay data is old
        for view, table in rows.items():
            fs.write_online(self.store, view, table.assign(event_timestamp=written_at))
        keys = ev[["user_id", "card_id", "device_id"]].to_dict("records")
        values = fs.online(self.store, keys)
        X = F.build_frame(ev, values, self.vocab)
        xgb_p = self.bundle.predict(X)
        gnn_p = F.graph_score(values)
        final = F.blend(xgb_p, gnn_p, self.bundle.meta["stacker"])
        flagged = final >= self.bundle.review
        reasons = iter(self.bundle.reasons(X[flagged])) if flagged.any() else iter(())
        ms = (time.perf_counter() - t0) * 1000 / len(events)
        results = []
        for i, e in enumerate(events):
            decision = "BLOCK" if final[i] >= self.bundle.block else "REVIEW" if flagged[i] else "ALLOW"
            results.append({
                "transaction_id": e["transaction_id"], "event_ts": ev["event_ts"].iloc[i].isoformat(),
                "user_id": e["user_id"], "card_id": e["card_id"], "device_id": e["device_id"],
                "location_id": e["location_id"], "amount": float(e["amount"]),
                "score": float(final[i]), "xgb_score": float(xgb_p[i]), "gnn_score": float(gnn_p[i]),
                "decision": decision, "model_version": self.bundle.version,
                "reasons": next(reasons) if flagged[i] else [], "latency_ms": round(ms, 2),
                "_features": X.iloc[i],
            })
        return results
