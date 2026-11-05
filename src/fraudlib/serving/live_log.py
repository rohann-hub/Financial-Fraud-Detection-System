"""Append scored traffic to parquet files for drift checks and retraining."""
from __future__ import annotations

import time

import pandas as pd

from fraudlib.config import path


class LiveLog:
    def __init__(self, flush_rows: int = 500):
        self.flush_rows, self.buffer = flush_rows, []

    def add(self, event: dict, result: dict) -> None:
        features = result["_features"].rename(lambda c: f"feat_{c}").to_dict()
        keep = {k: result[k] for k in ("score", "xgb_score", "gnn_score", "decision", "model_version")}
        self.buffer.append({**event, **keep, **features})
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        name = path("live_dir") / f"scored-{time.time_ns()}.parquet"
        pd.DataFrame(self.buffer).to_parquet(name, index=False)
        self.buffer = []
