"""After a replay, compare what the live system decided with the true labels."""
import json

import numpy as np
import pandas as pd

from fraudlib.config import path
from fraudlib.models.metrics import evaluate
from fraudlib.monitor.labels import live_events

live = live_events()
truth = pd.read_parquet(path("processed_dir") / "replay.parquet", columns=["transaction_id", "is_fraud"])
m = live.merge(truth, on="transaction_id")
flagged = m["decision"] != "ALLOW"
report = {"rows": len(m), "fraud_rows": int(m.is_fraud.sum()), **evaluate(m["is_fraud"], m["score"], m["amount"]),
          "alert_rate": float(flagged.mean()), "alert_precision": float(m.loc[flagged, "is_fraud"].mean()) if flagged.any() else None,
          "alert_recall": float(m.loc[flagged, "is_fraud"].sum() / max(1, m.is_fraud.sum())),
          "decisions": m["decision"].value_counts().to_dict()}
(path("artifact_dir") / "live_results.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
