"""Prove training and serving agree.

Takes the first N test transactions and scores them through the real live path: feature engine, Feast
online write, Feast online read, model. Then rebuilds the same transactions through the offline path:
Parquet tables, Feast point-in-time join, model. Every feature and every score must match.

    python scripts/verify_parity.py --rows 3000
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudlib.config import path
from fraudlib.features import store as fs
from fraudlib.models import frame as F
from fraudlib.serving.pipeline import ScoringPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=50, help="events per micro-batch, as the Kafka consumer sees them")
    a = ap.parse_args()

    df = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    t2 = pd.Timestamp(json.loads((path("processed_dir") / "split.json").read_text())["t2"])
    test = df[df["event_ts"] >= t2].head(a.rows).reset_index(drop=True)

    # always start from the training snapshot, whatever a live run has done since
    pipe = ScoringPipeline(engine_path=path("artifact_dir") / "engine_state.pkl")
    events = test.drop(columns="is_fraud").to_dict("records")
    for e in events:
        e["event_ts"] = e["event_ts"].isoformat()
        for k, v in e.items():
            if isinstance(v, float) and np.isnan(v):
                e[k] = None
    live = []
    for i in range(0, len(events), a.batch):
        live += pipe.score(events[i:i + a.batch])

    live_x = pd.DataFrame([r["_features"] for r in live]).reset_index(drop=True)
    offline_fv = fs.historical(fs.get_store(), test, fs.ALL_REFS)
    offline_x = F.build_frame(test, offline_fv, pipe.vocab).reset_index(drop=True)

    worst, bad = 0.0, {}
    for c in F.MODEL_COLUMNS:
        x, y = live_x[c].to_numpy(dtype="float64"), offline_x[c].to_numpy(dtype="float64")
        nan_mismatch = np.isnan(x) != np.isnan(y)
        diff = np.nan_to_num(np.abs(x - y), nan=0.0) / (1 + np.nan_to_num(np.abs(y)))
        if nan_mismatch.any() or diff.max() > 1e-6:
            bad[c] = {"nan_mismatch": int(nan_mismatch.sum()), "max_relative_diff": float(diff.max())}
        worst = max(worst, float(diff.max()))

    batch_xgb = pipe.bundle.predict(offline_x)
    batch_gnn = F.graph_score(offline_fv)
    batch = F.blend(batch_xgb, batch_gnn, pipe.bundle.meta["stacker"])
    online = np.array([r["score"] for r in live])
    # Training reports metrics through XGBClassifier, serving predicts through a raw Booster. Same trees?
    trained = xgb.XGBClassifier()
    trained.load_model(path("artifact_dir") / "xgb_model.json")
    api_gap = float(np.abs(trained.predict_proba(offline_x[pipe.bundle.columns])[:, 1] - batch_xgb).max())
    score_gap = float(np.abs(batch - online).max())
    report = {"rows": len(live), "columns_compared": len(F.MODEL_COLUMNS), "columns_mismatched": len(bad),
              "worst_relative_feature_diff": worst, "max_score_difference": score_gap,
              "training_vs_serving_predictor_gap": api_gap, "mismatches": bad}
    (path("artifact_dir") / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if bad or score_gap > 1e-4 or api_gap > 1e-5 else 0)


if __name__ == "__main__":
    main()
