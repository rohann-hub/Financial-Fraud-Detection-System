"""Compare live traffic with the training reference and retrain automatically when it has moved.

    python -m fraudlib.monitor.drift            report only
    python -m fraudlib.monitor.drift --retrain  retrain if drift crosses the threshold
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from fraudlib.config import ROOT, params, path
from fraudlib.monitor.labels import live_events

# Columns analysts and auditors can reason about. Monitoring all 100+ would hide signal in noise.
MONITORED = ["amount", "log_amount", "hour", "product_code_code", "device_type_code", "user_amount_vs_avg",
             "user_txn_count_24h", "user_new_device", "user_new_location", "card_txn_count_24h",
             "card_distinct_users_7d", "device_distinct_users_7d", "C1", "D1", "score"]


def has_drifted(metric_name: str, value: float) -> bool:
    """Evidently picks the test per column and sample size. A p-value drifts when LOW; a distance such as
    Wasserstein, which it uses above 1,000 rows, drifts when HIGH. The metric name says which it chose."""
    threshold = float(metric_name.rsplit("threshold=", 1)[1].rstrip(")"))
    return value < threshold if "p_value" in metric_name else value >= threshold


def compare(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[float, list[str], object]:
    cols = [c for c in MONITORED if c in reference and c in current]
    ref, cur = reference[cols].astype("float64"), current[cols].astype("float64")
    definition = DataDefinition(numerical_columns=cols)
    result = Report([DataDriftPreset(drift_share=params()["monitoring"]["drift_share_threshold"])]).run(
        Dataset.from_pandas(cur, data_definition=definition), Dataset.from_pandas(ref, data_definition=definition))
    drifted, share = [], None
    for m in result.dict()["metrics"]:
        name, value = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            share = float(value["share"])
        elif name.startswith("ValueDrift") and has_drifted(name, float(value)):
            drifted.append(name.split("column=")[1].split(",")[0])
    return share, sorted(drifted), result


def current_reference() -> pd.DataFrame:
    """Reference rows from the champion's bundle, so drift is measured against what it was trained on."""
    from fraudlib.serving.pipeline import fetch_bundle
    return pd.read_parquet(fetch_bundle() / "reference.parquet")


def run(retrain: bool) -> dict:
    cfg = params()["monitoring"]
    mon = path("monitoring_dir")
    state_file = mon / "drift_latest.json"
    live = live_events()
    previous = json.loads(state_file.read_text()) if state_file.exists() else {}
    last = previous.get("last_retrain_ts", 0)
    summary = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": len(live),
               "threshold": cfg["drift_share_threshold"]}
    if len(live) < cfg["min_rows"]:
        summary |= {"status": "waiting", "message": f"Need {cfg['min_rows']} live rows, have {len(live)}", "last_retrain_ts": last}
        state_file.write_text(json.dumps(summary, indent=2))
        print(summary["message"])
        return summary

    # Live rows carry the raw event and the model's view of it (feat_*). Drift is about the model's view.
    newest = live.sort_values("event_ts").tail(cfg["current_rows"])
    recent = newest.filter(like="feat_").rename(columns=lambda c: c.removeprefix("feat_")).assign(score=newest["score"])
    share, drifted_columns, result = compare(current_reference(), recent)
    result.save_html(str(mon / "drift_report.html"))
    drifted = share >= cfg["drift_share_threshold"]
    cooling = time.time() - last < cfg["cooldown_hours"] * 3600
    summary |= {"status": "drift" if drifted else "stable", "drift_share": share,
                "drifted_columns": drifted_columns, "last_retrain_ts": last}
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'data/mlflow.db').as_posix()}"))
    mlflow.set_experiment(params()["mlflow"]["experiment"])
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metrics({"drift_share": share, "live_rows": len(live)})
        mlflow.log_artifact(str(mon / "drift_report.html"))

    if drifted and retrain and not cooling:
        summary["retrain"] = "started"
        state_file.write_text(json.dumps(summary, indent=2))
        from fraudlib.pipeline import retrain as do_retrain
        summary["retrain"] = do_retrain()
        if "skipped" not in summary["retrain"]:      # no labels yet: try again next check, not next day
            summary["last_retrain_ts"] = time.time()
    elif drifted:
        summary["retrain"] = "cooldown" if cooling else "not requested"
    state_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true")
    run(ap.parse_args().retrain)
