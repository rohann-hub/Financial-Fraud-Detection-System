"""Training pipeline. Each step reads and writes files, so any step can be re-run alone.

    python -m fraudlib.pipeline all --raw data/raw/ieee
    python -m fraudlib.pipeline prepare|eda|features|xgb|gnn|register|retrain|refresh-graph
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression

from fraudlib.config import ROOT, params, path
from fraudlib.data.ieee import load_ieee, split_mask, time_split
from fraudlib.features import store as fs
from fraudlib.features.engine import FeatureEngine, to_rows
from fraudlib.models import frame as F
from fraudlib.models.metrics import decisions, evaluate, thresholds


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mlflow() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'data/mlflow.db').as_posix()}"))
    mlflow.set_experiment(params()["mlflow"]["experiment"])


def _load() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    df = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    cuts = json.loads((path("processed_dir") / "split.json").read_text())
    return df, pd.Timestamp(cuts["t1"]), pd.Timestamp(cuts["t2"])


# ---------------------------------------------------------------- step 1
def _write_split(df: pd.DataFrame, t1: pd.Timestamp, t2: pd.Timestamp, raw: str, history_end: pd.Timestamp) -> None:
    df.to_parquet(path("processed_dir") / "transactions.parquet", index=False)
    (path("processed_dir") / "split.json").write_text(json.dumps(
        {"t1": str(t1), "t2": str(t2), "raw": str(raw), "history_end": str(history_end)}))


def prepare(raw: Path) -> None:
    df = load_ieee(raw)
    s = params()["split"]
    t1, t2 = time_split(df, s["train_frac"], s["val_frac"])
    _write_split(df, t1, t2, str(raw), t2)
    # The test period is what gets replayed. It is kept apart so retraining can never read a row that
    # has not been replayed yet, and so its true labels stay out of reach except to the simulators.
    df[df["event_ts"] >= t2].to_parquet(path("processed_dir") / "replay.parquet", index=False)
    _log(f"{len(df):,} transactions, fraud rate {df.is_fraud.mean():.3%}, train<{t1}, val<{t2}")


def roll_forward(extra: pd.DataFrame) -> None:
    """Build the retraining data: everything before the replay began, plus live rows with a confirmed label.

    The live window is cut into three by time, because a model only improves if it trains on new traffic:
      first third   joins training
      second third  validation (blend weights and thresholds)
      last third    the promotion gate, which neither champion nor challenger has trained on
    """
    df, _, _ = _load()
    cuts = json.loads((path("processed_dir") / "split.json").read_text())
    history_end = pd.Timestamp(cuts["history_end"])
    base = df[df["event_ts"] < history_end]
    both = pd.concat([base, extra[base.columns]], ignore_index=True).sort_values(["event_ts", "transaction_id"], ignore_index=True)
    k = len(extra) // 3
    t1, t2 = both["event_ts"].iloc[-2 * k], both["event_ts"].iloc[-k]
    _write_split(both, t1, t2, cuts["raw"], history_end)
    _log(f"rolled forward: {len(base):,} history + {len(extra):,} labelled live rows, train<{t1}, val<{t2}")


# ---------------------------------------------------------------- step 2
def features() -> None:
    df, _, t2 = _load()
    before = df["event_ts"] < t2
    engine = FeatureEngine()
    first = to_rows(engine, df[before])
    engine.save(path("artifact_dir") / "engine_state.pkl")   # live scoring resumes from exactly here
    second = to_rows(engine, df[~before])
    for view in first:
        pd.concat([first[view], second[view]], ignore_index=True).to_parquet(path("feature_dir") / f"{view}.parquet", index=False)
    # Graph tables must exist before Feast can read them; `gnn` overwrites these.
    for name, key in (("user_graph", "user_id"), ("card_graph", "card_id"), ("device_graph", "device_id")):
        p = path("feature_dir") / f"{name}.parquet"
        if not p.exists():
            pd.DataFrame({key: pd.Series(dtype=str), "event_timestamp": pd.to_datetime([], utc=True),
                          f"{name.split('_')[0]}_graph_risk": pd.Series(dtype="float32")}).to_parquet(p, index=False)
    store = fs.get_store()
    fs.apply(store)
    t = time.time()
    tset = fs.historical(store, df, fs.BEHAVIOUR_REFS)
    tset.to_parquet(path("feature_dir") / "training_set.parquet", index=False)
    _log(f"point-in-time join of {len(tset):,} rows took {time.time() - t:.0f}s")


# ---------------------------------------------------------------- step 3
def _xy() -> dict:
    df, t1, t2 = _load()
    fv = pd.read_parquet(path("feature_dir") / "training_set.parquet")
    assert (fv["transaction_id"].values == df["transaction_id"].values).all()
    split = split_mask(df, t1, t2)
    vocab = F.fit_vocab(df[split == "train"])
    return {"df": df, "fv": fv, "X": F.build_frame(df, fv, vocab), "y": df["is_fraud"].to_numpy(), "split": split, "vocab": vocab}


def _smote(X: pd.DataFrame, y: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    from imblearn.over_sampling import SMOTE
    # SMOTE interpolates between neighbours, so it needs complete data. It runs on training rows only:
    # resampling before the split would put synthetic copies of test fraud into training.
    filled = X.fillna(X.median()).fillna(0)
    return SMOTE(sampling_strategy=cfg["smote_sampling_strategy"], k_neighbors=cfg["smote_k_neighbors"],
                 random_state=7).fit_resample(filled, y)


def save_best_trees(model: xgb.XGBClassifier, target: Path) -> None:
    """Keep only the trees early stopping chose. XGBClassifier stops at best_iteration on its own, but the
    raw Booster used in serving would run every tree, so the served model would not be the evaluated one."""
    model.get_booster()[: model.best_iteration + 1].save_model(target)


def train_xgb() -> None:
    _mlflow()
    d = _xy()
    X, y, split = d["X"], d["y"], d["split"]
    cfg, imb = params()["xgboost"], params()["imbalance"]
    tr = np.where(split == "train")[0]
    cut = int(len(tr) * .85)                 # last 15% of train stops boosting; val stays untouched
    fit_i, stop_i = tr[:cut], tr[cut:]
    amount = d["df"]["amount"].to_numpy()
    best = (-1.0, None, None)
    for strategy in imb["strategies"]:
        with mlflow.start_run(run_name=f"xgb_{strategy}"):
            Xf, yf = X.iloc[fit_i], y[fit_i]
            spw = 1.0
            if strategy == "class_weight":
                spw = float((yf == 0).sum() / max(1, (yf == 1).sum()))
            elif strategy == "smote":
                Xf, yf = _smote(Xf, yf, imb)
            model = xgb.XGBClassifier(**cfg, tree_method="hist", eval_metric="aucpr", scale_pos_weight=spw,
                                      n_jobs=os.cpu_count(), random_state=7)
            t = time.time()
            model.fit(Xf, yf, eval_set=[(X.iloc[stop_i], y[stop_i])], verbose=False)
            metrics = {"train_seconds": time.time() - t, "best_iteration": model.best_iteration}
            for name in ("val", "test"):
                i = split == name
                metrics |= evaluate(y[i], model.predict_proba(X[i])[:, 1], amount[i], f"{name}_")
            mlflow.log_params({**cfg, "strategy": strategy, "scale_pos_weight": spw, "rows": len(yf)})
            mlflow.log_metrics(metrics)
            _log(f"xgb {strategy:13s} val PR-AUC {metrics['val_pr_auc']:.4f}  test {metrics['test_pr_auc']:.4f}  ({metrics['train_seconds']:.0f}s)")
            if metrics["val_pr_auc"] > best[0]:      # chosen on val; test is only reported
                best = (metrics["val_pr_auc"], strategy, model)
    save_best_trees(best[2], path("artifact_dir") / "xgb_model.json")
    (path("artifact_dir") / "vocab.json").write_text(json.dumps(d["vocab"]))
    (path("artifact_dir") / "xgb_choice.json").write_text(json.dumps({"strategy": best[1], "val_pr_auc": best[0]}))
    _log(f"selected strategy: {best[1]}")


# ---------------------------------------------------------------- step 4
def _graph_cutoffs(start: pd.Timestamp, end: pd.Timestamp, must_have: pd.Timestamp, days: int) -> list[pd.Timestamp]:
    """Dates the batch graph job would have run: every `days`, anchored on the serving cut."""
    step = pd.Timedelta(days=days)
    n_back, n_fwd = int((must_have - start) / step), int((end - must_have) / step)
    return sorted({start, *(must_have + i * step for i in range(-n_back, n_fwd + 1))})


def train_gnn(serve_last: bool = False) -> None:
    from fraudlib.models import gnn as G

    _mlflow()
    df, t1, t2 = _load()
    cfg = params()["gnn"]
    pre = df[df["event_ts"] < t1]
    delay, horizon = cfg["label_delay_days"], pd.Timedelta(days=cfg["horizon_days"])

    def snap(cut: pd.Timestamp, with_target: bool = True) -> G.Snapshot:
        future = df[df["event_ts"].between(cut, cut + horizon, inclusive="left")] if with_target else None
        return G.build_snapshot(df[df["event_ts"] < cut], cut, delay, future=future)

    with mlflow.start_run(run_name="gnn"):
        # Target: "does this node show fraud in the next horizon_days?" The last fit cut's target must end
        # before t1, otherwise fit labels would come from the validation period.
        fit_cuts = [pre["event_ts"].iloc[int(len(pre) * f)] for f in cfg["fit_cut_fracs"]]
        fit_cuts = [c for c in fit_cuts if c + horizon <= t1]
        fits, val = [snap(c) for c in fit_cuts], snap(t1)
        t = time.time()
        model, curve = G.train(fits, val, cfg)
        nodes = {k: val.data[k].num_nodes for k in G.NODE_KEYS}
        mlflow.log_params({**cfg, **{f"nodes_{k}": v for k, v in nodes.items()}})
        mlflow.log_metrics({"val_node_pr_auc": max(curve), "epochs": len(curve), "train_seconds": time.time() - t})
        _log(f"gnn nodes {nodes}, {len(fits)} fit cuts, best val node PR-AUC {max(curve):.4f} in {len(curve)} epochs ({time.time() - t:.0f}s)")

        # One row per node per refresh date. Feast's point-in-time join then gives every transaction the
        # newest graph built BEFORE it, which is exactly what a scheduled refresh would have served.
        end = df["event_ts"].max() + pd.Timedelta(milliseconds=1)
        serve_cut = end if serve_last else t2
        cuts = _graph_cutoffs(t1, end, serve_cut, cfg["refresh_days"])
        parts: dict[str, list] = {t: [] for t in G.NODE_KEYS}
        for cut in cuts:
            risk = G.node_risk(model, snap(cut, with_target=False))
            for ntype, key in G.NODE_KEYS.items():
                parts[ntype].append(pd.DataFrame({key: risk[ntype].index, "event_timestamp": cut,
                                                  f"{ntype}_graph_risk": risk[ntype].to_numpy()}))
        for ntype in G.NODE_KEYS:
            pd.concat(parts[ntype], ignore_index=True).to_parquet(path("feature_dir") / f"{ntype}_graph.parquet", index=False)
        (path("artifact_dir") / "graph_cut.json").write_text(json.dumps({"serve_cut": str(serve_cut), "refreshes": len(cuts)}))
        import torch
        torch.save(model.state_dict(), path("artifact_dir") / "gnn.pt")
        mlflow.log_artifact(str(path("artifact_dir") / "gnn.pt"))


def refresh_graph() -> None:
    """The scheduled graph job: rescore the graph with live traffic and the labels known today."""
    import torch

    from fraudlib.models import gnn as G
    from fraudlib.monitor.labels import known_label_events

    df, _, _ = _load()
    history_end = pd.Timestamp(json.loads((path("processed_dir") / "split.json").read_text())["history_end"])
    base = df[df["event_ts"] < history_end]
    live = known_label_events()
    rows = pd.concat([base, live[base.columns]], ignore_index=True) if len(live) else base
    rows = rows.drop_duplicates("transaction_id", keep="last").sort_values("event_ts", ignore_index=True)
    cut = rows["event_ts"].max() + pd.Timedelta(milliseconds=1)
    cfg = params()["gnn"]
    snap = G.build_snapshot(rows, cut, cfg["label_delay_days"])
    model = G.FraudGNN(snap.data["user"].x.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    model.load_state_dict(torch.load(path("artifact_dir") / "gnn.pt"))
    risk = G.node_risk(model, snap)
    store, now = fs.get_store(), pd.Timestamp.now(tz="UTC")
    for ntype, key in G.NODE_KEYS.items():
        fs.write_online(store, f"{ntype}_graph", pd.DataFrame(
            {key: risk[ntype].index, "event_timestamp": now, f"{ntype}_graph_risk": risk[ntype].to_numpy()}))
    _log(f"graph refreshed from {len(rows):,} transactions, cut {cut}")


# ---------------------------------------------------------------- step 4b
def _graph_scores(df: pd.DataFrame) -> pd.DataFrame:
    return fs.historical(fs.get_store(), df, fs.GRAPH_REFS)


def _champion_scores(bundle_dir: Path, d: dict, mask: np.ndarray, gnn_p: np.ndarray) -> np.ndarray:
    """Score with the champion's own vocabulary: category codes shift whenever training data changes."""
    meta = json.loads((bundle_dir / "bundle.json").read_text())
    m = xgb.XGBClassifier()
    m.load_model(bundle_dir / "xgb_model.json")
    X = F.build_frame(d["df"][mask], d["fv"][mask], meta["vocab"])
    return F.blend(m.predict_proba(X[meta["columns"]])[:, 1], gnn_p[mask], meta["stacker"])


def blend_and_register(promote_gate: bool = False) -> None:
    _mlflow()
    d = _xy()
    df, X, y, split = d["df"], d["X"], d["y"], d["split"]
    amount = df["amount"].to_numpy()
    model = xgb.XGBClassifier()
    model.load_model(path("artifact_dir") / "xgb_model.json")
    xgb_p = model.predict_proba(X)[:, 1]
    mask = split != "train"
    gnn_p = np.zeros(len(df))
    gnn_p[mask] = F.graph_score(_graph_scores(df[mask]))
    v, te = split == "val", split == "test"

    lr = LogisticRegression(C=100.0, max_iter=2000).fit(F.blend_inputs(xgb_p[v], gnn_p[v]), y[v])
    stacker = dict(zip(("xgb_weight", "gnn_weight", "unseen_weight"), map(float, lr.coef_[0]))) | {"intercept": float(lr.intercept_[0])}
    p = F.blend(xgb_p, gnn_p, stacker)
    dcfg = params()["decision"]
    th = thresholds(y[v], p[v], dcfg["review_alert_rate"], dcfg["block_precision"])

    report = {"rows": {"train": int((split == "train").sum()), "val": int(v.sum()), "test": int(te.sum())},
              "xgboost": evaluate(y[te], xgb_p[te], amount[te]), "gnn": evaluate(y[te], gnn_p[te], amount[te]),
              "blend": evaluate(y[te], p[te], amount[te]), "stacker": stacker, "thresholds": th,
              "graph_coverage_on_test": float((gnn_p[te] > 0).mean()), "decision_on_test": decisions(y[te], p[te], th)}
    kinds_file = Path(json.loads((path("processed_dir") / "split.json").read_text())["raw"]) / "fraud_kinds.csv"
    if kinds_file.exists():      # synthetic data only: which planted fraud pattern did each model catch?
        kinds = pd.read_csv(kinds_file, dtype={"TransactionID": str}).set_index("TransactionID")["kind"]
        k = df["transaction_id"].map(kinds).to_numpy()
        n = max(1, int(te.sum() * dcfg["review_alert_rate"]))
        by = {}
        for label, score in (("xgboost", xgb_p), ("blend", p)):
            cut = np.sort(score[te])[-n]
            by[label] = {str(kind): float((score[te & (k == kind)] >= cut).mean()) for kind in ("takeover", "testing", "ring", "blend")}
        report["recall_by_pattern_at_review_budget"] = by

    bundle = Path(tempfile.mkdtemp())
    shutil.copy(path("artifact_dir") / "xgb_model.json", bundle)
    choice = json.loads((path("artifact_dir") / "xgb_choice.json").read_text())
    (bundle / "bundle.json").write_text(json.dumps({
        "columns": F.MODEL_COLUMNS, "vocab": d["vocab"], "stacker": stacker, "thresholds": th,
        "xgb_strategy": choice["strategy"], "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    ref = X[v].assign(score=p[v]).sample(min(params()["monitoring"]["reference_rows"], int(v.sum())), random_state=7)
    ref.to_parquet(bundle / "reference.parquet", index=False)
    (path("artifact_dir") / "evaluation.json").write_text(json.dumps(report, indent=2))
    for name in ("xgboost", "gnn", "blend"):
        r = report[name]
        _log(f"test {name:8s} PR-AUC {r['pr_auc']:.4f}  ROC {r['roc_auc']:.4f}  recall@1% {r['recall_at_1pct']:.3f}")

    cfg = params()["mlflow"]
    client = mlflow.MlflowClient()
    champion_pr = None
    if promote_gate:
        try:
            mv = client.get_model_version_by_alias(cfg["model_name"], "champion")
            old = Path(mlflow.artifacts.download_artifacts(mv.source))
            champion_pr = evaluate(y[te], _champion_scores(old, d, te, gnn_p))["pr_auc"]
        except MlflowException:
            champion_pr = None
    new_pr = report["blend"]["pr_auc"]
    promote = champion_pr is None or new_pr > champion_pr + params()["monitoring"]["promotion_margin"]
    with mlflow.start_run(run_name="blend") as run:
        mlflow.log_params({f"stacker_{k}": v for k, v in stacker.items()} | {f"threshold_{k}": v for k, v in th.items()})
        mlflow.log_metrics({f"{m}_test_{k}": v for m in ("xgboost", "gnn", "blend") for k, v in report[m].items()})
        if champion_pr is not None:
            mlflow.log_metric("champion_test_pr_auc", champion_pr)
        mlflow.log_artifacts(str(bundle), "bundle")
        mlflow.log_artifact(str(path("artifact_dir") / "evaluation.json"))
        source = f"{run.info.artifact_uri}/bundle"
    try:
        client.create_registered_model(cfg["model_name"])
    except MlflowException:
        pass
    mv = client.create_model_version(cfg["model_name"], source, run_id=run.info.run_id)
    if promote:
        client.set_registered_model_alias(cfg["model_name"], "champion", mv.version)
        store = fs.get_store()
        cut = pd.Timestamp(json.loads((path("artifact_dir") / "graph_cut.json").read_text())["serve_cut"])
        for ntype in ("user", "card", "device"):
            rows = pd.read_parquet(path("feature_dir") / f"{ntype}_graph.parquet")
            fs.write_online(store, f"{ntype}_graph", rows[rows["event_timestamp"] == cut].assign(event_timestamp=pd.Timestamp.now(tz="UTC")))
    (path("artifact_dir") / "last_registration.json").write_text(json.dumps(
        {"version": mv.version, "promoted": promote, "challenger_pr_auc": new_pr, "champion_pr_auc": champion_pr}))
    _log(f"registered {cfg['model_name']} v{mv.version}; promoted={promote} (challenger {new_pr:.4f}, champion {champion_pr})")


def retrain() -> dict:
    """Fold scored live traffic and its confirmed labels into history, then rerun behind the promotion gate."""
    from fraudlib.monitor.labels import labelled_live_events

    extra = labelled_live_events()
    cfg = params()["monitoring"]
    if len(extra) < cfg["min_labelled_rows"] or extra["is_fraud"].sum() < cfg["min_labelled_fraud"]:
        msg = (f"Not enough confirmed labels: {len(extra)} rows, {int(extra['is_fraud'].sum()) if len(extra) else 0} fraud. "
               f"Need {cfg['min_labelled_rows']} rows and {cfg['min_labelled_fraud']} fraud")
        _log(msg)
        return {"skipped": msg}
    roll_forward(extra)
    features()
    train_xgb()
    train_gnn(serve_last=True)
    blend_and_register(promote_gate=True)
    return json.loads((path("artifact_dir") / "last_registration.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "prepare", "eda", "features", "xgb", "gnn", "register", "retrain", "refresh-graph"])
    ap.add_argument("--raw", type=Path, default=ROOT / params()["paths"]["raw_dir"])
    a = ap.parse_args()
    if a.step in ("all", "prepare"):
        prepare(a.raw)
    if a.step in ("all", "eda"):
        from fraudlib.data.eda import run as eda
        eda()
    if a.step in ("all", "features"):
        features()
    if a.step in ("all", "xgb"):
        train_xgb()
    if a.step in ("all", "gnn"):
        train_gnn()
    if a.step in ("all", "register"):
        blend_and_register()
    if a.step == "retrain":
        retrain()
    if a.step == "refresh-graph":
        refresh_graph()


if __name__ == "__main__":
    main()
