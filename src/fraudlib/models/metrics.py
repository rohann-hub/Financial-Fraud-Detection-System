"""Evaluation for rare-event detection. Accuracy is excluded on purpose: predicting "never fraud"
scores about 96.5% accuracy on IEEE-CIS while catching nothing."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


MIN_BLOCK_SUPPORT = 30


def evaluate(y: np.ndarray, p: np.ndarray, amount: np.ndarray | None = None, prefix: str = "") -> dict:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    k = max(1, int(len(y) * .01))
    top = order[:k]
    prec, rec, _ = precision_recall_curve(y, p)
    out = {
        "pr_auc": average_precision_score(y, p),
        "pr_auc_baseline": float(y.mean()),   # a random model scores the fraud rate
        "roc_auc": roc_auc_score(y, p),
        "precision_at_1pct": float(y[top].mean()),
        "recall_at_1pct": float(y[top].sum() / max(1, y.sum())),
        "recall_at_precision_90": float(rec[prec >= .9].max()) if (prec >= .9).any() else 0.0,
    }
    if amount is not None:
        amount = np.asarray(amount)
        out["amount_recall_at_1pct"] = float((amount[top] * y[top]).sum() / max(1e-9, (amount * y).sum()))
    return {f"{prefix}{k}": float(v) for k, v in out.items()}


def thresholds(y: np.ndarray, p: np.ndarray, alert_rate: float, block_precision: float) -> dict:
    """Pick the two operating points on validation data only.

    block:  scores from here up keep precision at or above `block_precision` all the way to the top.
    review: analysts see the band just below block, sized to `alert_rate` of all traffic. Blocked
            transactions need no queue slot, so capacity goes on the cases that are actually uncertain.
    """
    y, p = np.asarray(y), np.asarray(p)
    prec, _, thr = precision_recall_curve(y, p)
    under = np.where(prec[:-1] < block_precision)[0]
    start = (under[-1] + 1) if len(under) else 0
    block = float(thr[start]) if start < len(thr) else 1.01   # 1.01 = precision unreachable, never auto-block
    if (p >= block).sum() < MIN_BLOCK_SUPPORT:                # one lucky row is not evidence
        block = 1.01
    below = np.sort(p[p < block])[::-1]
    k = max(1, int(len(p) * alert_rate))
    review = float(below[min(k, len(below)) - 1]) if len(below) else block
    return {"review": review, "block": block}


def decisions(y: np.ndarray, p: np.ndarray, th: dict) -> dict:
    """What the two thresholds do on a given period."""
    y, p = np.asarray(y), np.asarray(p)
    block, review = p >= th["block"], (p >= th["review"]) & (p < th["block"])
    share = lambda m: float(y[m].mean()) if m.any() else None      # noqa: E731
    return {"block_rate": float(block.mean()), "block_precision": share(block),
            "review_rate": float(review.mean()), "review_precision": share(review),
            "fraud_caught_share": float(y[block | review].sum() / max(1, y.sum()))}
