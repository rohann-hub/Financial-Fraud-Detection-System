"""Exploratory analysis. Writes reports/eda.md and three charts."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudlib.config import ROOT, path

INK, MUTED, FRAUD, LEGIT = "#1f2933", "#6b7785", "#b4263a", "#2f6f8f"


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1d9")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e6eaee", linewidth=.8)
    ax.set_axisbelow(True)


def run() -> None:
    df = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    fraud = df["is_fraud"] == 1
    rate = fraud.mean()
    days = (df["event_ts"] - df["event_ts"].min()).dt.days

    fig, ax = plt.subplots(figsize=(7, 3))
    weekly = df.groupby(days // 7)["is_fraud"].mean() * 100
    ax.plot(weekly.index, weekly.values, color=FRAUD, linewidth=2)
    ax.axhline(rate * 100, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Week", color=MUTED, fontsize=9)
    ax.set_ylabel("Fraud rate, %", color=MUTED, fontsize=9)
    _style(ax, "Fraud rate moves over time, so splits must follow time")
    fig.tight_layout(); fig.savefig(out / "eda_fraud_rate_by_week.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.logspace(np.log10(max(df["amount"].min(), .25)), np.log10(df["amount"].max()), 45)
    ax.hist(df.loc[~fraud, "amount"], bins=bins, density=True, color=LEGIT, alpha=.75, label="Legitimate")
    ax.hist(df.loc[fraud, "amount"], bins=bins, density=True, color=FRAUD, alpha=.75, label="Fraud")
    ax.set_xscale("log")
    ax.set_xlabel("Amount (log scale)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Amounts overlap heavily: amount alone cannot separate fraud")
    fig.tight_layout(); fig.savefig(out / "eda_amount.png", dpi=130); plt.close(fig)

    known = df[df["device_id"] != "unknown"]
    per_dev = known.groupby("device_id").agg(users=("user_id", "nunique"), fraud=("is_fraud", "mean"), n=("is_fraud", "size"))
    per_dev["bucket"] = pd.cut(per_dev["users"], [0, 1, 2, 5, 20, 1e9], labels=["1", "2", "3-5", "6-20", "21+"])
    by = per_dev.groupby("bucket", observed=True).apply(lambda g: np.average(g["fraud"], weights=g["n"]), include_groups=False) * 100
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(by.index.astype(str), by.values, color=FRAUD, width=.55)
    ax.set_xlabel("Accounts on the device", color=MUTED, fontsize=9)
    ax.set_ylabel("Fraud rate, %", color=MUTED, fontsize=9)
    _style(ax, "Fraud rate by how many accounts share a device")
    fig.tight_layout(); fig.savefig(out / "eda_shared_devices.png", dpi=130); plt.close(fig)

    summary = {
        "rows": len(df), "fraud_rows": int(fraud.sum()), "fraud_rate": float(rate),
        "imbalance_ratio": float((~fraud).sum() / max(1, fraud.sum())),
        "accuracy_of_always_legit": float(1 - rate), "days": int(days.max()) + 1,
        "users": int(df["user_id"].nunique()), "cards": int(df["card_id"].nunique()),
        "devices": int(known["device_id"].nunique()), "device_known_share": float(len(known) / len(df)),
        "median_amount_fraud": float(df.loc[fraud, "amount"].median()),
        "median_amount_legit": float(df.loc[~fraud, "amount"].median()),
        "missing_share_top": df.isna().mean().sort_values(ascending=False).head(8).round(3).to_dict(),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    raw = json.loads((path("processed_dir") / "split.json").read_text())["raw"]
    source = "SYNTHETIC test data, not a real dataset" if "synthetic" in raw else raw
    (out / "eda.md").write_text(f"""# Exploratory data analysis

Data source: {source}

- {summary['rows']:,} transactions over {summary['days']} days; {summary['fraud_rows']:,} are fraud.
- **Fraud rate {rate:.2%}, or 1 fraud per {summary['imbalance_ratio']:.0f} legitimate transactions.**
- A model that always answers "legitimate" scores {summary['accuracy_of_always_legit']:.2%} accuracy and catches nothing.
  That is why this project evaluates with PR-AUC.
- {summary['users']:,} rebuilt accounts, {summary['cards']:,} cards, {summary['devices']:,} devices.
- A device is known for only {summary['device_known_share']:.1%} of transactions. Device features and graph edges cover that share.
- Median amount: fraud {summary['median_amount_fraud']:.2f}, legitimate {summary['median_amount_legit']:.2f}.

![Fraud rate by week](eda_fraud_rate_by_week.png)

![Amount distribution](eda_amount.png)

![Fraud by device sharing](eda_shared_devices.png)
""")
    print(f"EDA written to {out}; fraud rate {rate:.3%}")
