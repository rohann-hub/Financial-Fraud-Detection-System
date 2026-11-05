"""Generate a TEST dataset in the exact IEEE-CIS file format.

Purpose: run and test the whole platform without the Kaggle download. Results on it say nothing about
real-world performance. It writes the same CSV columns as Kaggle so the real loader is what gets tested.

Fraud patterns planted:
  takeover   existing customer, new device and location, fast burst of large payments
  testing    stolen card, burst of tiny amounts
  ring       sleeper accounts sharing a few devices and cards. Each transaction looks ordinary, and sharing
             alone is not suspicious here because legitimate customers share cards and devices too.
             What gives a ring away is a neighbour with confirmed fraud, which only the graph can see.
  blend      individually ordinary fraud with no planted signal (keeps PR-AUC below 1)

Because rings are built to be invisible to tabular features, any lift the GNN shows here is a designed
outcome. It proves the graph pipeline works. It is not evidence of how much a graph helps on real data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlib.data.ieee import C_COLS, D_COLS, M_COLS, V_COLS

DAYS = 180
EMAILS = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "icloud.com"])
NETWORKS = np.array(["visa", "mastercard", "american express", "discover"])
DEVICES = np.array(["iOS Device", "Windows", "MacOS", "SM-G960U", "SM-A205U", "Pixel 3", "Moto G"])


def _card_pool(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """IEEE-CIS card fields are coarse (issuer, BIN, type), so many customers share one card_id."""
    pool = pd.DataFrame({"card1": rng.integers(1000, 18000, n), "card2": rng.integers(100, 600, n),
                         "card5": rng.integers(100, 240, n)})
    weight = 1 / np.arange(1, n + 1) ** .8
    pool["p"] = weight / weight.sum()
    return pool


def _users(rng: np.random.Generator, n: int, pool: pd.DataFrame) -> pd.DataFrame:
    card = pool.iloc[rng.choice(len(pool), n, p=pool["p"].to_numpy())].reset_index(drop=True)
    return pd.DataFrame({
        "card1": card["card1"], "card2": card["card2"],
        "card3": 150, "card5": card["card5"],
        "card4": rng.choice(NETWORKS, n, p=[.65, .3, .03, .02]),
        "card6": rng.choice(["debit", "credit"], n, p=[.75, .25]),
        "addr1": rng.integers(100, 540, n), "addr2": 87,
        "email": rng.choice(EMAILS, n, p=[.45, .2, .15, .1, .05, .05]),
        # Most customers predate the data, and some join during it, so traffic stays roughly level and a
        # young account is not odd in itself. A negative day just means "joined before the data starts".
        "first_day": rng.integers(-300, DAYS - 20, n),
        "mean_amt": np.exp(rng.normal(3.9, .7, n)),
        "device": [f"{DEVICES[i]}_{j}" for j, i in enumerate(rng.integers(0, len(DEVICES), n))],
        "has_identity": rng.random(n) < .3,
        "activity": rng.pareto(1.3, n) + 1,
    })


def generate(n_txns: int = 200_000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_users = n_txns // 7
    pool = _card_pool(rng, max(50, n_users // 4))
    u = _users(rng, n_users, pool)
    # A few shared legitimate devices (family tablets, shop terminals): shared is not automatically fraud.
    shared = rng.choice(n_users, n_users // 12, replace=False)
    u.loc[shared, "device"] = [f"{DEVICES[i % len(DEVICES)]}_S{i}" for i in rng.integers(0, n_users // 80, len(shared))]
    u.loc[shared, "has_identity"] = True
    # Occasional customers on public machines (cafes, shared offices). In a single transaction they look just
    # like a ring member: few payments, a device several accounts use. Only the graph can tell them apart.
    public = rng.choice(np.setdiff1d(np.arange(n_users), shared), n_users // 15, replace=False)
    u.loc[public, "device"] = [f"{DEVICES[i % len(DEVICES)]}_P{i}" for i in rng.integers(0, n_users // 150, len(public))]
    u.loc[public, "has_identity"] = True
    u.loc[public, "activity"] = rng.uniform(1, 3, len(public))

    rows: list[tuple] = []  # (user_idx, day_seconds, amount, device, addr1, is_fraud, kind)

    def add(uid, t, amt, dev=None, addr=None, fraud=0, kind="legit"):
        rows.append((uid, t, amt, dev, addr, fraud, kind))

    # legitimate traffic
    p = (u["activity"] / u["activity"].sum()).to_numpy()
    who = rng.choice(n_users, int(n_txns * .965), p=p)
    active_from = np.clip(u["first_day"].to_numpy()[who], 0, None)
    t = (active_from + rng.random(len(who)) * (DAYS - active_from)) * 86400
    hour_shift = np.clip(rng.normal(14, 4.5, len(who)), 0, 23.9) * 3600
    t = (t // 86400) * 86400 + hour_shift
    amt = np.exp(np.log(u["mean_amt"].to_numpy()[who]) + rng.normal(0, .6, len(who)))
    roam = rng.random(len(who))
    for i in range(len(who)):
        dev = f"Travel_{rng.integers(1e6)}" if roam[i] < .02 else None
        addr = int(rng.integers(100, 540)) if roam[i] < .015 else None
        add(int(who[i]), float(t[i]), float(amt[i]), dev, addr)

    budget = n_txns - len(rows)
    # takeover: 25%
    spent = 0
    while spent < budget * .25:
        v, start = int(rng.integers(n_users)), rng.random() * (DAYS - 1) * 86400
        k, dev, addr = int(rng.integers(2, 7)), f"Fraud_{rng.integers(1e7)}", int(rng.integers(100, 540))
        for j in range(k):
            add(v, start + j * rng.integers(40, 600), float(u.mean_amt[v] * rng.uniform(2, 6)), dev, addr, 1, "takeover")
        spent += k
    # card testing: 10%
    spent = 0
    while spent < budget * .10:
        v, start, k = int(rng.integers(n_users)), rng.random() * (DAYS - 1) * 86400, int(rng.integers(3, 9))
        for j in range(k):
            add(v, start + j * rng.integers(20, 200), float(rng.uniform(.5, 4)), None, None, 1, "testing")
        spent += k
    # rings: 45%. Accounts opened weeks earlier, a few shared devices and cards, activity spread over weeks.
    spent, extra = 0, []
    while spent < budget * .45:
        size, born = int(rng.integers(6, 30)), rng.random() * (DAYS - 25) * 86400
        members = _users(rng, size, pool)
        members["first_day"] = int(born // 86400) - rng.integers(0, 300, size)    # sleeper accounts, not new ones
        shared = pool.iloc[rng.choice(len(pool), int(rng.integers(1, 4)), p=pool["p"].to_numpy())]
        pick = rng.integers(0, len(shared), size)
        for c in ("card1", "card2", "card5"):
            members[c] = shared[c].to_numpy()[pick]
        devs = [f"{DEVICES[rng.integers(len(DEVICES))]}_R{rng.integers(1e7)}" for _ in range(rng.integers(1, 4))]
        offset = n_users + sum(len(x) for x in extra)
        extra.append(members)
        for m in range(size):
            for _ in range(int(rng.integers(1, 4))):
                add(offset + m, born + rng.random() * 40 * 86400, float(members.mean_amt[m] * np.exp(rng.normal(0, .6))),
                    str(rng.choice(devs)), None, 1, "ring")
                spent += 1
    # blend: 20%
    for _ in range(int(budget * .20)):
        add(int(rng.integers(n_users)), rng.random() * DAYS * 86400, float(np.exp(rng.normal(4.1, .7))),
            None, None, 1, "blend")

    pop = pd.concat([u, *extra], ignore_index=True)
    ev = pd.DataFrame(rows, columns=["uid", "t", "amt", "dev", "addr", "isFraud", "kind"])
    ev = ev[ev["t"].between(0, DAYS * 86400)].sort_values("t", ignore_index=True)
    return _to_ieee(ev, pop, rng)


def _to_ieee(ev: pd.DataFrame, pop: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(ev)
    p = pop.iloc[ev["uid"].to_numpy()].reset_index(drop=True)
    dt = (ev["t"].to_numpy().astype("int64") + 86400)
    day = dt // 86400
    tx = pd.DataFrame({
        "TransactionID": np.arange(2987000, 2987000 + n), "isFraud": ev["isFraud"].to_numpy(),
        "TransactionDT": dt, "TransactionAmt": ev["amt"].to_numpy().round(3),
        "ProductCD": rng.choice(list("WCRHS"), n, p=[.74, .12, .07, .05, .02]),
        "card1": p["card1"], "card2": p["card2"], "card3": p["card3"], "card4": p["card4"],
        "card5": p["card5"], "card6": p["card6"],
        "addr1": ev["addr"].fillna(p["addr1"]).astype(int), "addr2": p["addr2"],
        "dist1": np.where(rng.random(n) < .6, np.nan, rng.exponential(60, n).round()),
        "P_emaildomain": p["email"], "R_emaildomain": None,
    })
    tx["card1"] = tx["card1"].astype(int)
    for c in C_COLS:
        tx[c] = rng.poisson(1.5, n).astype(float)
    for c in D_COLS:
        tx[c] = np.where(rng.random(n) < .4, np.nan, rng.integers(0, 300, n).astype(float))
    # D1 must stay consistent with the first-seen day: this is what lets the loader rebuild accounts.
    tx["D1"] = (day - (p["first_day"].to_numpy() + 1)).clip(0).astype(float)  # always >= 0: first_day precedes the event
    for c in M_COLS:
        tx[c] = rng.choice(["T", "F", None], n, p=[.6, .2, .2])
    tx["M4"] = rng.choice(["M0", "M1", "M2", None], n, p=[.3, .1, .1, .5])
    for c in V_COLS:
        tx[c] = rng.normal(1, .3, n).round(3)
    # A weak noisy signal on two V columns, as Vesta's real engineered columns provide.
    noisy = (ev["isFraud"].to_numpy() == 1) & (rng.random(n) < .3)
    tx.loc[noisy, "V258"] += rng.exponential(1.5, noisy.sum())
    tx.loc[noisy, "V294"] += rng.exponential(1.5, noisy.sum())

    device = ev["dev"].where(ev["dev"].notna(), p["device"])
    seen = (ev["dev"].notna() | p["has_identity"]).to_numpy()
    ident = pd.DataFrame({
        "TransactionID": tx["TransactionID"], "DeviceType": np.where(device.str.contains("Windows|MacOS"), "desktop", "mobile"),
        "DeviceInfo": device, "id_30": "Android 8.0.0", "id_31": "chrome 65.0", "id_33": "1920x1080",
    })[seen].reset_index(drop=True)
    tx["kind"] = ev["kind"]
    return tx, ident


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    tx, ident = generate(a.rows, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx.drop(columns="kind").to_csv(out / "train_transaction.csv", index=False)
    ident.to_csv(out / "train_identity.csv", index=False)
    tx[["TransactionID", "kind"]].to_csv(out / "fraud_kinds.csv", index=False)  # answer key for ring recall
    print(f"{len(tx):,} transactions, fraud rate {tx.isFraud.mean():.2%}, written to {out}")


if __name__ == "__main__":
    main()
