"""Turn the raw IEEE-CIS Kaggle files into one clean, time-ordered transaction table.

IEEE-CIS has no account column. We rebuild one the way the competition's top solutions did:
D1 is "days since this card was first seen", so (transaction day - D1) is constant for a customer.
    user_id = card fields + (day - D1)

Winning solutions also added addr1. We deliberately do not: addr1 is our location signal, and putting it
in the account key would make a "location change" impossible by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2017-12-01", tz="UTC")  # TransactionDT is an offset; this anchor is conventional

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in (1, 2, 3, 4, 5, 10, 11, 15)]
M_COLS = [f"M{i}" for i in (1, 2, 3, 5, 6, 7, 8, 9)]  # True/False flags (M4 is categorical, kept below)
# Vesta columns repeatedly reported as strong signals; a compact subset keeps memory reasonable.
V_COLS = [f"V{i}" for i in (12, 13, 29, 30, 48, 49, 53, 54, 69, 70, 74, 75, 90, 91, 94, 149, 156, 189,
                            200, 201, 244, 257, 258, 283, 294, 306, 308, 310, 312, 317)]
RAW_NUMERIC = ["dist1", *C_COLS, *D_COLS, *M_COLS, *V_COLS]
CATEGORICAL = ["product_code", "card_network", "card_kind", "device_type", "email_domain", "M4"]

_TXN_COLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
             "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
             "dist1", "P_emaildomain", "M4", *C_COLS, *D_COLS, *M_COLS, *V_COLS]
_ID_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]


def _read(path: Path, wanted: list[str]) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0)
    head.columns = [c.replace("id-", "id_") for c in head.columns]
    present = [c for c in wanted if c in head.columns]
    df = pd.read_csv(path, usecols=lambda c: c.replace("id-", "id_") in present)
    df.columns = [c.replace("id-", "id_") for c in df.columns]
    return df.reindex(columns=wanted)  # columns absent from the file become NaN


def _join_str(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = df[cols[0]].astype("string").fillna("na")
    for c in cols[1:]:
        out = out + "|" + df[c].astype("string").fillna("na")
    return out


def load_ieee(raw_dir: Path) -> pd.DataFrame:
    tx_path = Path(raw_dir) / "train_transaction.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"{tx_path} not found. Download IEEE-CIS from Kaggle, see README.")
    tx = _read(tx_path, _TXN_COLS)
    ident = _read(Path(raw_dir) / "train_identity.csv", _ID_COLS)
    df = tx.merge(ident, on="TransactionID", how="left")

    for c in ("card1", "card2", "card3", "card5", "addr1", "addr2"):
        df[c] = df[c].astype("Int64")  # avoids '1234.0' inside ids

    day = (df["TransactionDT"] // 86400).astype("int64")
    origin = (day - df["D1"]).astype("Int64").astype("string").fillna("na")
    out = pd.DataFrame({
        "transaction_id": df["TransactionID"].astype("int64").astype(str),
        # +1 ms per same-second row so every event has a unique, stable timestamp
        "event_ts": START + pd.to_timedelta(df["TransactionDT"], unit="s")
        + pd.to_timedelta(df.groupby("TransactionDT").cumcount(), unit="ms"),
        "is_fraud": df["isFraud"].astype("int8"),
        "amount": df["TransactionAmt"].astype("float64"),
        "user_id": _join_str(df, ["card1", "card2", "card3", "card5"]) + "|" + origin,
        "card_id": _join_str(df, ["card1", "card2", "card3", "card5"]),
        "location_id": _join_str(df, ["addr1", "addr2"]),
        "product_code": df["ProductCD"].astype("string"),
        "card_network": df["card4"].astype("string"),
        "card_kind": df["card6"].astype("string"),
        "device_type": df["DeviceType"].astype("string"),
        "email_domain": df["P_emaildomain"].astype("string"),
        "M4": df["M4"].astype("string"),
    })
    # A device is only known when an identity row exists (about a quarter of real IEEE traffic).
    has_device = df["DeviceInfo"].notna() | df["id_30"].notna() | df["id_33"].notna()
    out["device_id"] = _join_str(df, ["DeviceInfo", "id_30", "id_31", "id_33"]).where(has_device, "unknown")
    out.loc[df["addr1"].isna(), "location_id"] = "unknown"

    for c in M_COLS:
        out[c] = df[c].map({"T": 1.0, "F": 0.0})
    for c in ("dist1", *C_COLS, *D_COLS, *V_COLS):
        out[c] = df[c].astype("float32")

    str_cols = out.select_dtypes("string").columns
    out[str_cols] = out[str_cols].astype(object).where(out[str_cols].notna(), None)
    return out.sort_values(["event_ts", "transaction_id"], ignore_index=True)


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Cut points on the time axis (by row count, since traffic is uneven). Test = after the second cut."""
    n = len(df)
    return df["event_ts"].iloc[int(n * train_frac)], df["event_ts"].iloc[int(n * (train_frac + val_frac))]


def split_mask(df: pd.DataFrame, t1: pd.Timestamp, t2: pd.Timestamp) -> np.ndarray:
    return np.select([df["event_ts"] < t1, df["event_ts"] < t2], ["train", "val"], default="test")
