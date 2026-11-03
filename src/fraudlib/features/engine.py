"""The one place behavioural features are computed.

`FeatureEngine.process(event)` is called once per transaction, in time order, by BOTH:
  - the batch pipeline, to build the Feast offline tables used for training
  - the live scoring service, to update Feast's online store
So training and serving cannot disagree: there is no second implementation to drift.

Each row describes the entity *including the current transaction*, which is known at decision time.
No feature reads a label, so this is not leakage.
"""
from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HOUR, DAY, WEEK = 3600.0, 86400.0, 7 * 86400.0
UNKNOWN = "unknown"

USER_FEATURES = [
    "user_txn_count_1h", "user_txn_count_24h", "user_txn_count_7d",
    "user_amount_sum_1h", "user_amount_sum_24h", "user_txn_count_total",
    "user_amount_mean", "user_amount_std", "user_amount_vs_avg", "user_amount_zscore",
    "user_secs_since_last", "user_device_changed", "user_new_device", "user_location_changed",
    "user_new_location", "user_distinct_devices_7d", "user_distinct_locations_7d",
]
CARD_FEATURES = ["card_txn_count_1h", "card_txn_count_24h", "card_amount_sum_24h",
                 "card_distinct_users_7d", "card_distinct_devices_7d"]
DEVICE_FEATURES = ["device_txn_count_1h", "device_txn_count_24h", "device_distinct_users_7d",
                   "device_distinct_cards_7d", "device_user_count_total"]


class Window:
    """Sliding count and sum over a fixed span of event time."""
    __slots__ = ("span", "items", "total")

    def __init__(self, span: float):
        self.span, self.items, self.total = span, deque(), 0.0

    def add(self, ts: float, amount: float) -> None:
        self.items.append((ts, amount))
        self.total += amount
        cutoff = ts - self.span
        while self.items and self.items[0][0] <= cutoff:
            self.total -= self.items.popleft()[1]


def _distinct(seen: dict, key: str, ts: float) -> int:
    """Distinct keys in the last 7 days; `seen` maps key to last-seen time."""
    seen[key] = ts
    stale = [k for k, t in seen.items() if t <= ts - WEEK]
    for k in stale:
        del seen[k]
    return len(seen)


@dataclass
class UserState:
    h1: Window = field(default_factory=lambda: Window(HOUR))
    d1: Window = field(default_factory=lambda: Window(DAY))
    w1: Window = field(default_factory=lambda: Window(WEEK))
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0            # Welford running variance
    last_ts: float | None = None
    last_device: str | None = None
    last_location: str | None = None
    devices: set = field(default_factory=set)
    locations: set = field(default_factory=set)
    recent_devices: dict = field(default_factory=dict)
    recent_locations: dict = field(default_factory=dict)


@dataclass
class CardState:
    h1: Window = field(default_factory=lambda: Window(HOUR))
    d1: Window = field(default_factory=lambda: Window(DAY))
    users: dict = field(default_factory=dict)
    devices: dict = field(default_factory=dict)


@dataclass
class DeviceState:
    h1: Window = field(default_factory=lambda: Window(HOUR))
    d1: Window = field(default_factory=lambda: Window(DAY))
    users: dict = field(default_factory=dict)
    cards: dict = field(default_factory=dict)
    all_users: set = field(default_factory=set)


class FeatureEngine:
    def __init__(self):
        self.users: dict[str, UserState] = {}
        self.cards: dict[str, CardState] = {}
        self.devices: dict[str, DeviceState] = {}

    def process(self, ts: float, user_id: str, card_id: str, device_id: str, location_id: str,
                amount: float) -> tuple[dict, dict, dict | None]:
        """Update state with one transaction; return (user_row, card_row, device_row)."""
        has_device, has_location = device_id != UNKNOWN, location_id != UNKNOWN

        # --- user: compare against history BEFORE folding this transaction in ---
        u = self.users.setdefault(user_id, UserState())
        std = (u.m2 / u.n) ** .5 if u.n > 1 else 0.0
        user = {
            "user_amount_mean": u.mean, "user_amount_std": std,
            "user_amount_vs_avg": amount / u.mean if u.n and u.mean > 0 else 1.0,
            "user_amount_zscore": (amount - u.mean) / std if std > 0 else 0.0,
            "user_secs_since_last": ts - u.last_ts if u.last_ts is not None else -1.0,
            "user_device_changed": float(bool(has_device and u.last_device and device_id != u.last_device)),
            "user_new_device": float(bool(has_device and u.devices and device_id not in u.devices)),
            "user_location_changed": float(bool(has_location and u.last_location and location_id != u.last_location)),
            "user_new_location": float(bool(has_location and u.locations and location_id not in u.locations)),
        }
        u.n += 1
        delta = amount - u.mean
        u.mean += delta / u.n
        u.m2 += delta * (amount - u.mean)
        u.last_ts = ts
        for w in (u.h1, u.d1, u.w1):
            w.add(ts, amount)
        if has_device:
            u.last_device = device_id
            u.devices.add(device_id)
            ndev = _distinct(u.recent_devices, device_id, ts)
        else:
            ndev = len(u.recent_devices)
        if has_location:
            u.last_location = location_id
            u.locations.add(location_id)
            nloc = _distinct(u.recent_locations, location_id, ts)
        else:
            nloc = len(u.recent_locations)
        user.update({
            "user_txn_count_1h": float(len(u.h1.items)), "user_txn_count_24h": float(len(u.d1.items)),
            "user_txn_count_7d": float(len(u.w1.items)), "user_amount_sum_1h": u.h1.total,
            "user_amount_sum_24h": u.d1.total, "user_txn_count_total": float(u.n),
            "user_distinct_devices_7d": float(ndev), "user_distinct_locations_7d": float(nloc),
        })

        # --- card ---
        c = self.cards.setdefault(card_id, CardState())
        c.h1.add(ts, amount)
        c.d1.add(ts, amount)
        ncu = _distinct(c.users, user_id, ts)
        ncd = _distinct(c.devices, device_id, ts) if has_device else len(c.devices)
        card = {"card_txn_count_1h": float(len(c.h1.items)), "card_txn_count_24h": float(len(c.d1.items)),
                "card_amount_sum_24h": c.d1.total, "card_distinct_users_7d": float(ncu),
                "card_distinct_devices_7d": float(ncd)}

        # --- device (only when one was observed) ---
        device = None
        if has_device:
            d = self.devices.setdefault(device_id, DeviceState())
            d.h1.add(ts, amount)
            d.d1.add(ts, amount)
            d.all_users.add(user_id)
            device = {"device_txn_count_1h": float(len(d.h1.items)), "device_txn_count_24h": float(len(d.d1.items)),
                      "device_distinct_users_7d": float(_distinct(d.users, user_id, ts)),
                      "device_distinct_cards_7d": float(_distinct(d.cards, card_id, ts)),
                      "device_user_count_total": float(len(d.all_users))}
        return user, card, device

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "FeatureEngine":
        with open(path, "rb") as f:
            return pickle.load(f)


def _frame(rows: list[dict], key: str, ids: list, times: list, columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns).astype("float32")
    df.insert(0, key, ids)
    df.insert(1, "event_timestamp", pd.to_datetime(times, utc=True))
    return df


def to_rows(engine: FeatureEngine, events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run events through the engine and return one Feast-ready table per entity."""
    users, cards, devices, dev_ids, dev_ts = [], [], [], [], []
    epoch = ((events["event_ts"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    cols = zip(epoch, events["user_id"], events["card_id"], events["device_id"], events["location_id"],
               events["amount"], events["event_ts"])
    for ts, uid, cid, did, loc, amt, stamp in cols:
        user, card, device = engine.process(float(ts), uid, cid, did, loc, float(amt))
        users.append(user)
        cards.append(card)
        if device is not None:
            devices.append(device)
            dev_ids.append(did)
            dev_ts.append(stamp)
    times = events["event_ts"].tolist()
    return {
        "user_features": _frame(users, "user_id", events["user_id"].tolist(), times, USER_FEATURES),
        "card_features": _frame(cards, "card_id", events["card_id"].tolist(), times, CARD_FEATURES),
        "device_features": _frame(devices, "device_id", dev_ids, dev_ts, DEVICE_FEATURES),
    }
