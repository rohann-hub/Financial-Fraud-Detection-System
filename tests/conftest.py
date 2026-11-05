import pandas as pd
import pytest


def txn(i, seconds, user="u1", card="c1", device="d1", location="l1", amount=100.0):
    return {"transaction_id": str(i), "ts": float(seconds), "user_id": user, "card_id": card,
            "device_id": device, "location_id": location, "amount": amount}


@pytest.fixture
def make_events():
    def build(rows):
        df = pd.DataFrame(rows)
        df["event_ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df
    return build
