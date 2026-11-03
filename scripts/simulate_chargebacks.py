"""Stand-in for a bank's chargeback feed: reveals the true label of already-scored replay transactions."""
import pandas as pd

from fraudlib.config import path
from fraudlib.monitor.labels import live_events

live = live_events()
truth = pd.read_parquet(path("processed_dir") / "replay.parquet", columns=["transaction_id", "is_fraud"])
out = truth[truth["transaction_id"].isin(live["transaction_id"])]
out.to_parquet(path("live_dir") / "chargebacks.parquet", index=False)
print(f"{len(out):,} chargeback labels written, {int(out.is_fraud.sum())} fraud")
