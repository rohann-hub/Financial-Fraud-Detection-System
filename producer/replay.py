"""Replay the held-out test period into Kafka as if it were live traffic. Labels are never sent.

    python producer/replay.py --rate 200 --limit 20000
    python producer/replay.py --rate 500 --drift      # shift the data to demonstrate drift detection
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np
import pandas as pd
from aiokafka import AIOKafkaProducer

from fraudlib.config import path

TOPIC = os.environ.get("TOPIC_TRANSACTIONS", "fraud.transactions")


def load_events(limit: int | None, skip: int, drift: bool) -> pd.DataFrame:
    df = pd.read_parquet(path("processed_dir") / "replay.parquet").drop(columns="is_fraud").iloc[skip:]
    df = df.iloc[:limit] if limit else df
    if drift:   # a merchant-mix and spending shift: larger amounts, different hours, more of one product
        rng = np.random.default_rng(1)
        df = df.assign(amount=df["amount"] * 3.5, C1=df["C1"] + 6, D1=df["D1"] + 120,
                       product_code=np.where(rng.random(len(df)) < .6, "S", df["product_code"]))
    return df


def to_message(row: dict) -> bytes:
    row["event_ts"] = row["event_ts"].isoformat()
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
    return json.dumps(clean, allow_nan=False).encode()


async def replay(a: argparse.Namespace) -> None:
    df = load_events(a.limit, a.skip, a.drift)
    producer = AIOKafkaProducer(bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"), linger_ms=20)
    for attempt in range(30):
        try:
            await producer.start()
            break
        except Exception as e:
            print(f"waiting for Kafka ({e})")
            await asyncio.sleep(2)
    start, sent = time.perf_counter(), 0
    try:
        for row in df.to_dict("records"):
            await producer.send(TOPIC, key=row["user_id"].encode(), value=to_message(row))
            sent += 1
            if a.rate and sent % max(1, a.rate // 20) == 0:
                behind = sent / a.rate - (time.perf_counter() - start)
                if behind > 0:
                    await asyncio.sleep(behind)
            if sent % 5000 == 0:
                print(f"sent {sent:,}")
        await producer.flush()
    finally:
        await producer.stop()
    took = time.perf_counter() - start
    print(f"sent {sent:,} transactions in {took:.1f}s ({sent / took:.0f}/s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=int, default=200, help="transactions per second; 0 = as fast as possible")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--drift", action="store_true")
    asyncio.run(replay(ap.parse_args()))
