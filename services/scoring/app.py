"""Scoring service: a FastAPI app with a Kafka consumer running inside it.

Consumes fraud.transactions, fetches features from Feast, scores with the champion model, and publishes
to fraud.scores (everything) and fraud.alerts (review or block). POST /score runs the same code on demand.

Run one replica. The sliding-window state lives in this process (see docs/DESIGN.md).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

import pandas as pd
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from fraudlib.models.frame import EVENT_COLUMNS
from fraudlib.serving.live_log import LiveLog
from fraudlib.serving.pipeline import ScoringPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("scoring")

KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_ENABLED = os.environ.get("KAFKA_ENABLED", "true") == "true"
T_IN = os.environ.get("TOPIC_TRANSACTIONS", "fraud.transactions")
T_SCORES = os.environ.get("TOPIC_SCORES", "fraud.scores")
T_ALERTS = os.environ.get("TOPIC_ALERTS", "fraud.alerts")
T_DEAD = os.environ.get("TOPIC_DEAD_LETTER", "fraud.dead-letter")
REQUIRED = ("transaction_id", "event_ts", "amount", "user_id", "card_id")


class Txn(BaseModel):
    """Minimum fields for a score. Any other event column may be included."""
    model_config = ConfigDict(extra="allow")
    transaction_id: str
    event_ts: str
    amount: float
    user_id: str
    card_id: str
    device_id: str = "unknown"
    location_id: str = "unknown"


def normalise(e: dict) -> dict:
    """Reject anything that would crash the scorer, so one bad message cannot stall the partition."""
    if not isinstance(e, dict):
        raise ValueError("event must be a JSON object")
    missing = [k for k in REQUIRED if e.get(k) is None]
    if missing:
        raise ValueError(f"missing {missing}")
    amount = float(e["amount"])
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"bad amount {e['amount']!r}")
    e["amount"] = amount
    ts = pd.Timestamp(e["event_ts"])            # raises ValueError on rubbish
    e["event_ts"] = (ts if ts.tzinfo else ts.tz_localize("UTC")).isoformat()
    for k in ("transaction_id", "user_id", "card_id"):
        e[k] = str(e[k])
    e.setdefault("device_id", "unknown")
    e.setdefault("location_id", "unknown")
    for c in EVENT_COLUMNS:
        e.setdefault(c, None)
    return e


class State:
    pipeline: ScoringPipeline
    live = LiveLog()
    lock = asyncio.Lock()               # feature state is updated by one scorer at a time
    counts: Counter = Counter()
    latency = deque(maxlen=2000)
    started = time.time()


S = State()


async def _score(events: list[dict]) -> list[dict]:
    async with S.lock:
        results = await asyncio.to_thread(S.pipeline.score, events)
        for e, r in zip(events, results):
            S.live.add(e, r)
            S.counts["scored"] += 1
            S.counts[r["decision"].lower()] += 1
            S.latency.append(r["latency_ms"])
    return results


def _public(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


async def consume() -> None:
    while True:
        consumer = AIOKafkaConsumer(T_IN, bootstrap_servers=KAFKA, group_id="scoring", enable_auto_commit=False,
                                    auto_offset_reset="earliest")
        producer = AIOKafkaProducer(bootstrap_servers=KAFKA, linger_ms=5)
        try:
            await consumer.start()
            await producer.start()
            log.info("consuming %s", T_IN)
            while True:
                polled = await consumer.getmany(timeout_ms=200, max_records=200)
                messages = [m for part in polled.values() for m in part]
                if not messages:
                    continue
                events = []
                for m in messages:
                    try:
                        events.append(normalise(json.loads(m.value)))
                    except (ValueError, TypeError) as err:
                        S.counts["dead_letter"] += 1
                        await producer.send(T_DEAD, value=json.dumps({"error": str(err), "raw": m.value.decode("utf-8", "replace")}).encode())
                if events:
                    for r in await _score(events):
                        body = json.dumps(_public(r)).encode()
                        key = r["user_id"].encode()
                        await producer.send(T_SCORES, key=key, value=body)
                        if r["decision"] != "ALLOW":
                            await producer.send(T_ALERTS, key=key, value=body)
                    await producer.flush()
                await consumer.commit()      # only after results are published: at-least-once
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("consumer stopped; reconnecting in 3s")
            await asyncio.sleep(3)
        finally:
            await consumer.stop()
            await producer.stop()


async def housekeeping() -> None:
    """Check MLflow for a new champion, and persist feature state, every 30 seconds."""
    while True:
        await asyncio.sleep(float(os.environ.get("RELOAD_SECONDS", 30)))
        try:
            async with S.lock:
                await asyncio.to_thread(S.pipeline.maybe_reload)
                await asyncio.to_thread(S.live.flush)
                await asyncio.to_thread(S.pipeline.snapshot)
        except Exception:
            log.exception("housekeeping failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    S.pipeline = await asyncio.to_thread(ScoringPipeline)
    tasks = [asyncio.create_task(housekeeping())]
    if KAFKA_ENABLED:
        tasks.append(asyncio.create_task(consume()))
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    S.live.flush()


app = FastAPI(title="Fraud scoring service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": S.pipeline.bundle.version}


@app.post("/score")
async def score(txn: Txn) -> dict:
    try:
        return _public((await _score([normalise(txn.model_dump())]))[0])
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/stats")
def stats() -> dict:
    lat = sorted(S.latency)
    up = time.time() - S.started
    return {"model_version": S.pipeline.bundle.version, "uptime_seconds": round(up), **S.counts,
            "latency_ms_p50": lat[len(lat) // 2] if lat else None,
            "latency_ms_p95": lat[int(len(lat) * .95)] if lat else None}


@app.get("/metrics")
def metrics():
    from fastapi.responses import PlainTextResponse
    lines = [f'fraud_events_total{{outcome="{k}"}} {v}' for k, v in S.counts.items()]
    return PlainTextResponse("\n".join(lines) + "\n")
