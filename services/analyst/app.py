"""Analyst API: turns alerts into cases, tracks their status and records feedback labels."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from aiokafka import AIOKafkaConsumer
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("analyst")

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///data/analyst.db")
KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_ENABLED = os.environ.get("KAFKA_ENABLED", "true") == "true"
T_ALERTS = os.environ.get("TOPIC_ALERTS", "fraud.alerts")
SECRET = os.environ.get("SECRET_KEY", "dev-only-change-me").encode()
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "dev-internal-token")
USERS = dict(u.split(":", 1) for u in os.environ.get("ANALYST_USERS", "analyst:demo123,lead:demo123").split(","))
MONITORING = Path(os.environ.get("MONITORING_DIR", "data/monitoring"))
STATUSES = ("open", "investigating", "escalated", "closed")
LABELS = ("fraud", "legit")

if DB_URL.startswith("sqlite:///"):
    Path(DB_URL.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(DB_URL, pool_pre_ping=True)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True)
    event_ts: Mapped[str] = mapped_column(String(40), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    card_id: Mapped[str] = mapped_column(String(128), index=True)
    device_id: Mapped[str] = mapped_column(String(256), index=True)
    location_id: Mapped[str] = mapped_column(String(64))
    amount: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float, index=True)
    xgb_score: Mapped[float] = mapped_column(Float)
    gnn_score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(10), index=True)
    model_version: Mapped[str] = mapped_column(String(16))
    reasons: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    assignee: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    events: Mapped[list["CaseEvent"]] = relationship(back_populates="case", order_by="CaseEvent.id")


class CaseEvent(Base):
    """Append-only history of who did what to a case."""
    __tablename__ = "case_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    case: Mapped[Case] = relationship(back_populates="events")


def db():
    with Session(engine) as s:
        yield s


# ---------- auth: a signed, expiring token ----------
def _sign(payload: str) -> str:
    return hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_token(user: str) -> str:
    payload = f"{user}:{int(time.time()) + 12 * 3600}"
    return base64.urlsafe_b64encode(f"{payload}:{_sign(payload)}".encode()).decode()


def current_user(authorization: str = Header(default="")) -> str:
    try:
        raw = base64.urlsafe_b64decode(authorization.removeprefix("Bearer ").encode()).decode()
        user, exp, sig = raw.rsplit(":", 2)
    except Exception:
        raise HTTPException(401, "Sign in to continue") from None
    if not hmac.compare_digest(sig, _sign(f"{user}:{exp}")) or int(exp) < time.time():
        raise HTTPException(401, "Session expired. Sign in again")
    return user


# ---------- alert ingestion ----------
def store_alert(a: dict) -> bool:
    fields = ("transaction_id", "event_ts", "user_id", "card_id", "device_id", "location_id", "amount", "score",
              "xgb_score", "gnn_score", "decision", "model_version", "reasons")
    with Session(engine) as s:
        case = Case(**{k: a[k] for k in fields})
        case.events.append(CaseEvent(actor="system", action="created", detail=f"{a['decision']} at score {a['score']:.3f}"))
        s.add(case)
        try:
            s.commit()
            return True
        except IntegrityError:       # Kafka delivers at least once; the unique key makes this harmless
            s.rollback()
            return False


async def consume_alerts() -> None:
    while True:
        consumer = AIOKafkaConsumer(T_ALERTS, bootstrap_servers=KAFKA, group_id="analyst", auto_offset_reset="earliest")
        try:
            await consumer.start()
            async for m in consumer:
                await asyncio.to_thread(store_alert, json.loads(m.value))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("alert consumer stopped; reconnecting in 3s")
            await asyncio.sleep(3)
        finally:
            await consumer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for attempt in range(30):
        try:
            Base.metadata.create_all(engine)
            break
        except Exception as e:
            log.info("waiting for database: %s", e)
            await asyncio.sleep(2)
    task = asyncio.create_task(consume_alerts()) if KAFKA_ENABLED else None
    yield
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


app = FastAPI(title="Fraud analyst API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])


# ---------- API ----------
class Login(BaseModel):
    username: str
    password: str


class CasePatch(BaseModel):
    status: str | None = None
    assignee: str | None = None


class Feedback(BaseModel):
    label: str
    note: str | None = None


def row(c: Case) -> dict:
    return {k: getattr(c, k) for k in ("id", "transaction_id", "event_ts", "user_id", "card_id", "device_id", "location_id",
                                       "amount", "score", "xgb_score", "gnn_score", "decision", "model_version", "reasons",
                                       "status", "assignee", "label")} | {"created_at": c.created_at.isoformat()}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/login")
def login(body: Login) -> dict:
    expected = USERS.get(body.username)
    if expected is None or not hmac.compare_digest(expected, body.password):
        raise HTTPException(401, "Wrong username or password")
    return {"token": make_token(body.username), "user": body.username}


@app.get("/api/cases")
def list_cases(status: str | None = None, decision: str | None = None, q: str | None = None,
               limit: int = Query(50, le=200), offset: int = 0, s: Session = Depends(db), _: str = Depends(current_user)) -> dict:
    query = select(Case)
    if status:
        query = query.where(Case.status == status)
    if decision:
        query = query.where(Case.decision == decision)
    if q:
        like = f"%{q}%"
        query = query.where(or_(Case.transaction_id.like(like), Case.user_id.like(like), Case.card_id.like(like), Case.device_id.like(like)))
    total = s.scalar(select(func.count()).select_from(query.subquery()))
    items = s.scalars(query.order_by(Case.score.desc(), Case.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "items": [row(c) for c in items]}


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, s: Session = Depends(db), _: str = Depends(current_user)) -> dict:
    c = s.get(Case, case_id)
    if c is None:
        raise HTTPException(404, "Case not found")
    links = [Case.user_id == c.user_id, Case.card_id == c.card_id]
    if c.device_id != "unknown":
        links.append(Case.device_id == c.device_id)
    linked = s.scalars(select(Case).where(or_(*links), Case.id != c.id).order_by(Case.event_ts.desc()).limit(50)).all()
    return row(c) | {
        "history": [{"actor": e.actor, "action": e.action, "detail": e.detail, "at": e.at.isoformat()} for e in c.events],
        "linked": [row(x) | {"shared": [n for n, a in (("user", x.user_id == c.user_id), ("card", x.card_id == c.card_id),
                                                       ("device", x.device_id == c.device_id and c.device_id != "unknown")) if a]}
                   for x in linked]}


@app.patch("/api/cases/{case_id}")
def update_case(case_id: int, body: CasePatch, s: Session = Depends(db), user: str = Depends(current_user)) -> dict:
    c = s.get(Case, case_id)
    if c is None:
        raise HTTPException(404, "Case not found")
    if body.status:
        if body.status not in STATUSES:
            raise HTTPException(422, f"Status must be one of {STATUSES}")
        if body.status != c.status:
            c.events.append(CaseEvent(actor=user, action="status", detail=f"{c.status} to {body.status}"))
            c.status = body.status
    if body.assignee is not None and body.assignee != c.assignee:
        c.assignee = body.assignee or None
        c.events.append(CaseEvent(actor=user, action="assigned", detail=body.assignee or "unassigned"))
    s.commit()
    return get_case(case_id, s, user)


@app.post("/api/cases/{case_id}/feedback")
def feedback(case_id: int, body: Feedback, s: Session = Depends(db), user: str = Depends(current_user)) -> dict:
    if body.label not in LABELS:
        raise HTTPException(422, f"Label must be one of {LABELS}")
    c = s.get(Case, case_id)
    if c is None:
        raise HTTPException(404, "Case not found")
    c.label, c.status = body.label, "closed"
    c.assignee = c.assignee or user
    c.events.append(CaseEvent(actor=user, action="labelled", detail=body.label + (f": {body.note}" if body.note else "")))
    s.commit()
    return get_case(case_id, s, user)


@app.get("/api/summary")
def summary(s: Session = Depends(db), _: str = Depends(current_user)) -> dict:
    by_status = dict(s.execute(select(Case.status, func.count()).group_by(Case.status)).all())
    by_decision = dict(s.execute(select(Case.decision, func.count()).group_by(Case.decision)).all())
    by_label = dict(s.execute(select(Case.label, func.count()).where(Case.label.is_not(None)).group_by(Case.label)).all())
    reviewed = sum(by_label.values())
    by_day = s.execute(select(func.substr(Case.event_ts, 1, 10), Case.decision, func.count())
                       .group_by(func.substr(Case.event_ts, 1, 10), Case.decision)).all()
    days: dict[str, dict] = {}
    for day, decision, n in by_day:
        days.setdefault(day, {"day": day, "REVIEW": 0, "BLOCK": 0})[decision] = n
    return {"by_status": {k: by_status.get(k, 0) for k in STATUSES}, "by_decision": by_decision, "labels": by_label,
            "precision": by_label.get("fraud", 0) / reviewed if reviewed else None, "reviewed": reviewed,
            "per_day": [days[k] for k in sorted(days)], "drift": _drift()}


def _drift() -> dict:
    f = MONITORING / "drift_latest.json"
    return json.loads(f.read_text()) if f.exists() else {"status": "waiting", "message": "No drift check has run yet"}


@app.get("/api/drift/report")
def drift_report(token: str = Query("")) -> FileResponse:
    current_user(token)   # opened in a new tab, so the token travels in the query string
    f = MONITORING / "drift_report.html"
    if not f.exists():
        raise HTTPException(404, "No drift report yet")
    return FileResponse(f)


@app.get("/api/labels/export")
def export_labels(x_internal_token: str = Header(default=""), s: Session = Depends(db)) -> list[dict]:
    """Confirmed labels for the retraining job."""
    if not hmac.compare_digest(x_internal_token, INTERNAL_TOKEN):
        raise HTTPException(403, "Internal endpoint")
    rows = s.execute(select(Case.transaction_id, Case.label).where(Case.label.is_not(None))).all()
    return [{"transaction_id": t, "label": 1 if label == "fraud" else 0} for t, label in rows]
