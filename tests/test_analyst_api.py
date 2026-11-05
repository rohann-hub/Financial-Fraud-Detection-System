"""Case workflow: alert in, feedback out, duplicates ignored."""
import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
    monkeypatch.setenv("KAFKA_ENABLED", "false")
    monkeypatch.setenv("MONITORING_DIR", str(tmp_path))
    sys.path.insert(0, "services/analyst")
    mod = importlib.import_module("app")
    importlib.reload(mod)
    with TestClient(mod.app) as c:
        token = c.post("/api/login", json={"username": "analyst", "password": "demo123"}).json()["token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c, mod
    sys.modules.pop("app", None)
    sys.path.remove("services/analyst")


def alert(i, **kw):
    return {"transaction_id": str(i), "event_ts": "2018-05-01T10:00:00+00:00", "user_id": "u1", "card_id": "c1",
            "device_id": "d1", "location_id": "l1", "amount": 99.5, "score": .9, "xgb_score": .8, "gnn_score": .7,
            "decision": "REVIEW", "model_version": "1", "reasons": [], **kw}


def test_requires_sign_in(api):
    c, _ = api
    assert c.get("/api/cases", headers={"Authorization": ""}).status_code == 401


def test_tampered_token_is_rejected(api):
    c, mod = api
    import base64
    forged = base64.urlsafe_b64encode(b"admin:9999999999:deadbeef").decode()
    assert c.get("/api/cases", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_redelivered_alert_creates_one_case(api):
    c, mod = api
    assert mod.store_alert(alert(1)) is True
    assert mod.store_alert(alert(1)) is False
    assert c.get("/api/cases").json()["total"] == 1


def test_feedback_closes_case_and_reaches_retraining_export(api):
    c, mod = api
    mod.store_alert(alert(2))
    cid = c.get("/api/cases").json()["items"][0]["id"]
    r = c.post(f"/api/cases/{cid}/feedback", json={"label": "fraud", "note": "called customer"}).json()
    assert r["status"] == "closed" and r["label"] == "fraud"
    assert any(h["action"] == "labelled" for h in r["history"])
    assert c.get("/api/labels/export", headers={"x-internal-token": "wrong"}).status_code == 403
    assert c.get("/api/labels/export", headers={"x-internal-token": "dev-internal-token"}).json() == [{"transaction_id": "2", "label": 1}]


def test_linked_cases_share_entities(api):
    c, mod = api
    mod.store_alert(alert(3, device_id="ring"))
    mod.store_alert(alert(4, user_id="u9", card_id="c9", device_id="ring"))
    mod.store_alert(alert(5, user_id="u8", card_id="c8", device_id="other"))
    items = {x["transaction_id"]: x for x in c.get("/api/cases").json()["items"]}
    linked = c.get(f"/api/cases/{items['3']['id']}").json()["linked"]
    assert [x["transaction_id"] for x in linked] == ["4"] and linked[0]["shared"] == ["device"]


def test_unknown_device_does_not_link_cases(api):
    c, mod = api
    mod.store_alert(alert(6, user_id="a", card_id="a", device_id="unknown"))
    mod.store_alert(alert(7, user_id="b", card_id="b", device_id="unknown"))
    cid = c.get("/api/cases", params={"q": "a"}).json()["items"][-1]["id"]
    assert c.get(f"/api/cases/{cid}").json()["linked"] == []


def test_invalid_status_rejected(api):
    c, mod = api
    mod.store_alert(alert(8))
    cid = c.get("/api/cases").json()["items"][0]["id"]
    assert c.patch(f"/api/cases/{cid}", json={"status": "banana"}).status_code == 422


def test_summary_counts(api):
    c, mod = api
    mod.store_alert(alert(9, decision="BLOCK"))
    s = c.get("/api/summary").json()
    assert s["by_status"]["open"] == 1 and s["by_decision"]["BLOCK"] == 1 and s["precision"] is None
    assert s["per_day"][0]["BLOCK"] == 1
