"""Demo helper: works through open cases the way an analyst would, using the true label.

    python scripts/simulate_analyst.py --cases 60 --api http://localhost:8001
"""
import argparse

import httpx
import pandas as pd

from fraudlib.config import path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cases", type=int, default=60)
ap.add_argument("--api", default="http://localhost:8001")
a = ap.parse_args()

truth = pd.read_parquet(path("processed_dir") / "replay.parquet", columns=["transaction_id", "is_fraud"]).set_index("transaction_id")["is_fraud"]
with httpx.Client(base_url=a.api, timeout=30) as c:
    token = c.post("/api/login", json={"username": "analyst", "password": "demo123"}).json()["token"]
    c.headers["Authorization"] = f"Bearer {token}"
    items = c.get("/api/cases", params={"status": "open", "limit": a.cases}).json()["items"]
    fraud = 0
    for item in items:
        label = "fraud" if truth[item["transaction_id"]] == 1 else "legit"
        fraud += label == "fraud"
        c.post(f"/api/cases/{item['id']}/feedback", json={"label": label, "note": "Simulated review"}).raise_for_status()
print(f"reviewed {len(items)} cases: {fraud} fraud, {len(items) - fraud} legitimate")
