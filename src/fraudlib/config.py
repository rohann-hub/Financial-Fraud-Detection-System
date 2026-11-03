"""Load configs/params.yaml. Paths resolve against the repo root (or FRAUD_HOME in containers)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("FRAUD_HOME", Path(__file__).resolve().parents[2]))


@lru_cache
def params() -> dict:
    return yaml.safe_load(Path(os.environ.get("FRAUD_PARAMS", ROOT / "configs/params.yaml")).read_text())


def path(name: str) -> Path:
    """Resolve a directory from the `paths` block, creating it if needed."""
    p = ROOT / params()["paths"][name]
    p.mkdir(parents=True, exist_ok=True)
    return p
