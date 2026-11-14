"""Minimal experiment log: one JSON line per training/analysis run.

Keeps parameters, metrics and the git commit together so results stay reproducible,
without needing an external tracking server.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "experiments" / "runs.jsonl"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def _plain(value):
    return value.item() if hasattr(value, "item") else str(value)


def log_run(name: str, params: dict, metrics: dict, path: Path = RUNS_PATH) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": name,
        "git_commit": _git_commit(),
        "params": params,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_plain) + "\n")
    return record


def read_runs(path: Path = RUNS_PATH, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:] if limit else rows
