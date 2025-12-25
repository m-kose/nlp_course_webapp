from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artifacts_dir: Path
    events_jsonl: Path
    generations_jsonl: Path


def init_run_dir(run_dir: str | Path) -> RunPaths:
    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    _ensure_dir(run_dir)
    _ensure_dir(artifacts_dir)
    return RunPaths(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        events_jsonl=run_dir / "events.jsonl",
        generations_jsonl=run_dir / "generations.jsonl",
    )


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.paths = init_run_dir(run_dir)

    def log_event(self, event: str, **fields: Any) -> None:
        record = {"ts": _utc_now_iso(), "event": event, **fields}
        with self.paths.events_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def write_generation(self, record: Dict[str, Any]) -> None:
        with self.paths.generations_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def write_artifact_bytes(self, rel_path: str, data: bytes) -> Path:
        out_path = self.paths.artifacts_dir / rel_path
        _ensure_dir(out_path.parent)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            f.write(data)
        os.replace(tmp_path, out_path)
        return out_path

    def time_block(self, name: str, **fields: Any):
        return _TimeBlock(self, name=name, fields=fields)


class _TimeBlock:
    def __init__(self, logger: RunLogger, name: str, fields: Dict[str, Any]):
        self._logger = logger
        self._name = name
        self._fields = fields
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        self._logger.log_event(f"{self._name}.start", **self._fields)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_s = time.perf_counter() - self._start
        self._logger.log_event(
            f"{self._name}.end",
            elapsed_s=elapsed_s,
            ok=exc_type is None,
            error=(None if exc_type is None else f"{exc_type.__name__}: {exc}"),
            **self._fields,
        )
        return False

