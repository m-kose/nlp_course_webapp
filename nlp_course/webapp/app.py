from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
import csv
import re
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from nlp_course.run_logger import RunLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_PROMPTS_DIR = REPO_ROOT / "prompts"
UPLOADS_DIR = REPO_ROOT / "data" / "uploads"

_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.=") else "_" for c in s)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_int_list(val: Any) -> List[int]:
    if val is None:
        return []
    if isinstance(val, list):
        out: List[int] = []
        for x in val:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    if isinstance(val, str):
        out = []
        for part in val.replace(",", " ").split():
            try:
                out.append(int(part.strip()))
            except Exception:
                continue
        return out
    return []


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def _read_last_jsonl(path: Path, *, max_bytes: int = 250_000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with path.open("rb") as f:
        f.seek(start)
        data = f.read()
    lines = data.splitlines()
    out: List[Dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_first_jsonl(path: Path, *, max_lines: int = 500) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _read_csv_table(path: Path, *, max_rows: int = 500) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "columns": [], "rows": [], "truncated": False, "error": None}
    rows: List[Dict[str, Any]] = []
    truncated = False
    cols: List[str] = []
    error: Optional[str] = None
    try:
        # Some tables (e.g. per_generation.csv) can contain long text fields.
        csv.field_size_limit(10_000_000)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                if i >= max_rows:
                    truncated = True
                    break
                rows.append(dict(row))
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return {
        "path": str(path),
        "exists": True,
        "columns": cols,
        "rows": rows,
        "truncated": truncated,
        "error": error,
    }


def _tail_text_file(path: Path, *, offset: int, max_bytes: int = 200_000) -> Tuple[List[str], int]:
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    if not data:
        return [], offset
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset
    chunk = data[: last_nl + 1]
    new_offset = offset + len(chunk)
    text = chunk.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines, new_offset


def _resolve_dir(raw: str | None, *, default: Path) -> Path:
    if not raw:
        return default
    p = Path(raw)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _resolve_prompts_dir(raw: str | None, *, create: bool) -> Path:
    prompts_dir = _resolve_dir(raw, default=DEFAULT_PROMPTS_DIR).resolve()
    root = REPO_ROOT.resolve()
    try:
        prompts_dir.relative_to(root)
    except Exception:
        raise SystemExit(f"prompts_dir must be under repo root: {prompts_dir}")
    if create:
        _ensure_dir(prompts_dir)
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise SystemExit(f"Invalid prompts_dir: {prompts_dir}")
    return prompts_dir


def _normalize_prompt_id(raw: Any) -> str:
    prompt_id = str(raw or "").strip()
    if not prompt_id:
        raise SystemExit("prompt_id is required")
    if not _PROMPT_ID_RE.fullmatch(prompt_id):
        raise SystemExit("Invalid prompt_id. Use letters/numbers plus '_' or '-', e.g. my_prompt_v1")
    return prompt_id


def _resolve_file(raw: str | None) -> Optional[Path]:
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _make_run_dir(*, run_name: str) -> Path:
    _ensure_dir(RUNS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{_safe_name(run_name) or 'run'}"
    out = RUNS_DIR / base
    if not out.exists():
        return out
    out = RUNS_DIR / f"{base}_{uuid.uuid4().hex[:8]}"
    return out


@dataclass
class _ProcHandle:
    proc: subprocess.Popen
    stdout_f: Any
    stderr_f: Any


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: Dict[str, _ProcHandle] = {}

    def start(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dataset = str(cfg.get("dataset") or "jsonl")
        allowed = [s.strip() for s in (os.environ.get("WEBAPP_ALLOWED_DATASETS") or "xsum").split(",") if s.strip()]
        if dataset not in allowed:
            raise SystemExit(f"Dataset '{dataset}' is disabled on this server. Allowed: {', '.join(allowed)}")
        run_name = str(cfg.get("run_name") or f"{dataset}")
        run_dir = _make_run_dir(run_name=run_name)

        docs_path = _resolve_file(cfg.get("docs"))
        prompts_dir = _resolve_dir(cfg.get("prompts_dir"), default=DEFAULT_PROMPTS_DIR)

        if not prompts_dir.exists() or not prompts_dir.is_dir():
            raise SystemExit(f"Invalid prompts_dir: {prompts_dir}")

        models = cfg.get("models") or []
        if not isinstance(models, list) or not models:
            raise SystemExit("At least one model is required.")

        seeds = _parse_int_list(cfg.get("seeds") or [1])
        if not seeds:
            seeds = [1]

        prompt_ids = cfg.get("prompt_ids")
        if prompt_ids is not None and not isinstance(prompt_ids, list):
            raise SystemExit("prompt_ids must be a list of prompt ids (strings).")

        run_dir_rel = run_dir.relative_to(REPO_ROOT)
        stdout_path = run_dir / "web_stdout.log"
        stderr_path = run_dir / "web_stderr.log"
        _ensure_dir(run_dir)

        cmd: List[str] = [
            "python",
            "scripts/run_ollama_matrix.py",
            "--run-dir",
            str(run_dir_rel),
            "--dataset",
            dataset,
            "--prompts-dir",
            str(prompts_dir.relative_to(REPO_ROOT) if prompts_dir.is_relative_to(REPO_ROOT) else prompts_dir),
            "--models",
            *[str(m) for m in models],
            "--seeds",
            *[str(s) for s in seeds],
        ]

        if dataset == "jsonl":
            if docs_path is None:
                raise SystemExit("--docs is required for dataset=jsonl")
            cmd += ["--docs", str(docs_path)]
        else:
            if docs_path is not None:
                cmd += ["--docs", str(docs_path)]

        split = cfg.get("split")
        if split:
            cmd += ["--split", str(split)]

        hf_cache_dir = cfg.get("hf_cache_dir")
        if hf_cache_dir:
            cmd += ["--hf-cache-dir", str(hf_cache_dir)]

        if bool(cfg.get("shuffle")):
            cmd += ["--shuffle"]
        if cfg.get("shuffle_seed") is not None:
            cmd += ["--shuffle-seed", str(int(cfg.get("shuffle_seed")))]

        if prompt_ids:
            cmd += ["--prompt-ids", *[str(p) for p in prompt_ids]]

        for k in ["top_logprobs", "temperature", "top_p", "num_ctx", "num_predict", "max_docs", "status_every"]:
            if cfg.get(k) is not None and cfg.get(k) != "":
                cmd += [f"--{k.replace('_', '-')}", str(cfg.get(k))]

        embed_model = cfg.get("embed_model")
        if embed_model:
            cmd += ["--embed-model", str(embed_model)]

        ollama_url = cfg.get("ollama_url")
        if ollama_url:
            cmd += ["--ollama-url", str(ollama_url)]
        if cfg.get("connect_timeout_s") is not None:
            cmd += ["--connect-timeout-s", str(float(cfg.get("connect_timeout_s")))]
        if cfg.get("read_timeout_s") is not None:
            cmd += ["--read-timeout-s", str(float(cfg.get("read_timeout_s")))]

        if bool(cfg.get("no_preflight")):
            cmd += ["--no-preflight"]

        web_meta = {
            "ts_utc": _utc_now_iso(),
            "run_dir": str(run_dir_rel),
            "cfg": cfg,
            "cmd": cmd,
        }
        (run_dir / "web_run.json").write_text(json.dumps(web_meta, indent=2), encoding="utf-8")

        stdout_f = stdout_path.open("ab", buffering=0)
        stderr_f = stderr_path.open("ab", buffering=0)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)

        # Log a small breadcrumb into the run's events.jsonl.
        RunLogger(run_dir).log_event("web.run.started", pid=proc.pid, cmd=cmd)

        run_id = run_dir.name
        with self._lock:
            self._procs[run_id] = _ProcHandle(proc=proc, stdout_f=stdout_f, stderr_f=stderr_f)

        return {"run_id": run_id, "run_dir": str(run_dir_rel), "pid": proc.pid}

    def stop(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            handle = self._procs.get(run_id)
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise SystemExit(f"Unknown run_id: {run_id}")

        RunLogger(run_dir).log_event("web.run.stop_requested")

        if handle is None:
            return {"ok": False, "message": "Run is not managed by this server (already finished or server restarted)."}

        proc = handle.proc
        if proc.poll() is not None:
            return {"ok": True, "message": "Run already finished.", "returncode": proc.returncode}

        proc.terminate()
        for _ in range(40):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()

        return {"ok": True, "message": "Stop signal sent.", "returncode": proc.poll()}

    def cleanup_finished(self) -> None:
        done: List[str] = []
        with self._lock:
            for run_id, handle in self._procs.items():
                if handle.proc.poll() is not None:
                    done.append(run_id)
            for run_id in done:
                handle = self._procs.pop(run_id)
                try:
                    handle.stdout_f.close()
                except Exception:
                    pass
                try:
                    handle.stderr_f.close()
                except Exception:
                    pass

    def proc_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            handle = self._procs.get(run_id)
        if handle is None:
            return None
        proc = handle.proc
        return {"pid": proc.pid, "running": proc.poll() is None, "returncode": proc.poll()}


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    mgr = RunManager()

    @app.before_request
    def _cleanup() -> None:
        mgr.cleanup_finished()

    @app.before_request
    def _maybe_require_token() -> Any:
        required = (os.environ.get("WEBAPP_TOKEN") or "").strip()
        if not required:
            return None

        if request.endpoint == "static" or request.path.startswith("/static/"):
            return None

        if request.path == "/auth":
            return None

        provided = ((request.headers.get("X-Auth-Token") or "").strip() or (request.args.get("token") or "").strip())

        if provided != required:
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "Unauthorized", "auth_required": True}), 401
            next_url = request.full_path or request.path
            if next_url.endswith("?"):
                next_url = next_url[:-1]
            parts = urlsplit(next_url)
            qs = [(k, v) for (k, v) in parse_qsl(parts.query, keep_blank_values=True) if k != "token"]
            next_url = urlunsplit(("", "", parts.path, urlencode(qs, doseq=True), parts.fragment))
            return redirect("/auth?" + urlencode({"next": next_url}), code=302)

        return None

    def _safe_next_url(raw: str | None) -> str:
        if not raw:
            return "/"
        raw = str(raw)
        if len(raw) > 2048:
            return "/"
        if raw.startswith("//") or "://" in raw:
            return "/"
        if not raw.startswith("/"):
            return "/"
        return raw

    @app.get("/auth")
    def auth_get():
        next_url = _safe_next_url(request.args.get("next"))
        return render_template("auth.html", next_url=next_url, error=None)

    @app.post("/auth")
    def auth_post():
        required = (os.environ.get("WEBAPP_TOKEN") or "").strip()
        if not required:
            return redirect("/", code=302)

        token = (request.form.get("token") or "").strip()
        next_url = _safe_next_url(request.form.get("next"))
        if token != required:
            return render_template("auth.html", next_url=next_url, error="Invalid token."), 401

        parts = urlsplit(next_url)
        qs = parse_qsl(parts.query, keep_blank_values=True)
        qs = [(k, v) for (k, v) in qs if k != "token"]
        qs.append(("token", token))
        dest = urlunsplit(("", "", parts.path, urlencode(qs, doseq=True), parts.fragment))
        return redirect(dest, code=302)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/runs/<run_id>")
    def run_view(run_id: str):
        return render_template("run.html", run_id=run_id)

    @app.get("/api/ollama/tags")
    def api_ollama_tags():
        base_url = str(request.args.get("ollama_url") or os.environ.get("OLLAMA_URL") or "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/tags"
        try:
            r = requests.get(url, timeout=(3.0, 10.0))
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}", "ollama_url": base_url, "models": []}), 502

        models = []
        for m in (payload.get("models") or []):
            if isinstance(m, dict) and "name" in m:
                models.append(str(m["name"]))
        models = sorted(set(models))
        return jsonify({"ok": True, "ollama_url": base_url, "models": models})

    @app.get("/api/prompts")
    def api_prompts():
        raw_dir = request.args.get("prompts_dir")
        try:
            prompts_dir = _resolve_prompts_dir(str(raw_dir) if raw_dir else None, create=False)
        except SystemExit as e:
            return jsonify({"ok": False, "error": str(e), "prompts": []}), 400
        prompts = sorted([p.stem for p in prompts_dir.glob("*.txt")])
        return jsonify({"ok": True, "prompts_dir": str(prompts_dir), "prompts": prompts})

    @app.get("/api/prompts/<prompt_id>")
    def api_prompt_get(prompt_id: str):
        raw_dir = request.args.get("prompts_dir")
        try:
            prompts_dir = _resolve_prompts_dir(str(raw_dir) if raw_dir else None, create=False)
            prompt_id = _normalize_prompt_id(prompt_id)
        except SystemExit as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        path = prompts_dir / f"{prompt_id}.txt"
        if not path.exists():
            return jsonify({"ok": False, "error": f"Prompt not found: {prompt_id}"}), 404
        return jsonify({"ok": True, "prompt_id": prompt_id, "content": path.read_text(encoding="utf-8")})

    @app.post("/api/prompts")
    def api_prompt_save():
        payload = request.get_json(force=True, silent=False) or {}
        raw_dir = payload.get("prompts_dir")
        try:
            prompts_dir = _resolve_prompts_dir(str(raw_dir) if raw_dir else None, create=True)
            prompt_id = _normalize_prompt_id(payload.get("prompt_id"))
        except SystemExit as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        content = str(payload.get("content") or "")
        out_path = prompts_dir / f"{prompt_id}.txt"
        out_path.write_text(content, encoding="utf-8")
        return jsonify({"ok": True, "prompt_id": prompt_id})

    @app.post("/api/runs")
    def api_runs_start():
        cfg = request.get_json(force=True, silent=False) or {}
        try:
            out = mgr.start(cfg)
        except SystemExit as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
        return jsonify({"ok": True, **out})

    @app.post("/api/uploads")
    def api_uploads():
        _ensure_dir(UPLOADS_DIR)
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "Missing form field: file"}), 400
        f = request.files["file"]
        if not f or not getattr(f, "filename", ""):
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        name = secure_filename(f.filename) or "docs.jsonl"
        out_name = f"{uuid.uuid4().hex[:12]}_{_safe_name(name)}"
        out_path = UPLOADS_DIR / out_name
        f.save(out_path)
        return jsonify({"ok": True, "path": str(out_path.relative_to(REPO_ROOT))})

    @app.get("/api/runs")
    def api_runs_list():
        _ensure_dir(RUNS_DIR)
        items = []
        for p in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not p.is_dir():
                continue
            run_id = p.name
            items.append(_get_run_status(run_id=run_id, mgr=mgr))
        return jsonify({"ok": True, "runs": items})

    @app.get("/api/runs/<run_id>/status")
    def api_run_status(run_id: str):
        status = _get_run_status(run_id=run_id, mgr=mgr)
        return jsonify({"ok": True, "run": status})

    @app.post("/api/runs/<run_id>/stop")
    def api_run_stop(run_id: str):
        try:
            out = mgr.stop(run_id)
        except SystemExit as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
        return jsonify({"ok": True, **out})

    @app.get("/api/runs/<run_id>/tail")
    def api_run_tail(run_id: str):
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            abort(404)
        which = str(request.args.get("file") or "events")
        offset = int(request.args.get("offset") or 0)
        file_map = {
            "events": run_dir / "events.jsonl",
            "generations": run_dir / "generations.jsonl",
            "stdout": run_dir / "web_stdout.log",
            "stderr": run_dir / "web_stderr.log",
            "metrics_stdout": run_dir / "metrics_stdout.log",
            "metrics_stderr": run_dir / "metrics_stderr.log",
        }
        path = file_map.get(which)
        if path is None:
            return jsonify({"ok": False, "error": f"Unknown file: {which}"}), 400
        lines, new_offset = _tail_text_file(path, offset=offset)
        return jsonify({"ok": True, "file": which, "offset": new_offset, "lines": lines})

    @app.post("/api/runs/<run_id>/compute_metrics")
    def api_run_compute_metrics(run_id: str):
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            abort(404)

        payload = request.get_json(force=True, silent=True) or {}
        baseline_prompt = payload.get("baseline_prompt")
        tau = payload.get("tau")
        retrieval_k = payload.get("retrieval_k")
        metrics = payload.get("metrics")

        cmd: List[str] = ["python", "scripts/compute_metrics.py", "--run-dir", str(run_dir.relative_to(REPO_ROOT))]
        if baseline_prompt:
            cmd += ["--baseline-prompt", str(baseline_prompt)]
        if tau is not None and tau != "":
            cmd += ["--tau", str(float(tau))]
        if isinstance(retrieval_k, list) and retrieval_k:
            cmd += ["--retrieval-k", *[str(int(x)) for x in retrieval_k]]
        if metrics is not None:
            allowed = {"all", "reference", "logprobs", "embeddings", "bertscore", "bartscore"}
            if isinstance(metrics, list):
                vals = [str(m) for m in metrics if str(m) in allowed]
                cmd += ["--metrics", *vals]
            elif isinstance(metrics, str):
                vals = [m.strip() for m in metrics.split(",") if m.strip() in allowed]
                cmd += ["--metrics", *vals]
            else:
                cmd += ["--metrics"]

        RunLogger(run_dir).log_event("web.metrics.started", cmd=cmd)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        (run_dir / "metrics_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "metrics_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        RunLogger(run_dir).log_event("web.metrics.done", returncode=proc.returncode)
        if proc.returncode != 0:
            return (
                jsonify(
                    {
                        "ok": False,
                        "returncode": proc.returncode,
                        "stderr_tail": (proc.stderr or "")[-2000:],
                    }
                ),
                400,
            )
        return jsonify({"ok": True, "returncode": proc.returncode})

    @app.get("/api/runs/<run_id>/metrics/tables")
    def api_run_metrics_tables(run_id: str):
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            abort(404)
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.exists():
            return jsonify({"ok": True, "tables": {}, "available": [], "meta": None})

        meta_path = metrics_dir / "metrics_meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        table_files = [
            "by_model_prompt.csv",
            "stability_across_seeds.csv",
            "mi_prompt_from_summary.csv",
            "retrieval_alignment.csv",
            "genericity_index.csv",
            "prompt_reliance_index.csv",
        ]
        tables: Dict[str, Any] = {}
        available: List[str] = []
        for name in table_files:
            p = metrics_dir / name
            if p.exists():
                available.append(name)
            tables[name] = _read_csv_table(p, max_rows=500)

        # per_generation can be large; expose a small head by default.
        per_gen = metrics_dir / "per_generation.csv"
        if per_gen.exists():
            available.append("per_generation.csv")
        tables["per_generation.csv"] = _read_csv_table(per_gen, max_rows=200)

        return jsonify({"ok": True, "available": sorted(set(available)), "tables": tables, "meta": meta})

    @app.get("/runs/<run_id>/files/<path:rel_path>")
    def run_file(run_id: str, rel_path: str):
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            abort(404)
        # Constrain to the run directory.
        target = (run_dir / rel_path).resolve()
        if not str(target).startswith(str(run_dir.resolve()) + os.sep):
            abort(400)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_from_directory(run_dir, rel_path, as_attachment=True)

    @app.get("/runs/<run_id>/export/summaries.csv")
    def run_export_summaries_csv(run_id: str):
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            abort(404)
        gens_path = run_dir / "generations.jsonl"
        if not gens_path.exists():
            abort(404)

        def gen() -> Iterable[str]:
            header = [
                "ts_start_unix",
                "ts_end_unix",
                "model",
                "prompt_id",
                "doc_id",
                "seed",
                "summary",
                "reference",
                "error",
            ]
            buf = StringIO()
            w = csv.writer(buf)
            w.writerow(header)
            yield buf.getvalue()

            with gens_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    row = [
                        rec.get("ts_start_unix"),
                        rec.get("ts_end_unix"),
                        rec.get("model"),
                        rec.get("prompt_id"),
                        rec.get("doc_id"),
                        rec.get("seed"),
                        rec.get("summary"),
                        rec.get("reference"),
                        rec.get("error"),
                    ]
                    buf = StringIO()
                    w = csv.writer(buf)
                    w.writerow(row)
                    yield buf.getvalue()

        headers = {"Content-Disposition": f'attachment; filename="summaries_{run_id}.csv"'}
        return Response(gen(), mimetype="text/csv; charset=utf-8", headers=headers)

    return app


def _get_run_status(*, run_id: str, mgr: RunManager) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    status: Dict[str, Any] = {"run_id": run_id, "run_dir": str(run_dir.relative_to(REPO_ROOT))}
    if not run_dir.exists():
        status["state"] = "missing"
        return status

    proc = mgr.proc_status(run_id)
    status["proc"] = proc

    events_path = run_dir / "events.jsonl"
    gens_path = run_dir / "generations.jsonl"
    web_cfg_path = run_dir / "web_run.json"

    if web_cfg_path.exists():
        try:
            status["web"] = json.loads(web_cfg_path.read_text(encoding="utf-8"))
        except Exception:
            status["web"] = None

    status["generations_count"] = _count_lines(gens_path)

    total_calls = None
    last_cell = None
    done = False
    if events_path.exists():
        events = _read_last_jsonl(events_path)
        for ev in reversed(events):
            if total_calls is None and ev.get("event") == "run.matrix":
                total_calls = ev.get("total_calls")
            if last_cell is None and ev.get("event") == "ollama.generate.start":
                last_cell = {
                    "model": ev.get("model"),
                    "prompt_id": ev.get("prompt_id"),
                    "doc_id": ev.get("doc_id"),
                    "seed": ev.get("seed"),
                }
            if ev.get("event") == "run.done":
                done = True
            if total_calls is not None and last_cell is not None and done:
                break

        # For large runs, the tail may not include the early run.matrix line; scan the head (cheap).
        if total_calls is None:
            head = _read_first_jsonl(events_path, max_lines=500)
            for ev in head:
                if ev.get("event") == "run.matrix":
                    total_calls = ev.get("total_calls")
                    break

    status["total_calls"] = total_calls
    status["last_cell"] = last_cell
    status["done_event"] = done

    if proc and proc.get("running"):
        status["state"] = "running"
    else:
        if done:
            status["state"] = "done"
        else:
            status["state"] = "idle"
    return status
