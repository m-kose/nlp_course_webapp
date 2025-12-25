from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 300.0


TimeoutT = Union[float, Tuple[float, float]]


def _post_json(url: str, payload: Dict[str, Any], timeout: TimeoutT) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def generate(
    *,
    cfg: OllamaConfig,
    model: str,
    prompt: str,
    seed: Optional[int],
    options: Dict[str, Any],
    logprobs: bool = True,
    top_logprobs: int = 20,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {**options},
        "logprobs": bool(logprobs),
        "top_logprobs": int(top_logprobs),
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    return _post_json(
        f"{cfg.base_url}/api/generate",
        payload,
        (float(cfg.connect_timeout_s), float(cfg.read_timeout_s)),
    )


def embed(
    *,
    cfg: OllamaConfig,
    model: str,
    text: str,
) -> List[float]:
    payload: Dict[str, Any] = {"model": model, "input": text}
    out = _post_json(
        f"{cfg.base_url}/api/embed",
        payload,
        (float(cfg.connect_timeout_s), float(cfg.read_timeout_s)),
    )
    embeddings = out.get("embeddings")
    if not embeddings or not isinstance(embeddings, list) or not embeddings[0]:
        raise ValueError(f"Unexpected /api/embed response shape: keys={list(out.keys())}")
    if len(embeddings) != 1:
        raise ValueError(f"Expected single embedding vector, got {len(embeddings)}")
    return embeddings[0]


def extract_token_logprobs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Best-effort parser for Ollama 'logprobs' output.

    We intentionally keep this permissive because Ollama's response schema can
    vary by version/model. Metric computation should handle '[]' gracefully.
    """
    logprobs = payload.get("logprobs")
    if logprobs is None:
        return []
    if isinstance(logprobs, list):
        return [x for x in logprobs if isinstance(x, dict)]
    if isinstance(logprobs, dict):
        tokens = logprobs.get("tokens") or logprobs.get("token_logprobs") or logprobs.get("logprobs")
        if isinstance(tokens, list):
            return [x for x in tokens if isinstance(x, dict)]
    return []


def top_logprobs_as_dict(entry: Dict[str, Any]) -> Dict[str, float]:
    top = entry.get("top_logprobs")
    if top is None:
        return {}
    if isinstance(top, dict):
        out: Dict[str, float] = {}
        for k, v in top.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    if isinstance(top, list):
        out = {}
        for item in top:
            if not isinstance(item, dict):
                continue
            tok = item.get("token") or item.get("text")
            lp = item.get("logprob")
            if tok is None or lp is None:
                continue
            try:
                out[str(tok)] = float(lp)
            except Exception:
                continue
        return out
    return {}
