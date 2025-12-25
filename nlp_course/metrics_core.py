from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .ollama import top_logprobs_as_dict


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class LogprobMetrics:
    token_count: int
    sum_logprob: float
    mean_nll: float
    entropy_mean: float
    entropy_slope: float
    entropy_spike_count: int


def compute_logprob_metrics(token_entries: List[Dict[str, Any]]) -> Optional[LogprobMetrics]:
    if not token_entries:
        return None

    token_logprobs: List[float] = []
    entropies: List[float] = []

    for entry in token_entries:
        lp = entry.get("logprob")
        if lp is None:
            lp = entry.get("token_logprob")
        lp_f = safe_float(lp)
        if not math.isfinite(lp_f):
            continue
        token_logprobs.append(lp_f)

        top = top_logprobs_as_dict(entry)
        if top:
            probs = np.array([math.exp(v) for v in top.values()], dtype=np.float64)
            probs = np.clip(probs, 0.0, 1.0)
            s = float(probs.sum())
            p_other = max(0.0, 1.0 - s)
            if p_other > 0:
                probs = np.append(probs, p_other)
            probs = np.clip(probs, 1e-12, 1.0)
            probs = probs / float(probs.sum())
            ent = -float(np.sum(probs * np.log(probs)))
            entropies.append(ent)

    if not token_logprobs:
        return None

    token_count = len(token_logprobs)
    sum_logprob = float(np.sum(token_logprobs))
    mean_nll = float(-sum_logprob / token_count)

    if entropies:
        entropy_mean = float(np.mean(entropies))
        if len(entropies) >= 2:
            x = np.arange(len(entropies), dtype=np.float64)
            slope = float(np.polyfit(x, np.array(entropies, dtype=np.float64), deg=1)[0])
        else:
            slope = float("nan")
        std = float(np.std(entropies)) if len(entropies) >= 2 else 0.0
        spike_thr = entropy_mean + 2.0 * std
        spikes = int(np.sum(np.array(entropies) > spike_thr)) if std > 0 else 0
    else:
        entropy_mean, slope, spikes = float("nan"), float("nan"), 0

    return LogprobMetrics(
        token_count=token_count,
        sum_logprob=sum_logprob,
        mean_nll=mean_nll,
        entropy_mean=entropy_mean,
        entropy_slope=slope,
        entropy_spike_count=spikes,
    )


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_ci(values: np.ndarray, *, iters: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    n = len(values)
    means = []
    for _ in range(iters):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi

