from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from nlp_course.ollama import OllamaConfig, embed, generate, extract_token_logprobs
from nlp_course.datasets import load_docs
from nlp_course.run_logger import RunLogger


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.=") else "_" for c in s)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for p in sorted(prompts_dir.glob("*.txt")):
        prompts[p.stem] = p.read_text(encoding="utf-8").strip()
    if not prompts:
        raise SystemExit(f"No prompts found in: {prompts_dir}")
    return prompts


def build_prompt(prompt_template: str, doc_text: str) -> str:
    return f"{prompt_template}\n\n---\nDOCUMENT:\n{doc_text}\n---\n\nSUMMARY:\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="jsonl",
        choices=["jsonl", "xsum", "cnn_dailymail"],
        help="Document source. Use cnn_dailymail to download via HF datasets; use xsum for the included XSum-format JSONL.",
    )
    ap.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="JSONL path (required for --dataset=jsonl; optional for --dataset=xsum which defaults to ./data/xsum_dataset.jsonl or $XSUM_JSONL).",
    )
    ap.add_argument("--split", type=str, default="validation", help="HF split when using --dataset=cnn_dailymail.")
    ap.add_argument("--hf-cache-dir", type=Path, default=None, help="Optional HF datasets cache dir.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle docs before selecting --max-docs.")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="Shuffle seed.")
    ap.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    ap.add_argument(
        "--prompt-ids",
        nargs="*",
        default=None,
        help="Optional list of prompt ids (filenames without .txt) to run from --prompts-dir. If omitted, runs all prompts.",
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--ollama-url", type=str, default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--connect-timeout-s", type=float, default=5.0, help="HTTP connect timeout for Ollama calls.")
    ap.add_argument("--read-timeout-s", type=float, default=300.0, help="HTTP read timeout for Ollama calls.")
    ap.add_argument("--no-preflight", action="store_true", help="Skip Ollama connectivity/model preflight.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--embed-model", type=str, default=None, help="If set, call /api/embed for doc + summary.")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--status-every", type=int, default=1, help="Print a status line every N generations (0=never).")
    return ap.parse_args()


def preflight_ollama(*, cfg: OllamaConfig, models: List[str], logger: RunLogger) -> None:
    """
    Fail fast if Ollama isn't reachable, and warn if requested models aren't present.
    """
    url = f"{cfg.base_url}/api/tags"
    with logger.time_block("ollama.preflight", url=url):
        try:
            r = requests.get(url, timeout=(cfg.connect_timeout_s, min(10.0, cfg.read_timeout_s)))
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            raise SystemExit(
                f"Cannot reach Ollama at {cfg.base_url}. Start it (e.g. `ollama serve`) and ensure the API is available. "
                f"({type(e).__name__}: {e})"
            )

    names = set()
    for m in (payload.get("models") or []):
        if isinstance(m, dict) and "name" in m:
            names.add(str(m["name"]))
    missing = [m for m in models if m not in names]
    if missing:
        logger.log_event("ollama.preflight.model_missing", missing=missing, available_count=len(names))
        print(
            "Warning: some models are not present in Ollama yet: "
            + ", ".join(missing)
            + " (run `ollama pull <model>`).",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    cfg = OllamaConfig(
        base_url=args.ollama_url,
        connect_timeout_s=args.connect_timeout_s,
        read_timeout_s=args.read_timeout_s,
    )
    logger = RunLogger(args.run_dir)

    if args.embed_model and np is None:
        raise SystemExit("Missing dependency: numpy (required when --embed-model is set). Install via `pip install -r requirements.txt`.")

    docs_result = load_docs(
        dataset=args.dataset,
        docs_path=(str(args.docs) if args.docs is not None else None),
        split=args.split,
        max_docs=args.max_docs,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        hf_cache_dir=(str(args.hf_cache_dir) if args.hf_cache_dir is not None else None),
    )
    docs = docs_result.docs
    prompts = load_prompts(args.prompts_dir)
    if args.prompt_ids:
        prompt_ids = [p for p in args.prompt_ids if p]
        missing = [p for p in prompt_ids if p not in prompts]
        if missing:
            raise SystemExit(f"Unknown --prompt-ids: {missing}. Available: {sorted(list(prompts.keys()))}")
        prompts = {pid: prompts[pid] for pid in prompt_ids}

    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }

    logger.log_event(
        "run.config",
        dataset=docs_result.meta,
        prompts_dir=str(args.prompts_dir),
        models=args.models,
        seeds=args.seeds,
        options=options,
        top_logprobs=args.top_logprobs,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        connect_timeout_s=args.connect_timeout_s,
        read_timeout_s=args.read_timeout_s,
    )

    if not args.no_preflight:
        preflight_ollama(cfg=cfg, models=args.models, logger=logger)

    n_docs = len(docs)
    n_prompts = len(prompts)
    n_models = len(args.models)
    n_seeds = max(1, len(args.seeds))
    total = n_docs * n_prompts * n_models * n_seeds
    logger.log_event(
        "run.matrix",
        n_docs=n_docs,
        n_prompts=n_prompts,
        n_models=n_models,
        n_seeds=n_seeds,
        total_calls=total,
    )
    print(
        f"Run matrix: docs={n_docs} * prompts={n_prompts} * models={n_models} * seeds={n_seeds} = {total} generations",
        flush=True,
    )
    if tqdm is not None:
        pbar = tqdm(total=total, desc="generate")
    else:
        pbar = None

    for doc in docs:
        doc_id = str(doc.get("doc_id"))
        doc_text = str(doc.get("text", ""))
        doc_ref = doc.get("reference")
        if not doc_id or not doc_text:
            raise SystemExit(f"Bad doc record: {doc}")

        for prompt_id, prompt_template in prompts.items():
            full_prompt = build_prompt(prompt_template, doc_text)
            prompt_hash = _sha256_text(prompt_template)
            full_prompt_hash = _sha256_text(full_prompt)

            for model in args.models:
                for seed in (args.seeds or [None]):
                    if pbar is not None:
                        pbar.set_postfix_str(f"{prompt_id} {model} seed={seed}")
                    if args.status_every and (pbar is None or (pbar.n % max(1, args.status_every) == 0)):
                        msg = f"[{pbar.n if pbar is not None else 0}/{total}] doc={doc_id} prompt={prompt_id} model={model} seed={seed}"
                        if pbar is not None:
                            pbar.write(msg)
                        else:
                            print(msg, flush=True)

                    started = time.time()
                    error = None
                    resp: Dict[str, Any] = {}
                    try:
                        with logger.time_block(
                            "ollama.generate",
                            model=model,
                            prompt_id=prompt_id,
                            doc_id=doc_id,
                            seed=seed,
                        ):
                            resp = generate(
                                cfg=cfg,
                                model=model,
                                prompt=full_prompt,
                                seed=seed,
                                options=options,
                                logprobs=True,
                                top_logprobs=args.top_logprobs,
                            )
                    except Exception as e:
                        error = f"{type(e).__name__}: {e}"

                    ended = time.time()

                    summary = (resp.get("response") or "").strip()
                    prompt_eval_count = resp.get("prompt_eval_count")
                    eval_count = resp.get("eval_count")
                    prompt_eval_duration = resp.get("prompt_eval_duration")
                    eval_duration = resp.get("eval_duration")

                    # Store logprobs (often large) as gzipped JSON artifact.
                    logprobs_path = None
                    token_entries = extract_token_logprobs(resp) if resp else []
                    if token_entries:
                        raw = json.dumps(token_entries, ensure_ascii=False).encode("utf-8")
                        rel = f"logprobs/{_safe_name(doc_id)}/{_safe_name(model)}/{_safe_name(prompt_id)}/seed={seed}.json.gz"
                        logger.write_artifact_bytes(rel, gzip.compress(raw))
                        logprobs_path = str(Path("artifacts") / rel)

                    doc_embed_path = None
                    summary_embed_path = None
                    if args.embed_model and summary and not error:
                        try:
                            with logger.time_block(
                                "ollama.embed",
                                embed_model=args.embed_model,
                                model=model,
                                prompt_id=prompt_id,
                                doc_id=doc_id,
                                seed=seed,
                                kind="doc",
                            ):
                                doc_vec = np.asarray(embed(cfg=cfg, model=args.embed_model, text=doc_text), dtype=np.float32)
                            with logger.time_block(
                                "ollama.embed",
                                embed_model=args.embed_model,
                                model=model,
                                prompt_id=prompt_id,
                                doc_id=doc_id,
                                seed=seed,
                                kind="summary",
                            ):
                                sum_vec = np.asarray(embed(cfg=cfg, model=args.embed_model, text=summary), dtype=np.float32)

                            def _save_npy_bytes(arr: np.ndarray) -> bytes:
                                buf = BytesIO()
                                np.save(buf, arr, allow_pickle=False)
                                return buf.getvalue()

                            doc_rel = f"embeddings/{doc_id}/doc.npy"
                            doc_rel = f"embeddings/{_safe_name(doc_id)}/doc.npy"
                            sum_rel = f"embeddings/{_safe_name(doc_id)}/{_safe_name(model)}/{_safe_name(prompt_id)}/seed={seed}.npy"
                            logger.write_artifact_bytes(doc_rel, _save_npy_bytes(doc_vec))
                            logger.write_artifact_bytes(sum_rel, _save_npy_bytes(sum_vec))
                            doc_embed_path = str(Path("artifacts") / doc_rel)
                            summary_embed_path = str(Path("artifacts") / sum_rel)
                        except Exception as e:
                            logger.log_event(
                                "ollama.embed.error",
                                error=f"{type(e).__name__}: {e}",
                                embed_model=args.embed_model,
                                model=model,
                                prompt_id=prompt_id,
                                doc_id=doc_id,
                                seed=seed,
                            )

                    record = {
                        "ts_start_unix": started,
                        "ts_end_unix": ended,
                        "model": model,
                        "prompt_id": prompt_id,
                        "doc_id": doc_id,
                        "seed": seed,
                        "options": {**options, **({"seed": seed} if seed is not None else {})},
                        "top_logprobs": args.top_logprobs,
                        "prompt_template_sha256": prompt_hash,
                        "full_prompt_sha256": full_prompt_hash,
                        "doc_sha256": _sha256_text(doc_text),
                        "doc_len_chars": len(doc_text),
                        "doc_len_words": len(doc_text.split()),
                        "reference": doc_ref,
                        "summary": summary,
                        "summary_sha256": _sha256_text(summary) if summary else None,
                        "summary_len_chars": len(summary),
                        "summary_len_words": len(summary.split()),
                        "prompt_eval_count": prompt_eval_count,
                        "eval_count": eval_count,
                        "prompt_eval_duration": prompt_eval_duration,
                        "eval_duration": eval_duration,
                        "logprobs_path": logprobs_path,
                        "doc_embedding_path": doc_embed_path,
                        "summary_embedding_path": summary_embed_path,
                        "error": error,
                        "raw_response_keys": (sorted(list(resp.keys())) if isinstance(resp, dict) else None),
                    }
                    logger.write_generation(record)
                    if pbar is not None:
                        pbar.update(1)

    if pbar is not None:
        pbar.close()
    logger.log_event("run.done", run_dir=str(args.run_dir))


if __name__ == "__main__":
    main()
