from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DocsResult:
    docs: List[Dict[str, Any]]
    meta: Dict[str, Any]


def _read_jsonl(*, path: str) -> List[Dict[str, Any]]:
    from pathlib import Path
    import json

    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _maybe_shuffle_and_limit(
    docs: List[Dict[str, Any]], *, shuffle: bool, shuffle_seed: int, max_docs: Optional[int]
) -> List[Dict[str, Any]]:
    if shuffle:
        import random

        rnd = random.Random(shuffle_seed)
        rnd.shuffle(docs)
    if max_docs is not None:
        docs = docs[: max(0, int(max_docs))]
    return docs


def _require_hf_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency: datasets. Install via `pip install -r requirements.txt`. ({type(e).__name__}: {e})"
        )
    return load_dataset


def load_docs(
    *,
    dataset: str,
    docs_path: Optional[str],
    split: str,
    max_docs: Optional[int],
    shuffle: bool,
    shuffle_seed: int,
    hf_cache_dir: Optional[str],
) -> DocsResult:
    if dataset == "jsonl":
        if not docs_path:
            raise SystemExit("--docs is required when --dataset=jsonl")
        docs = _read_jsonl(path=docs_path)
        docs = _maybe_shuffle_and_limit(docs, shuffle=shuffle, shuffle_seed=shuffle_seed, max_docs=max_docs)

        meta = {
            "dataset": "jsonl",
            "docs_path": docs_path,
            "split": None,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
            "max_docs": max_docs,
        }
        return DocsResult(docs=docs, meta=meta)

    if dataset == "xsum":
        docs_path = docs_path or os.environ.get("XSUM_JSONL") or os.path.join("data", "xsum_dataset.jsonl")
        if not os.path.exists(docs_path):
            raise SystemExit(
                f"Missing XSum JSONL: {docs_path} (pass --docs, set $XSUM_JSONL, or run from repo root)."
            )

        raw_docs = _read_jsonl(path=docs_path)
        raw_docs = _maybe_shuffle_and_limit(raw_docs, shuffle=shuffle, shuffle_seed=shuffle_seed, max_docs=max_docs)

        docs: List[Dict[str, Any]] = []
        for row in raw_docs:
            doc_id = row.get("doc_id", row.get("id"))
            text = row.get("text", row.get("document"))
            reference = row.get("reference", row.get("summary"))
            if doc_id is None or text is None:
                raise SystemExit(f"Bad XSum record (expected id/document): {row}")

            doc: Dict[str, Any] = {"doc_id": str(doc_id), "text": str(text).strip()}
            if reference is not None:
                doc["reference"] = str(reference).strip()
            docs.append(doc)

        meta = {
            "dataset": "xsum",
            "docs_path": docs_path,
            "split": None,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
            "max_docs": max_docs,
        }
        return DocsResult(docs=docs, meta=meta)

    if dataset == "cnn_dailymail":
        load_dataset = _require_hf_datasets()

        ds = load_dataset(
            "abisee/cnn_dailymail",
            "3.0.0",
            split=split,
            cache_dir=hf_cache_dir,
        )

        if shuffle:
            ds = ds.shuffle(seed=shuffle_seed)
        if max_docs is not None:
            ds = ds.select(range(min(max_docs, len(ds))))

        docs: List[Dict[str, Any]] = []
        for row in ds:
            docs.append(
                {
                    "doc_id": str(row.get("id")),
                    "text": str(row.get("article", "")),
                    "reference": str(row.get("highlights", "")),
                }
            )

        meta = {
            "dataset": "cnn_dailymail",
            "hf_dataset": "abisee/cnn_dailymail",
            "hf_config": "3.0.0",
            "split": split,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
            "max_docs": max_docs,
            "hf_cache_dir": hf_cache_dir,
        }
        return DocsResult(docs=docs, meta=meta)

    raise SystemExit(f"Unknown --dataset: {dataset}")
