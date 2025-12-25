from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_xsum(split: str, *, cache_dir: str | None) -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency: datasets. Install via `pip install -r requirements.txt`. ({e})")

    last_err: Exception | None = None
    for name in ["xsum", "EdinburghNLP/xsum"]:
        try:
            return load_dataset(name, split=split, cache_dir=cache_dir)
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    raise SystemExit(f"Failed to load XSum via HF datasets. Last error: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch XSum via HF datasets and write data/xsum_dataset.jsonl.")
    ap.add_argument("--out", default="data/xsum_dataset.jsonl", help="Output JSONL path.")
    ap.add_argument("--split", default="validation", help="HF split (train/validation/test).")
    ap.add_argument("--max-docs", type=int, default=None, help="Optional cap.")
    ap.add_argument("--cache-dir", default=None, help="Optional HF cache dir.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = _load_xsum(args.split, cache_dir=args.cache_dir)
    n = len(ds)
    if args.max_docs is not None:
        n = min(n, max(0, int(args.max_docs)))

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = ds[i]
            rec: Dict[str, Any] = {
                "id": row.get("id"),
                "document": row.get("document"),
                "summary": row.get("summary"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()

