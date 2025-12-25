from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = None  # type: ignore
pd = None  # type: ignore
LogisticRegression = None  # type: ignore
GroupKFold = None  # type: ignore

bootstrap_ci = None  # type: ignore
compute_logprob_metrics = None  # type: ignore
cosine_sim = None  # type: ignore
read_json_gz = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--baseline-prompt", type=str, default=None, help="Prompt id to use as P0 for PRI.")
    ap.add_argument("--tau", type=float, default=0.07, help="Temperature for InfoNCE-style scores.")
    ap.add_argument("--retrieval-k", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=["all"],
        choices=["all", "reference", "logprobs", "embeddings", "bertscore", "bartscore"],
        help=(
            "Metric groups to compute. Default: all. "
            "Use e.g. `--metrics reference logprobs` to skip embedding/BERTScore/BARTScore."
        ),
    )
    return ap.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _to_csv_atomic(df: "pd.DataFrame", path: Path, **kwargs: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, **kwargs)
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def try_rouge_bleu(rows: pd.DataFrame) -> pd.DataFrame:
    ref = rows["reference"].fillna("")
    hyp = rows["summary"].fillna("")
    if (ref == "").all():
        rows["rouge1_f"] = np.nan
        rows["rouge2_f"] = np.nan
        rows["rougeL_f"] = np.nan
        rows["bleu"] = np.nan
        return rows

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for r, h in zip(ref.tolist(), hyp.tolist()):
            if not r or not h:
                r1.append(np.nan), r2.append(np.nan), rl.append(np.nan)
                continue
            s = scorer.score(r, h)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        rows["rouge1_f"] = r1
        rows["rouge2_f"] = r2
        rows["rougeL_f"] = rl
    except Exception:
        rows["rouge1_f"] = np.nan
        rows["rouge2_f"] = np.nan
        rows["rougeL_f"] = np.nan

    try:
        import sacrebleu

        bleu_scores = []
        for r, h in zip(ref.tolist(), hyp.tolist()):
            if not r or not h:
                bleu_scores.append(np.nan)
                continue
            bleu_scores.append(sacrebleu.sentence_bleu(h, [r]).score)
        rows["bleu"] = bleu_scores
    except Exception:
        rows["bleu"] = np.nan

    return rows


def try_bertscore(rows: pd.DataFrame) -> pd.DataFrame:
    ref = rows["reference"].fillna("")
    hyp = rows["summary"].fillna("")
    if (ref == "").all():
        rows["bertscore_f1"] = np.nan
        return rows
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score(hyp.tolist(), ref.tolist(), lang="en", verbose=False)
        rows["bertscore_f1"] = [float(x) for x in F1]
    except Exception:
        rows["bertscore_f1"] = np.nan
    return rows


def try_bartscore(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal BARTScore-style scorer (optional): average NLL of reference given hypothesis.
    This requires transformers+torch and will download a model if not present.
    """
    ref = rows["reference"].fillna("")
    hyp = rows["summary"].fillna("")
    if (ref == "").all():
        rows["bartscore_ref_given_hyp_nll"] = np.nan
        return rows

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/bart-large-cnn"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        mdl.eval()

        scores = []
        with torch.no_grad():
            for r, h in zip(ref.tolist(), hyp.tolist()):
                if not r or not h:
                    scores.append(np.nan)
                    continue
                enc = tok(h, return_tensors="pt", truncation=True, max_length=1024).to(device)
                lab = tok(r, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
                out = mdl(**enc, labels=lab)
                # out.loss is mean per-token NLL (natural log)
                scores.append(float(out.loss.detach().cpu()))
        rows["bartscore_ref_given_hyp_nll"] = scores
    except Exception:
        rows["bartscore_ref_given_hyp_nll"] = np.nan
    return rows


def load_embedding(path: str | None) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.load(p, allow_pickle=False)
    except Exception:
        return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0 or not np.isfinite(n):
        return vec
    return vec / n


def genericity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Boilerplate / genericity proxy:
      - embedding: average pairwise cosine similarity across different docs
      - lexical: repeated trigram type rate across docs
    """
    rows = []
    for (model, prompt_id), sub in df.groupby(["model", "prompt_id"]):
        sub = sub.copy()

        # Collapse multiple seeds into one embedding per doc (mean, then normalize).
        emb_by_doc = []
        for doc_id, dsub in sub.dropna(subset=["summary_embedding"]).groupby("doc_id"):
            embs = np.stack(dsub["summary_embedding"].tolist(), axis=0)
            emb_by_doc.append(_normalize(np.mean(embs, axis=0)))

        if len(emb_by_doc) >= 2:
            sims = []
            for i in range(len(emb_by_doc)):
                for j in range(i + 1, len(emb_by_doc)):
                    sims.append(cosine_sim(emb_by_doc[i], emb_by_doc[j]))
            emb_genericity = float(np.nanmean(np.array(sims, dtype=np.float64)))
        else:
            emb_genericity = float("nan")

        # Repeated trigram rate across docs (use one summary per doc: first seed).
        trigram_counts = {}
        total_types = 0
        for doc_id, dsub in sub.groupby("doc_id"):
            s = str(dsub["summary"].iloc[0] or "").lower()
            toks = [t for t in "".join([c if c.isalnum() or c.isspace() else " " for c in s]).split() if t]
            trigrams = set(zip(toks, toks[1:], toks[2:])) if len(toks) >= 3 else set()
            for tg in trigrams:
                trigram_counts[tg] = trigram_counts.get(tg, 0) + 1
            total_types += len(trigrams)

        if trigram_counts:
            repeated_types = sum(1 for c in trigram_counts.values() if c >= 2)
            trigram_repeat_rate = repeated_types / max(1, len(trigram_counts))
        else:
            trigram_repeat_rate = float("nan")

        rows.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                "num_docs": int(sub["doc_id"].nunique()),
                "embedding_genericity_mean_pairwise_cos": emb_genericity,
                "repeated_trigram_type_rate": trigram_repeat_rate,
            }
        )
    return pd.DataFrame(rows)


def _require_deps(*, need_sklearn: bool) -> None:
    global np, pd, LogisticRegression, GroupKFold, bootstrap_ci, compute_logprob_metrics, cosine_sim, read_json_gz
    try:
        import numpy as _np  # type: ignore
        import pandas as _pd  # type: ignore

        from nlp_course.metrics_core import (  # type: ignore
            bootstrap_ci as _bootstrap_ci,
            compute_logprob_metrics as _compute_logprob_metrics,
            cosine_sim as _cosine_sim,
            read_json_gz as _read_json_gz,
        )

        if need_sklearn:
            from sklearn.linear_model import LogisticRegression as _LogisticRegression  # type: ignore
            from sklearn.model_selection import GroupKFold as _GroupKFold  # type: ignore
        else:
            _LogisticRegression = None  # type: ignore
            _GroupKFold = None  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing metric dependencies. Install via `pip install -r requirements.txt`. ({type(e).__name__}: {e})")

    np = _np
    pd = _pd
    LogisticRegression = _LogisticRegression
    GroupKFold = _GroupKFold
    bootstrap_ci = _bootstrap_ci
    compute_logprob_metrics = _compute_logprob_metrics
    cosine_sim = _cosine_sim
    read_json_gz = _read_json_gz


def _metric_flags(metrics: Sequence[str]) -> Dict[str, bool]:
    # argparse with nargs="*" can produce:
    # - ["all"] when the flag is omitted (default)
    # - [] when the flag is provided with no values (explicit "none")
    if metrics is not None and len(metrics) == 0:
        sel: set[str] = set()
        all_on = False
    else:
        sel = {m for m in (metrics or []) if m}
        all_on = ("all" in sel) or (not sel)
    return {
        "all": all_on,
        "reference": all_on or ("reference" in sel),
        "logprobs": all_on or ("logprobs" in sel),
        "embeddings": all_on or ("embeddings" in sel),
        "bertscore": all_on or ("bertscore" in sel),
        "bartscore": all_on or ("bartscore" in sel),
    }


def compute_retrieval_alignment(df: pd.DataFrame, *, tau: float, ks: List[int]) -> pd.DataFrame:
    # Use one doc embedding per doc_id, and one summary embedding per generation.
    doc_vecs = {}
    for doc_id, sub in df.dropna(subset=["doc_embedding"]).groupby("doc_id"):
        doc_vecs[doc_id] = sub["doc_embedding"].iloc[0]
    doc_ids = sorted(doc_vecs.keys())
    if not doc_ids:
        return pd.DataFrame()
    D = np.stack([doc_vecs[d] for d in doc_ids], axis=0)

    rows = []
    for i, row in df.dropna(subset=["summary_embedding"]).iterrows():
        doc_id = row["doc_id"]
        if doc_id not in doc_vecs:
            continue
        s = row["summary_embedding"]
        sims = (D @ s) / (np.linalg.norm(D, axis=1) * np.linalg.norm(s) + 1e-12)
        order = np.argsort(-sims)
        rank = int(np.where(np.array(doc_ids)[order] == doc_id)[0][0]) + 1
        # InfoNCE
        logits = sims / tau
        lse = float(np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits))
        pos = float(logits[doc_ids.index(doc_id)])
        infonce = float(-(pos - lse))

        out = {
            "model": row["model"],
            "prompt_id": row["prompt_id"],
            "doc_id": doc_id,
            "seed": row["seed"],
            "rank": rank,
            "infonce": infonce,
        }
        for k in ks:
            out[f"recall@{k}"] = 1.0 if rank <= k else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def compute_prompt_mi(df: pd.DataFrame) -> pd.DataFrame:
    # Per model: classifier prompt_id from summary embedding, group-split by doc_id.
    rows = []
    for model, sub in df.dropna(subset=["summary_embedding"]).groupby("model"):
        X = np.stack(sub["summary_embedding"].tolist(), axis=0)
        y = sub["prompt_id"].astype(str).to_numpy()
        groups = sub["doc_id"].astype(str).to_numpy()

        prompts = sorted(set(y.tolist()))
        if len(prompts) < 2 or len(np.unique(groups)) < 2:
            continue

        k = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=k)
        ce = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            # Newer scikit-learn versions removed the `multi_class` kwarg; with lbfgs, multinomial is used internally.
            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])
            classes = clf.classes_
            y_test = y[test_idx]
            # Cross-entropy (nats)
            idx = np.array([int(np.where(classes == yy)[0][0]) for yy in y_test])
            ce_fold = -np.mean(np.log(np.clip(probs[np.arange(len(idx)), idx], 1e-12, 1.0)))
            ce.append(float(ce_fold))

        cross_entropy = float(np.mean(ce)) if ce else float("nan")
        H = float(math.log(len(prompts)))
        mi = float(H - cross_entropy) if math.isfinite(cross_entropy) else float("nan")
        rows.append({"model": model, "num_prompts": len(prompts), "cross_entropy_nats": cross_entropy, "mi_nats": mi})
    return pd.DataFrame(rows)


def compute_pri(df: pd.DataFrame, baseline_prompt: str) -> pd.DataFrame:
    # PRI: compare each prompt vs baseline on same (doc, model, seed).
    base = df[df["prompt_id"] == baseline_prompt].copy()
    if base.empty:
        return pd.DataFrame()
    base = base[["doc_id", "model", "seed", "summary_embedding", "summary"]].rename(
        columns={"summary_embedding": "base_embedding", "summary": "base_summary"}
    )
    merged = df.merge(base, on=["doc_id", "model", "seed"], how="left")
    merged = merged.dropna(subset=["summary_embedding", "base_embedding"])
    if merged.empty:
        return pd.DataFrame()

    merged["pri_embed_dist"] = [
        1.0 - cosine_sim(a, b) for a, b in zip(merged["summary_embedding"].tolist(), merged["base_embedding"].tolist())
    ]
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        pri_lex = []
        for s, b in zip(merged["summary"].fillna("").tolist(), merged["base_summary"].fillna("").tolist()):
            if not s or not b:
                pri_lex.append(np.nan)
            else:
                pri_lex.append(scorer.score(b, s)["rougeL"].fmeasure)
        merged["pri_rougeL_vs_base"] = pri_lex
    except Exception:
        merged["pri_rougeL_vs_base"] = np.nan

    return merged[
        ["model", "prompt_id", "doc_id", "seed", "pri_embed_dist", "pri_rougeL_vs_base"]
    ]


def main() -> None:
    started_perf = time.perf_counter()
    args = parse_args()
    flags = _metric_flags(args.metrics)
    _require_deps(need_sklearn=bool(flags["embeddings"]))
    run_dir = args.run_dir
    gen_path = run_dir / "generations.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"Missing generations log: {gen_path}")

    rows = read_jsonl(gen_path)
    df = pd.DataFrame(rows)
    if "error" not in df.columns:
        df["error"] = None
    df = df[df["error"].isna()].copy()
    if df.empty:
        raise SystemExit("No successful generations found.")

    required_key = ["model", "prompt_id", "doc_id", "seed"]
    missing_key = [c for c in required_key if c not in df.columns]
    if missing_key:
        raise SystemExit(f"generations.jsonl missing required columns: {missing_key}")

    # Some optional columns may be missing on older runs; make them explicit.
    for col in [
        "reference",
        "summary",
        "logprobs_path",
        "doc_embedding_path",
        "summary_embedding_path",
        "summary_len_words",
        "doc_len_words",
        "summary_len_chars",
        "doc_len_chars",
        "eval_count",
        "prompt_eval_count",
        "eval_duration",
        "prompt_eval_duration",
        "ts_start_unix",
        "ts_end_unix",
    ]:
        if col not in df.columns:
            df[col] = np.nan if col.endswith(("_words", "_chars", "_count", "_duration", "_unix")) else None

    # Ensure numeric columns (Ollama sometimes returns null/missing)
    for col in ["eval_count", "prompt_eval_count", "eval_duration", "prompt_eval_duration", "ts_start_unix", "ts_end_unix"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If you rerun into the same --run-dir, generations.jsonl appends.
    # Deduplicate by the natural key and keep the latest completion time.
    before = len(df)
    df = df.sort_values(["ts_end_unix", "ts_start_unix"], na_position="last")
    df = df.drop_duplicates(subset=["model", "prompt_id", "doc_id", "seed"], keep="last")
    after = len(df)
    if after != before:
        print(f"Note: deduped successful generations {before} -> {after} (same run dir appended).")

    # Length & compression diagnostics
    df["compression_words"] = df["summary_len_words"] / df["doc_len_words"].replace(0, np.nan)
    df["compression_chars"] = df["summary_len_chars"] / df["doc_len_chars"].replace(0, np.nan)

    # Prompt cost / throughput
    df["total_latency_s"] = df["ts_end_unix"] - df["ts_start_unix"]
    df["output_toks_per_s"] = df["eval_count"] / (df["eval_duration"].replace(0, np.nan) / 1e9)

    # Logprob-derived metrics (NLL + entropy profile)
    if flags["logprobs"]:
        mean_nll = []
        entropy_mean = []
        entropy_slope = []
        entropy_spikes = []
        logprob_token_count = []
        for p in df["logprobs_path"].tolist():
            if not p:
                mean_nll.append(np.nan)
                entropy_mean.append(np.nan)
                entropy_slope.append(np.nan)
                entropy_spikes.append(np.nan)
                logprob_token_count.append(0)
                continue
            token_entries = read_json_gz(run_dir / Path(p))
            m = compute_logprob_metrics(token_entries)
            if m is None:
                mean_nll.append(np.nan)
                entropy_mean.append(np.nan)
                entropy_slope.append(np.nan)
                entropy_spikes.append(np.nan)
                logprob_token_count.append(0)
            else:
                mean_nll.append(m.mean_nll)
                entropy_mean.append(m.entropy_mean)
                entropy_slope.append(m.entropy_slope)
                entropy_spikes.append(m.entropy_spike_count)
                logprob_token_count.append(m.token_count)
        df["mean_nll"] = mean_nll
        df["entropy_mean"] = entropy_mean
        df["entropy_slope"] = entropy_slope
        df["entropy_spike_count"] = entropy_spikes
        df["logprob_token_count"] = logprob_token_count
    else:
        df["mean_nll"] = np.nan
        df["entropy_mean"] = np.nan
        df["entropy_slope"] = np.nan
        df["entropy_spike_count"] = np.nan
        df["logprob_token_count"] = 0

    # Typicality / instability across seeds: z-score within (model, prompt, doc)
    df["typicality_z_mean_nll_within_cell"] = (
        df.groupby(["model", "prompt_id", "doc_id"])["mean_nll"]
        .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan))
    )

    # Optional baselines (require reference)
    if flags["reference"]:
        df = try_rouge_bleu(df)
    else:
        df["rouge1_f"] = np.nan
        df["rouge2_f"] = np.nan
        df["rougeL_f"] = np.nan
        df["bleu"] = np.nan

    if flags["bertscore"]:
        df = try_bertscore(df)
    else:
        df["bertscore_f1"] = np.nan

    if flags["bartscore"]:
        df = try_bartscore(df)
    else:
        df["bartscore_ref_given_hyp_nll"] = np.nan

    # Load embeddings for embedding-based metrics.
    if flags["embeddings"]:
        df["doc_embedding"] = [
            load_embedding(str(run_dir / Path(p))) if p else None for p in df["doc_embedding_path"].tolist()
        ]
        df["summary_embedding"] = [
            load_embedding(str(run_dir / Path(p))) if p else None for p in df["summary_embedding_path"].tolist()
        ]
    else:
        df["doc_embedding"] = None
        df["summary_embedding"] = None

    # "Quality per token" examples
    df["rougeL_per_generated_tok"] = df["rougeL_f"] / df["eval_count"].replace(0, np.nan)

    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid writing full embedding vectors into CSV (can get enormous).
    _to_csv_atomic(
        df.drop(columns=["doc_embedding", "summary_embedding"], errors="ignore"),
        out_dir / "per_generation.csv",
        index=False,
    )

    # By (model, prompt): mean + bootstrap CI for a few key metrics
    agg_rows = []
    group_cols = ["model", "prompt_id"]
    for (model, prompt_id), sub in df.groupby(group_cols):
        rec = {"model": model, "prompt_id": prompt_id, "n": int(len(sub))}
        for col in [
            "rouge1_f",
            "rouge2_f",
            "rougeL_f",
            "bleu",
            "bertscore_f1",
            "bartscore_ref_given_hyp_nll",
            "compression_words",
            "mean_nll",
            "entropy_mean",
            "output_toks_per_s",
            "total_latency_s",
        ]:
            vals = sub[col].to_numpy(dtype=np.float64)
            rec[f"{col}_mean"] = float(np.nanmean(vals))
            lo, hi = bootstrap_ci(vals, iters=500, seed=0)
            rec[f"{col}_ci_lo"] = lo
            rec[f"{col}_ci_hi"] = hi
        agg_rows.append(rec)
    _to_csv_atomic(pd.DataFrame(agg_rows), out_dir / "by_model_prompt.csv", index=False)

    # Stability across seeds: variance of NLL and embedding similarity per (model, prompt, doc)
    stab_rows = []
    for (model, prompt_id, doc_id), sub in df.groupby(["model", "prompt_id", "doc_id"]):
        if len(sub) < 2:
            continue
        nll_var = float(np.nanvar(sub["mean_nll"].to_numpy(dtype=np.float64)))
        z_abs_mean = float(np.nanmean(np.abs(sub["typicality_z_mean_nll_within_cell"].to_numpy(dtype=np.float64))))
        embs = [e for e in sub["summary_embedding"].tolist() if e is not None and len(e) > 0]
        if len(embs) >= 2:
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sims.append(cosine_sim(embs[i], embs[j]))
            emb_sim_mean = float(np.nanmean(np.array(sims, dtype=np.float64)))
        else:
            emb_sim_mean = float("nan")
        stab_rows.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                "doc_id": doc_id,
                "k_seeds": int(len(sub)),
                "nll_var": nll_var,
                "mean_abs_typicality_z": z_abs_mean,
                "summary_emb_pairwise_cos_mean": emb_sim_mean,
            }
        )
    _to_csv_atomic(pd.DataFrame(stab_rows), out_dir / "stability_across_seeds.csv", index=False)

    # Prompt–summary MI proxy (prompt_id from summary embedding)
    if flags["embeddings"]:
        mi_df = compute_prompt_mi(df)
        if mi_df.empty:
            mi_df = pd.DataFrame(columns=["model", "num_prompts", "cross_entropy_nats", "mi_nats"])
    else:
        mi_df = pd.DataFrame(columns=["model", "num_prompts", "cross_entropy_nats", "mi_nats"])
    _to_csv_atomic(mi_df, out_dir / "mi_prompt_from_summary.csv", index=False)

    # Doc–summary contrastive alignment (retrieval / InfoNCE)
    if flags["embeddings"]:
        retrieval = compute_retrieval_alignment(df, tau=args.tau, ks=args.retrieval_k)
    else:
        retrieval = pd.DataFrame()
    if retrieval.empty:
        cols = ["model", "prompt_id", "doc_id", "seed", "rank", "infonce"] + [
            f"recall@{k}" for k in args.retrieval_k
        ]
        retrieval = pd.DataFrame(columns=cols)
    _to_csv_atomic(retrieval, out_dir / "retrieval_alignment.csv", index=False)

    # Boilerplate / genericity index
    gen = genericity_index(df) if flags["embeddings"] else pd.DataFrame()
    if gen.empty:
        gen = pd.DataFrame(
            columns=[
                "model",
                "prompt_id",
                "num_docs",
                "embedding_genericity_mean_pairwise_cos",
                "repeated_trigram_type_rate",
            ]
        )
    _to_csv_atomic(gen, out_dir / "genericity_index.csv", index=False)

    # Prompt reliance index
    pri: pd.DataFrame
    if flags["embeddings"] and args.baseline_prompt:
        pri = compute_pri(df, args.baseline_prompt)
    else:
        pri = pd.DataFrame()
    if pri.empty:
        pri = pd.DataFrame(columns=["model", "prompt_id", "doc_id", "seed", "pri_embed_dist", "pri_rougeL_vs_base"])
    _to_csv_atomic(pri, out_dir / "prompt_reliance_index.csv", index=False)

    _write_text_atomic(
        out_dir / "metrics_meta.json",
        json.dumps(
            {
                "ts_utc": _utc_now_iso(),
                "run_dir": str(run_dir),
                "baseline_prompt": args.baseline_prompt,
                "tau": args.tau,
                "retrieval_k": args.retrieval_k,
                "metrics_arg": args.metrics,
                "computed": {k: bool(v) for k, v in flags.items() if k != "all"},
                "elapsed_s": time.perf_counter() - started_perf,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
