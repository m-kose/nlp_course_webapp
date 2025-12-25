# nlp_course – Ollama summarization runs + metrics

This repo contains a lightweight pipeline to:

1) Generate summaries via **Ollama** (with `logprobs` + `top_logprobs`)
2) Log every `(model, prompt_id, doc_id, seed)` call in a reproducible run directory
3) Compute the metrics described in `metrics.txt` from those logs

## Quickstart

1. Start Ollama locally (default URL `http://localhost:11434`).
2. Ensure the model is available:
   - `ollama pull ministral-3:8b`

## Run generation (logs + optional embeddings)

Example using the included toy dataset:

```bash
python scripts/run_ollama_matrix.py \
  --docs data/example_docs.jsonl \
  --prompts-dir prompts \
  --models ministral-3:8b \
  --seeds 1 2 3 4 5 \
  --top-logprobs 20 \
  --embed-model nomic-embed-text \
  --run-dir runs/example
```

## Web UI (Flask)

Install deps:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python -m nlp_course.webapp
```

Then open `http://127.0.0.1:5000` and start a run. The UI can:
- Fetch installed Ollama models from `/api/tags`
- Start/stop runs (writes into `runs/`)
- Live-tail `events.jsonl` / `generations.jsonl` while a run is executing
- Edit/create prompt templates (writes into `prompts/`)

Optional (recommended if you ever open the port publicly): require an access token:

```bash
WEBAPP_TOKEN="change-me" python -m nlp_course.webapp
```

Then visit `http://127.0.0.1:5000/?token=change-me` (or open `/auth` to paste the token). The UI will keep using the token for API calls, but page loads require `?token=...`.

### Access the web UI from your laptop (SSH tunnel)

If the server is running on a remote machine (e.g. AWS EC2), keep the Flask server bound to `127.0.0.1` (default) and use an SSH tunnel from your laptop:

```bash
ssh -i /path/to/key.pem -L 5000:127.0.0.1:5000 ubuntu@<EC2_PUBLIC_IP_OR_DNS>
```

Then open this on your laptop:
- `http://127.0.0.1:5000`

Notes:
- This avoids opening port 5000 in your EC2 Security Group.
- The UI calls Ollama from the server’s perspective. If Ollama is on the same EC2 instance, keep `Ollama URL = http://localhost:11434`.

## Use CNN/DailyMail (HF datasets)

This downloads the dataset via `datasets.load_dataset("abisee/cnn_dailymail", "3.0.0")` and maps:
- `article` -> `text`
- `highlights` -> `reference`
- `id` -> `doc_id`

```bash
python scripts/run_ollama_matrix.py \
  --dataset cnn_dailymail \
  --split validation \
  --max-docs 200 \
  --shuffle --shuffle-seed 0 \
  --prompts-dir prompts \
  --models ministral-3:8b \
  --seeds 1 2 3 4 5 \
  --top-logprobs 20 \
  --run-dir runs/cnn_dm_val_200
```

## Use XSum (local JSONL)

This repo includes `data/xsum_dataset.jsonl` and maps:
- `id` -> `doc_id`
- `document` -> `text`
- `summary` -> `reference`

```bash
python scripts/run_ollama_matrix.py \
  --dataset xsum \
  --max-docs 200 \
  --shuffle --shuffle-seed 0 \
  --prompts-dir prompts \
  --models ministral-3:8b \
  --seeds 1 2 3 4 5 \
  --top-logprobs 20 \
  --run-dir runs/xsum_200
```

If the file is elsewhere, pass `--docs path/to/xsum.jsonl` or set `XSUM_JSONL`.

## Troubleshooting

- If it looks “stuck” at `generate: 0%`, the first Ollama call is still running (cold start, model load/download, long article prompt eval). The script now prints the active `(doc,prompt,model,seed)` cell and runs an Ollama preflight by default.
- If Ollama isn’t running, you’ll get a fast error from the preflight; start it (e.g. `ollama serve`) and ensure the model is pulled (e.g. `ollama pull ministral-3:8b`).
- Tune timeouts with `--connect-timeout-s` / `--read-timeout-s`, or skip preflight with `--no-preflight`.

## Compute metrics

```bash
python scripts/compute_metrics.py \
  --run-dir runs/example \
  --baseline-prompt zero_shot_prompting
```

Outputs:
- `runs/<run>/metrics/per_generation.csv`
- `runs/<run>/metrics/by_model_prompt.csv`
- `runs/<run>/metrics/mi_prompt_from_summary.csv`
- `runs/<run>/metrics/retrieval_alignment.csv`

## Metrics captured (short guide)

All metrics are computed from the run logs (and optional artifacts) and can be computed on a **partial run** (whatever has finished so far).

### Baseline “reference-based” metrics (need `reference`)

These require `reference` summaries (CNN/DailyMail highlights, or your own dataset’s references):

- **ROUGE-1/2/L (F1)**: overlap with the reference at unigram/bigram/LCS levels; best for “surface overlap” comparisons.
- **BLEU**: n-gram precision with brevity penalty; mainly a sanity baseline for summarization.
- **BERTScore (F1)**: semantic similarity using contextual embeddings; more tolerant of paraphrase/format changes.
- **BARTScore (optional)**: implemented as the average per-token NLL of the reference conditioned on the generated summary using `facebook/bart-large-cnn` (requires `torch` + `transformers` and may download weights).

### Length / control diagnostics

- **Summary length**: `summary_len_words`, `eval_count` (when provided by Ollama).
- **Compression ratio**: `compression_words` / `compression_chars` = summary length divided by document length.

### Ollama logprobs metrics (need `logprobs`)

These require generation with `logprobs: true` (the runner enables this):

- **Per-token NLL (`mean_nll`)**: average surprise of the sampled tokens under the model (lower is “more probable under the model”).
- **Entropy profile**: approximate next-token uncertainty from `top_logprobs`:
  - `entropy_mean`, `entropy_slope`, `entropy_spike_count`

### Robustness across seeds (need multiple seeds per cell)

Computed within each `(model, prompt_id, doc_id)` across seeds:

- **`nll_var`**: variance of `mean_nll` (instability proxy).
- **`mean_abs_typicality_z`**: how “atypical” a sample is versus its siblings (z-score of `mean_nll` within the cell).
- **`summary_emb_pairwise_cos_mean`**: average pairwise cosine similarity across seed summaries (needs embeddings).

### Embedding-based “scientific” metrics (need `--embed-model ...`)

If you run generation with `--embed-model <embedding_model>`, the runner stores doc + summary embeddings:

- **Prompt–summary MI proxy** (`mi_prompt_from_summary.csv`): trains a classifier to predict `prompt_id` from summary embeddings (per model); high MI means prompt style leaves a strong signature.
- **Doc–summary contrastive alignment** (`retrieval_alignment.csv`):
  - `rank` / `recall@k`: does the summary retrieve its own doc among a pool?
  - `infonce`: InfoNCE-style loss (lower is better alignment).
- **Prompt Reliance Index (PRI)** (`prompt_reliance_index.csv`, needs `--baseline-prompt`):
  - `pri_embed_dist`: embedding distance vs baseline prompt’s summary on the same `(doc, model, seed)`.
  - `pri_rougeL_vs_base`: lexical overlap vs baseline (optional; needs `rouge-score`).
- **Genericity index** (`genericity_index.csv`): detects “template-y” behavior:
  - `embedding_genericity_mean_pairwise_cos`: similarity across different docs (higher = more generic).
  - `repeated_trigram_type_rate`: repeated trigram types across docs (higher = more boilerplate).

### Cost / throughput (needs Ollama token counts + durations)

- **Latency**: `total_latency_s`
- **Throughput**: `output_toks_per_s` from `eval_count` and `eval_duration`
- **Example “quality per token”**: `rougeL_per_generated_tok` (if ROUGE is available)

## Clean a run directory

Dry-run:

```bash
python scripts/clean_run.py --run-dir runs/example
```

Actually delete (metrics + artifacts + logs):

```bash
python scripts/clean_run.py --run-dir runs/example --yes
```

## Data format

Input docs are JSONL with at least:

```json
{"doc_id":"doc-001","text":"...","reference":"(optional reference summary)"}
```
