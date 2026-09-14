# hybrid-comparisons

A benchmark harness for comparing Qdrant hybrid search configurations:
quantization / datatype schemes, prefetch strategies (dense+sparse vs
sparse-only), and final-stage rescorers (server-side ColBERT, a local
cross-encoder rerank, or plain RRF fusion with no rescore at all).

Vectors are computed server-side via **Qdrant Cloud Inference** (no
local embedding step). The pipeline is:

```
download-data  →  data/corpus.jsonl  →  qdrant-load  →  Qdrant Cloud collection
                                                              ↓
                                     results/*.json  ←  eval-harness
                                              ↓
                                     summarize_results.py → results/summary.csv
```

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) and Python 3.13+
- A Qdrant Cloud cluster with **Cloud Inference enabled** (Cluster Detail →
  Inference tab in the Qdrant Cloud Console)
- `QDRANT_URL` and `QDRANT_API_KEY` for that cluster

Each package under `packages/` is an independent `uv` project. Export the
credentials once, or drop a `.env` file (`QDRANT_URL=...` / `QDRANT_API_KEY=...`)
into `packages/eval-harness/` and `packages/qdrant-load/`.


## 1. Download the corpus

```bash
cd packages/download-data
uv run download-data ../../data/corpus.jsonl
```

Pulls the first 100k rows of
[`jordane95/msmarco-passage-corpus-with-query`](https://huggingface.co/datasets/jordane95/msmarco-passage-corpus-with-query)
and writes them to a single JSONL file, one row per line:

```json
{"pid": 0, "text": "...", "queries": ["...", "..."]}
```

`text` is the passage that gets embedded and indexed; `queries` are that
passage's real, associated search queries, later used by eval-harness to
build eval query sets — no synthetic/LLM-generated queries anywhere in this
pipeline. This is the only corpus asset the rest of the pipeline needs.

### More datasets: BeIR

`packages/download-beir` builds the same `{pid, text, queries}` JSONL shape
from any `BeIR/<name>-generated-queries` dataset on the Hub:

```bash
cd packages/download-beir
uv run download-beir dbpedia-entity-generated-queries          # writes ../../data/dbpedia-entity/corpus.jsonl
uv run download-beir scifact-generated-queries 15422           # optional 2nd arg caps how many rows are pulled
```

It writes to `../../data/<name>/corpus.jsonl`, where `<name>` is the dataset
name with any trailing `-generated-queries` stripped (so
`dbpedia-entity-generated-queries` → `data/dbpedia-entity/`,
`scifact-generated-queries` → `data/scifact/`). 

For each dataset, `packages/qdrant-load/configs/<name>/` mirrors every config
in `packages/qdrant-load/configs/` (same quantization schemes,
`collection_name` prefixed `<name>_` instead of `test_`, corpus path pointing
at `data/<name>/corpus.jsonl`). `run_sweep.sh`'s default glob only picks up
the top-level `configs/*.yml`, so running it as-is never touches these — point
it there explicitly with a separate `OUTPUT_DIR`:

```bash
OUTPUT_DIR=../../results/dbpedia RESCORERS="colbert,cross-encoder,rrf" \
  ./run_sweep.sh ../qdrant-load/configs/dbpedia/*.yml

# scifact's corpus is much smaller (15.4K passages vs. 100K) -- fewer queries
# is a reasonable call there:
N_QUERIES=100 OUTPUT_DIR=../../results/scifact RESCORERS="colbert,cross-encoder,rrf" \
  ./run_sweep.sh ../qdrant-load/configs/scifact/*.yml
```

To add another dataset, follow the same pattern: `download-beir
<name>-generated-queries`, a `configs/<name>/` directory of mirrored configs,
and a `results/<name>/` output directory.

## 2. Configs

Each YAML file under `packages/qdrant-load/configs/` describes one collection
to benchmark: a `collection_name`, the corpus path, and per-vector-type
quantization settings for `dense_vectors`, `colbert_vectors`, and
`sparse_vectors`. The embedding models themselves are fixed and not
config-driven:

| Vector | Model |
|---|---|
| dense | `sentence-transformers/all-minilm-l6-v2` |
| sparse | `Qdrant/bm25` |
| colbert | `answerdotai/answerai-colbert-small-v1` |

Add a new config by copying an existing one and changing `collection_name`
plus the `dense_vectors` / `colbert_vectors` / `sparse_vectors` blocks — see
the [Qdrant quantization docs](https://qdrant.tech/documentation/manage-data/quantization/)
for the available `quantization_config` / `dtype` combinations.

To upload a single config by hand (mostly for debugging, `eval-harness`
below does this for you as part of a benchmark run):

```bash
cd packages/qdrant-load
uv run qdrant-load configs/config_binary.yml
```

## 3. Run the benchmark

```bash
cd packages/eval-harness
uv run eval-harness ../qdrant-load/configs/config_binary.yml \
  --n-queries 500 --seed 42 \
  --k 10,20 \
  --prefetch-limit 25,50,100 \
  --modes hybrid,sparseonly \
  --rescorers colbert,cross-encoder,rrf \
  --output-dir ../../results
```

For a given config, `eval-harness`:

1. Uploads the corpus to a fresh collection **once** (`qdrant-load` under the
   hood).
2. Samples `--n-queries` real queries from the corpus (seeded, so the same
   sample is reused across runs).
3. Runs every combination of `--modes` × `--rescorers` × `--k` ×
   `--prefetch-limit` against that single upload, measuring recall/mrr/ndcg
   and sequential + concurrent latency for each.
4. Writes one JSON report per combination to `--output-dir`, then deletes the
   collection.

**Modes** (`--modes`, comma-separated): `hybrid` (dense + sparse prefetch,
RRF-fused) or `sparseonly` (sparse prefetch only).

**Rescorers** (`--rescorers`, comma-separated):
- `colbert` — server-side ColBERT MaxSim rescore of the prefetched candidates
- `cross-encoder` — local rerank via fastembed's `TextCrossEncoder`
  (`Xenova/ms-marco-MiniLM-L-6-v2`) over the prefetched candidates' text
- `rrf` — no second stage at all; the prefetch's RRF fusion *is* the result
  (requires `hybrid` mode — RRF needs ≥2 sources to fuse, so `rrf` +
  `sparseonly` combinations are skipped automatically)

Combinations where `prefetch_limit < k` are skipped automatically.

### Resuming an interrupted run

A sweep is resumable by default: `eval-harness` skips any combination that
already has a report file in `--output-dir`, and reuses an existing
collection instead of re-uploading if one is already there (e.g. left behind
by an interrupted run) — just re-run the same command to pick up where it
left off. Use `--overwrite` to force redoing combinations that already have a
report, and `--fresh` to force a clean delete + re-upload of the collection
first.

### Sweeping multiple configs: `run_sweep.sh`

```bash
cd packages/eval-harness
./run_sweep.sh                                    # every config in ../qdrant-load/configs
./run_sweep.sh ../qdrant-load/configs/config_binary.yml   # just one
```

A thin wrapper that calls `eval-harness` once per config (each config is
still uploaded only once). The grid is controlled by environment variables,
all with sensible defaults:

| Variable | Default |
|---|---|
| `N_QUERIES` | `500` |
| `SEED` | `42` |
| `K_VALUES` | `10,20` |
| `PREFETCH_VALUES` | `25,50,100` |
| `PREFETCH_MODES` | `hybrid,sparseonly` |
| `RESCORERS` | `colbert` |
| `WARMUP` | `10` |
| `CONCURRENCY` | `8` |
| `OUTPUT_DIR` | `../../results` |
| `OVERWRITE` | `0` (set `1` to force `--overwrite`) |
| `FRESH` | `0` (set `1` to force `--fresh`) |

```bash
RESCORERS="colbert,cross-encoder,rrf" ./run_sweep.sh
```

## 4. Summarize results

```bash
cd packages/eval-harness
python3 summarize_results.py ../../results
```

Reads every `results/*.json` report and writes `results/summary.csv` (one row
per config × mode × rescorer × k × prefetch_limit), also printing the same
table to stdout.

### Report JSON shape

Each `results/<collection>_k<k>_pf<prefetch_limit>_<mode>_<rescorer>.json`
looks like:

```json
{
  "collection_name": "test_binary",
  "config": "../qdrant-load/configs/config_binary.yml",
  "n_queries": 500,
  "k": 20,
  "prefetch_limit": 100,
  "use_dense_prefetch": true,
  "rescorer": "colbert",
  "concurrency": 8,
  "quality": {"recall@k": 0.97, "hit_rate@k": 0.97, "mrr@k": 0.71, "ndcg@k": 0.77},
  "latency": {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0},
  "throughput_qps": 60.9
}
```

`quality` is computed against each query's real source passage as the single
ground-truth relevant document (binary relevance). `latency` is measured
sequentially, one query at a time, after `--warmup` queries are discarded;
`throughput_qps` is measured separately with `--concurrency` concurrent
requests.
