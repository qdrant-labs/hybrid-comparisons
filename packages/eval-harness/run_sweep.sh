#!/bin/bash
# Sweep the eval harness across all qdrant-load configs. Each config is
# uploaded to its own collection exactly once; eval-harness itself sweeps
# every (mode, rescorer, k, prefetch_limit) combination against that single
# upload before the collection is deleted.
#
# Modes:
#   - hybrid:     dense + sparse prefetch -> RRF fusion -> rescore
#   - sparseonly: sparse-only prefetch -> rescore
# Rescorers:
#   - colbert:       server-side colbert MaxSim rescore
#   - cross-encoder: local fastembed cross-encoder rerank of the prefetched candidates
#   - rrf:           no second stage at all -- the dense+sparse RRF fusion is the final
#                     result. Only valid with mode=hybrid (eval-harness skips it otherwise).
#
# Resumable: eval-harness skips any (mode, rescorer, k, prefetch_limit)
# combination that already has a report in --output-dir, and reuses an
# existing collection instead of re-uploading -- so if a run gets interrupted
# (e.g. a network blip mid-sweep), just re-run this script to pick up where
# it left off. Pass OVERWRITE=1 / FRESH=1 to opt out of that.
#
# Usage:
#   ./run_sweep.sh [config1.yml config2.yml ...]
#
# If no configs are passed, all configs in ../qdrant-load/configs are used.
# Override the grid with env vars, e.g.:
#   K_VALUES="5,10" PREFETCH_VALUES="20,50" PREFETCH_MODES="hybrid" RESCORERS="colbert,cross-encoder,rrf" ./run_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="../qdrant-load/configs"
N_QUERIES="${N_QUERIES:-500}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-../../results}"
WARMUP="${WARMUP:-10}"
CONCURRENCY="${CONCURRENCY:-8}"

K_VALUES="${K_VALUES:-10,20}"
PREFETCH_VALUES="${PREFETCH_VALUES:-25,50,100}"
PREFETCH_MODES="${PREFETCH_MODES:-hybrid,sparseonly}"
RESCORERS="${RESCORERS:-colbert}"
OVERWRITE="${OVERWRITE:-0}"
FRESH="${FRESH:-0}"

OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"
FRESH_FLAG=""
[ "$FRESH" = "1" ] && FRESH_FLAG="--fresh"

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("$CONFIG_DIR"/*.yml)
fi

echo "Configs: ${CONFIGS[*]}"
echo "k values: $K_VALUES"
echo "prefetch_limit values: $PREFETCH_VALUES"
echo "prefetch modes: $PREFETCH_MODES"
echo "rescorers: $RESCORERS"
echo

for cfg in "${CONFIGS[@]}"; do
    echo "=== $cfg (single upload, sweeping all settings) ==="
    eval-harness "$cfg" \
        --n-queries "$N_QUERIES" \
        --seed "$SEED" \
        --k "$K_VALUES" \
        --prefetch-limit "$PREFETCH_VALUES" \
        --modes "$PREFETCH_MODES" \
        --rescorers "$RESCORERS" \
        --warmup "$WARMUP" \
        --concurrency "$CONCURRENCY" \
        --output-dir "$OUTPUT_DIR" \
        $OVERWRITE_FLAG $FRESH_FLAG
    echo
done

echo "Sweep complete. Reports in $OUTPUT_DIR"
