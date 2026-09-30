#!/usr/bin/env bash
# Run the LayerMem LoCoMo pipeline end to end: build -> search -> evaluate.
#
# Prerequisites (see LayerMem_imple.md appendix A and README of scripts/):
#   1. LayerMem/.venv           — pip install -e ".[dev]" plus litellm
#   2. LayerMem/.venv-serve     — mlx-lm, serving the model on :8080
#   3. benchmarks/locomo/locomo10.json
#
#   .venv-serve/bin/mlx_lm.server --model "$PWD/models/Qwen3-1.7B" \
#       --host 127.0.0.1 --port 8080 \
#       --chat-template-args '{"enable_thinking": false}'
#
# Usage:  ./run_layermem_locomo.sh [start_idx] [end_idx]
set -euo pipefail

START_IDX="${1:-0}"
END_IDX="${2:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKITS="$ROOT/memory_toolkits"
PY="$ROOT/.venv/bin/python"
MODEL="$ROOT/models/Qwen3-1.7B"
LOCOMO="$ROOT/benchmarks/locomo/locomo10.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
# litellm otherwise tries to fetch a price table from GitHub on every import.
export LITELLM_LOCAL_MODEL_COST_MAP=True

cd "$TOOLKITS"

echo "==> [1/3] building memory for conversations [$START_IDX, $END_IDX)"
"$PY" memory_construction.py \
    --memory-type LayerMem \
    --dataset-type LoCoMo \
    --dataset-path "$LOCOMO" \
    --config-path configs/LayerMem.json \
    --num-workers 1 \
    --start-idx "$START_IDX" --end-idx "$END_IDX" \
    --rerun \
    --message-preprocessor "memories.layers.layermem:locomo_message_preprocessor"

echo "==> [2/3] retrieving memories for the same range"
"$PY" memory_search.py \
    --memory-type LayerMem \
    --dataset-type LoCoMo \
    --dataset-path "$LOCOMO" \
    --config-path configs/LayerMem.json \
    --num-workers 1 \
    --top-k 10 \
    --start-idx "$START_IDX" --end-idx "$END_IDX"

echo "==> [3/3] answering and judging"
# The toolkit names the artefact "{layer}_{llm_model}_{dataset}_{top_k}_{start}_{end}.json"
# and llm_model is an absolute path, so the "filename" is really a nested path
# under LayerMem_/ — hence find rather than a shell glob.
RESULT_JSON="$(find LayerMem_ -type f -name "*_LoCoMo_*_${START_IDX}_${END_IDX}.json" \
    -not -name "*_evaluation.json" -print0 2>/dev/null \
    | xargs -0 ls -t 2>/dev/null | head -1)"
if [ -z "$RESULT_JSON" ]; then
    echo "search results file not found" >&2
    exit 1
fi
"$PY" memory_evaluation.py \
    --search-results-path "$RESULT_JSON" \
    --dataset-type LoCoMo \
    --qa-model "$MODEL" \
    --judge-model "$MODEL" \
    --qa-batch-size 1 \
    --judge-batch-size 1 \
    --api-config-path configs/api_config_local.json

echo "==> done: ${RESULT_JSON%.json}_evaluation.json"
