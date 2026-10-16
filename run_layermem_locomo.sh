#!/usr/bin/env bash
# Run the LayerMem LoCoMo pipeline end to end: build -> search -> evaluate.
#
# Prerequisites (see LayerMem_imple.md appendix A and README of scripts/):
#   1. LayerMem/.venv           — pip install -e ".[dev]" plus litellm
#   2. Export OPENAI_API_KEY in the calling terminal.
#   3. benchmarks/locomo/locomo10.json
#
#
# Usage: ./run_layermem_locomo.sh [start_idx] [end_idx] [llm_profile]
# Profile: gpt-4o-mini, defined in configs/LayerMem.json.
# Example: ./run_layermem_locomo.sh 0 1 gpt-4o-mini
set -euo pipefail

START_IDX="${1:-0}"
END_IDX="${2:-1}"
LLM_PROFILE="${3:-gpt-4o-mini}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKITS="$ROOT/memory_toolkits"
PY="$ROOT/.venv/bin/python"
LOCOMO="$ROOT/benchmarks/locomo/locomo10.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
# litellm otherwise tries to fetch a price table from GitHub on every import.
export LITELLM_LOCAL_MODEL_COST_MAP=True

cd "$TOOLKITS"

RESULT_JSON="results/LayerMem_${LLM_PROFILE}_LoCoMo_10_${START_IDX}_${END_IDX}.json"
echo "==> LLM profile: $LLM_PROFILE (construction, search, QA and judge)"

echo "==> [1/3] building memory for conversations [$START_IDX, $END_IDX)"
"$PY" memory_construction.py \
    --memory-type LayerMem \
    --dataset-type LoCoMo \
    --dataset-path "$LOCOMO" \
    --config-path configs/LayerMem.json \
    --llm-profile "$LLM_PROFILE" \
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
    --llm-profile "$LLM_PROFILE" \
    --output-path "$RESULT_JSON" \
    --num-workers 1 \
    --top-k 10 \
    --start-idx "$START_IDX" --end-idx "$END_IDX"

echo "==> [3/3] answering and judging"
if [ ! -f "$RESULT_JSON" ]; then
    echo "search results file not found" >&2
    exit 1
fi
"$PY" memory_evaluation.py \
    --search-results-path "$RESULT_JSON" \
    --dataset-type LoCoMo \
    --config-path configs/LayerMem.json \
    --llm-profile "$LLM_PROFILE" \
    --qa-batch-size 1 \
    --judge-batch-size 1

echo "==> done: ${RESULT_JSON%.json}_evaluation.json"
