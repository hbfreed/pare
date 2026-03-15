#!/usr/bin/env bash
set -euo pipefail

# Evaluate a model checkpoint with lm-eval via uvx.
# Usage: ./eval.sh <model_path_or_hf_id> [extra lm_eval args...]
#
# Examples:
#   ./eval.sh checkpoints/onpolicy-from-baseline/latest
#   ./eval.sh hbfreed/pruned_olmo3_4096_16_29_distilled_on_policy
#   ./eval.sh checkpoints/latest --tasks gsm8k_cot --limit 100

MODEL="${1:?Usage: $0 <model_path_or_hf_id> [extra args...]}"
shift

uvx --with "lm-eval[vllm]" lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks gsm8k_cot,hellaswag,arc_easy,truthfulqa_mc2 \
    --batch_size auto \
    "$@"
