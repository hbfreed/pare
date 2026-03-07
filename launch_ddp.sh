#!/usr/bin/env bash
# Launch DDP on-policy distillation.
# Usage: uv run bash launch_ddp.sh [nproc] [extra args...]
#   First arg sets nproc_per_node (default: 2).
#
# Examples:
#   uv run bash launch_ddp.sh 2 --micro-batch-size 32 --teacher-micro-batch-size 32 --lr 1e-5
#   uv run bash launch_ddp.sh 1 --sweep 2 --micro-batch-size 4  # smoke test with 1 rank

NPROC="${1:-2}"
shift 2>/dev/null || true

torchrun --nproc_per_node="$NPROC" distill_on_policy.py "$@"
