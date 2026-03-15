#!/usr/bin/env bash
set -euo pipefail

# Install nvtop for GPU monitoring
sudo apt-get install -y nvtop

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies (including vllm, cloudpickle, lm-eval)
uv sync --extra inference

# --- vLLM wheel caching ---
# Building vLLM from source takes ~20 min on cluster nodes. To skip this on
# future runs, after the first successful `uv sync`:
#   1. Find the built wheel: ls .venv/lib/python*/site-packages/vllm*.dist-info/WHEEL
#      or check `uv pip show vllm` for the installed version
#   2. Build a wheel: cd .venv/src/vllm && pip wheel --no-deps -w /tmp/wheels .
#   3. Upload: gh release upload v0.1 /tmp/wheels/vllm-*.whl
#   4. Add to pyproject.toml under [tool.uv.sources]:
#      vllm = { url = "https://github.com/hbfreed/pare/releases/download/v0.1/vllm-....whl" }

# Login to services (use env vars for non-interactive; falls back to interactive)
if [ -n "${HF_TOKEN:-}" ]; then
    uv run huggingface-cli login --token "$HF_TOKEN"
else
    uv run huggingface-cli login
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login "$WANDB_API_KEY"
else
    uv run wandb login
fi

# Launch training (2 DDP ranks by default, pass extra args through)
uv run bash launch_ddp.sh "$@"
