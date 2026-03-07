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

# Login to services
uv run huggingface-cli login
uv run wandb login

# Launch training (2 DDP ranks by default, pass extra args through)
uv run bash launch_ddp.sh "$@"
