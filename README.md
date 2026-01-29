# PARE

Pruning and knowledge distillation for large language models. Creates smaller, faster models from larger ones while preserving quality.

Based on NVIDIA's Minitron papers:
- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796)

## Overview

PARE takes a teacher model (Olmo 3 7B Instruct) and produces compressed student models through:

1. **Importance analysis** - Compute per-neuron, per-head, and per-layer importance scores
2. **Pruning** - Remove low-importance neurons, attention heads, and layers
3. **Distillation** - Train the pruned model to match teacher behavior using KL divergence

## Requirements

- Python 3.12
- CUDA-capable GPU with Flash Attention support

Install dependencies:
```bash
uv sync
```

## Usage

### 1. Compute importance scores

```bash
uv run importance_analysis.py
```

Analyzes the teacher model using 1024 calibration samples. Outputs to `importance_scores_tensors/`:

| Key | Shape | Description |
|-----|-------|-------------|
| `mlp` | `[n_layers, intermediate_size]` | Per-neuron importance (L2 norm of activations) |
| `attention` | `[n_layers, n_heads]` | Per-head importance |
| `attn_ln` | `[hidden_size]` | Aggregated attention layer norm importance |
| `ffn_ln` | `[hidden_size]` | Aggregated FFN layer norm importance |
| `layer` | `[n_layers]` | Cosine-similarity depth scores for layer pruning |

### 2. Prune the model

```bash
uv run prune.py
```

Applies width and depth pruning based on importance scores. Configure target dimensions in the script:
- `mlp_width`: FFN intermediate size
- `num_heads`: Attention heads
- `num_layers`: Transformer layers

Saves pruned model to `pruned_models/`.

### 3. Build distillation dataset

```bash
uv run build_distill_dataset.py
uv run generate_logprobs_hf.py --batch-size 1 --start-idx 0
uv run finalize_distill_dataset.py
```

### 4. Train the student model

```bash
uv run distill_off_policy.py
```

Runs off-policy distillation with KL divergence loss. Checkpoints automatically to HuggingFace Hub and logs to Weights & Biases. Resumes from latest checkpoint if interrupted.

## Project Structure

```
pare/
├── importance_analysis.py      # Importance score computation
├── prune.py                    # Model pruning
├── build_distill_dataset.py    # Dataset construction
├── finalize_distill_dataset.py # Dataset finalization
├── generate_logprobs_hf.py     # Teacher logprob extraction
├── distill_off_policy.py       # Distillation training
├── pruned_models/              # Pruned model outputs
├── importance_scores_tensors/  # Cached importance scores
└── cache/                      # Pre-packed training data
```

## Design Notes

Key choices from the Minitron papers:
- Width pruning preferred over depth for models under 15B parameters
- Single-shot importance estimation (iterative provides no benefit)
- KL divergence loss for distillation instead of conventional training
- Full attention layers protected when pruning sliding window attention models
