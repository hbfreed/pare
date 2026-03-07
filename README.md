# PARE

Pruning and knowledge distillation for large language models. Creates smaller, faster models from larger ones while preserving quality.

Based on NVIDIA's Minitron papers:
- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796)

## Overview

PARE takes a teacher model (Olmo 3 7B Instruct) and produces compressed student models through:

1. **Importance analysis** - Compute per-neuron, per-head, and per-layer importance scores
2. **Pruning** - Remove low-importance neurons, attention heads, and layers
3. **Off-policy distillation** - Train the pruned model on pre-generated teacher completions using KL divergence (comparison baseline)
4. **On-policy distillation** - Generate student rollouts, score with teacher, train with clipped policy gradient loss. Supports multi-GPU via DDP

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

### 3. Off-policy distillation (baseline)

Generate teacher completions with vLLM, then train:

```bash
# Generate completions across 3 GPUs
python generate_off_policy_completions.py --rank 0 &
python generate_off_policy_completions.py --rank 1 &
python generate_off_policy_completions.py --rank 2 &

# Train on pre-generated data
uv run distill_off_policy.py
```

### 4. On-policy distillation (main)

```bash
# Single-node DDP (GPU 0: vLLM gen, GPU 1: teacher, GPUs 2+: DDP training)
torchrun --nproc_per_node=2 distill_on_policy.py
```

Or use the launch script:
```bash
bash launch_ddp.sh
```

## Project Structure

```
pare/
├── importance_analysis.py              # Importance score computation
├── prune.py                            # Model pruning
├── generate_off_policy_completions.py  # Teacher completion generation (vLLM offline)
├── distill_off_policy.py               # Off-policy distillation training
├── distill_on_policy.py                # On-policy distillation with DDP
├── evals.py                            # Evaluation harness
├── launch_ddp.sh                       # DDP launch helper
├── build_distill_dataset.py            # One-shot: built hbfreed/Dolci-Instruct-RL-Completions
├── finalize_distill_dataset.py         # One-shot: built hbfreed/dolci-distill-packed
├── generate_logprobs_hf.py             # One-shot: teacher logprob extraction
├── pruned_models/                      # Pruned model outputs
├── importance_scores_tensors/          # Cached importance scores
└── checkpoints/                        # Training checkpoints
```

Note: `build_distill_dataset.py`, `generate_logprobs_hf.py`, and `finalize_distill_dataset.py` are one-shot pipeline scripts that produced the finalized datasets on HuggingFace Hub ([hbfreed/Dolci-Instruct-RL-Completions](https://huggingface.co/datasets/hbfreed/Dolci-Instruct-RL-Completions), [hbfreed/dolci-distill-packed](https://huggingface.co/datasets/hbfreed/dolci-distill-packed)).

## Design Notes

Key choices from the Minitron papers:
- Width pruning preferred over depth for models under 15B parameters
- Single-shot importance estimation (iterative provides no benefit)
- KL divergence loss for distillation instead of conventional training
- Full attention layers protected when pruning sliding window attention models
