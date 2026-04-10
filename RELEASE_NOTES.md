# Release Notes — March 2026

Cross-project release notes covering all active `hbfreed` repositories for March 2026.

---

## pare — Pruning and Knowledge Distillation

16 commits | [github.com/hbfreed/pare](https://github.com/hbfreed/pare)

### Cluster Deployment & vLLM Integration

The major theme this month was getting on-policy distillation running reliably on multi-GPU clusters with vLLM as the generation backend.

- **vLLM + torchrun deadlock resolution** — Resolved a series of CUDA subprocess deadlocks caused by DDP environment variables leaking into vLLM's EngineCore subprocess. The final fix clears all torchrun env vars by prefix before vLLM init and sets CUDA device before DDP init to prevent NCCL collisions. (`7b84889`, `ebbbde7`, `7f5bfd1`, `7900054`, `52724a0`)
- **Teacher/student GPU overlap** — Added a streaming queue architecture where the teacher runs in a background thread producing micro-batches while the student consumes concurrently, enabling real GPU overlap for `world_size==1`. (`b210ef5`)
- **Chunked weight sync** — Full state dict serialization exceeds msgspec's 4GB encoding limit; split into 2GB chunks via separate `collective_rpc` calls. (`c512190`)
- **CUDA graph capture for vLLM** — Removed `enforce_eager=True` to enable CUDA graph capture, improving inference throughput. (`9daefd4`)
- **One-command cluster bootstrap** — Added `setup_and_run.sh` for single-command cluster setup, plus HF Hub checkpoint pushing for full remote resume. (`fba0c01`)
- **vLLM memory utilization** — Bumped GPU memory utilization to 0.98 for maximum KV cache. (`97e06b9`)

### Evaluation & Cleanup

- **Eval harness** — Added `eval.sh` for running lm-eval benchmarks against trained models. (`a95f336`)
- **Dead script removal** — Removed one-off data processing scripts (`process_completions.py`, `process_logprobs.py`) and cleaned up commented-out code. (`a95f336`)
- **Context length tuning** — Set `MAX_CONTEXT_LENGTH` to 2048 for cluster deployment. (`c171672`)

---

## nanoMoEchat — MoE Transformer Training

12 commits | [github.com/hbfreed/nanoMoEchat](https://github.com/hbfreed/nanoMoEchat)

### Upstream Sync (Phases 1–4)

Ported four phases of improvements from the upstream nanochat project:

- **Phase 1: Infrastructure** — Modernized TF32 precision API, added `COMPUTE_DTYPE` auto-detection, added `get_peak_flops()` GPU table for accurate MFU reporting, improved DDP detection. (`714aa4c`)
- **Phase 2: Gradient clipping removal** — Upstream proved grad clipping costs ~2% MFU and was buggy (clipped per-GPU before DDP sync). (`ce524c6`)
- **Phase 3: Argparse migration** — Migrated all training scripts (`base_train`, `mid_train`, `chat_sft`, `chat_rl`) from `configurator.py` to argparse. Deleted autocast in favor of direct dtype management. (`a2c25fc`)
- **Phase 4: Flash Attention 3 & sliding window** — Added `nanochat/flash_attention.py` with FA3 auto-detection on Hopper GPUs (SDPA fallback elsewhere). Added sliding window attention via `window_pattern` config. Rewrote KVCache for flash_attn API. (`90f981d`)

### MoE Improvements

- **SMEBU bias balancing** — Replaced binary bias update with Soft-clamped Momentum Expert Bias Updates (Trinity, 2025). The old binary ±gamma updates caused biases to grow unboundedly (~45 after 4500 steps), drowning out router preferences. SMEBU uses magnitude-aware tanh updates with zero-centering. (`84738fa`)
- **Sequence-level load balance loss** — DeepSeek V3-style per-sequence load balance loss instead of per-batch. Handles both 1D (batch) and 2D (sequence) inputs via shape-agnostic ops. Added variable-size expert experiment configs (32x128 + 32x640). (`b23e153`)
- **Cut Cross Entropy** — Replaced `F.cross_entropy` with Apple's CCE, fusing lm_head projection + softcap + cross entropy into one kernel. Saves ~4-5GB VRAM per micro-batch. Switched to `cce_kahan_full_c` variant for pretraining numerical accuracy. (`b4c30c9`, `1addfc1`)

### Bug Fixes

- **Dead MoE buffers** — Non-persistent buffers (`expert_size_blocks`, `group_membership`, etc.) were all zeros after meta device → `to_empty` pipeline. Added `_init_buffers()` called from `init_weights`. Also kept `expert_bias` in float32 to prevent bf16 ULP (~0.004) from swallowing small bias updates. (`c798dee`)

---

## on-policy-olmo — On-Policy Distillation for OLMo

6 commits | [github.com/hbfreed/on-policy-olmo](https://github.com/hbfreed/on-policy-olmo)

### New Capabilities

- **OLMo 3 teacher** — Upgraded teacher model from OLMo-2-7B to OLMo-3-7B-Instruct. Added benchmark evals (gsm8k, arc, truthfulqa, ifeval) with async execution, argparse CLI, rolling checkpoint rotation, and torch.compile for student and teacher. (`0f4850f`)
- **DDP and SFT support** — Extracted shared utilities into `distill_utils.py`, added DDP-capable on-policy distillation, added SFT training pipeline with full README rewrite. (`dd666b5`)
- **Fused logprobs** — Replaced `log_softmax + gather` with `F.cross_entropy` for logprob computation, avoiding full `[B, T, 100K]` tensor materialization (~1.5 GiB savings). Added buffered teacher pipeline for flexible batch sizes. (`3b097dd`)

### Performance Optimizations

- **SFT training optimization** — CCE mode: lm_head pre-hook captures hidden state and feeds 1-token dummy, skipping the full `[B*T, H] @ [H, V]` matmul. KLD mode: pipelined teacher extraction (step N+1 prefetched async on teacher GPU). Removed per-micro-batch `.item()` sync points. Added async checkpoint saving. (`6240871`)
- **Rank-local memmap loading** — Each rank loads only its local shard from memmap instead of the full batch. (`bf19f9b`)

### Bug Fixes

- **Eval crash fix** — Added missing `mixed_precision_dtype` attribute to `HFFromExisting` for newer `lm_eval` compatibility. (`3828273`)

---

## nanodistill — Knowledge Distillation

1 commit | [github.com/hbfreed/nanodistill](https://github.com/hbfreed/nanodistill)

### Features

- **Muon + CCE-exact training** — Added MuonAdamW optimizer with Newton-Schulz5 orthogonalization (from nanochat), Muon Split (GLM-5) for per-head Q/K/V orthogonalization, and `cce_exact` implementation with Kahan summation fixing bf16 accumulation bugs in CCE v25.1.1. Restored learnable RMSNorm required for OLMo3's post-norm architecture. Added 3x3 hyperparameter sweep infrastructure and DDP support. (`ded23be`)

---

## variable-flex-olmo — Variable Expert MoE with OLMo

1 commit | [github.com/hbfreed/variable-flex-olmo](https://github.com/hbfreed/variable-flex-olmo)

### Bug Fixes

- **Expert unfreezing fix** — `load_pruned_model` was refactored to use `from_pretrained` which freezes all params after loading. Expert 1 now explicitly unfrozen before distillation training. (`186b051`)

---

## Summary

| Project | Commits | Highlights |
|---------|---------|------------|
| pare | 16 | vLLM cluster deployment, teacher/student GPU overlap, eval harness |
| nanoMoEchat | 12 | Upstream sync (FA3, argparse, COMPUTE_DTYPE), SMEBU, CCE, sequence-level LBL |
| on-policy-olmo | 6 | OLMo 3 teacher, DDP+SFT, fused logprobs, async pipelining |
| nanodistill | 1 | Muon+CCE-exact training with sweep infrastructure |
| variable-flex-olmo | 1 | Expert unfreezing bugfix |
