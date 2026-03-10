# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## 2026-03 (March 2026)

### Added

- Add on-policy HF checkpoint pushing, setup script, and pipeline cleanup (`fba0c01`)
- Add enforce_eager=True to vLLM to skip CUDA graph capture (`ffaf712`)
- Add teacher/student GPU overlap via streaming queue for world_size==1 (`b210ef5`)

### Changed

- Use prebuilt flash-attn wheel to skip CUDA compilation (`4e137b7`)
- Disable mid-training evals (lm-eval/datasets incompatibility) (`1cbda73`)
- Increase NCCL timeout to 30min for vLLM startup (`9533929`)
- Clear DDP env vars before vLLM init to prevent subprocess deadlock (`7900054`)
- Init vLLM before DDP to avoid CUDA subprocess deadlock (`ebbbde7`)
- Clear torchrun env vars before vLLM init to prevent EngineCore deadlock (`7f5bfd1`)
- Force vLLM V0 engine to avoid subprocess deadlock with torchrun (`52724a0`)
- Increase vLLM GPU memory utilization to 0.98 for more KV cache (`97e06b9`)
- Chunk weight sync to avoid msgspec 4GB encoding limit (`c512190`)
- Set MAX_CONTEXT_LENGTH to 2048 for cluster deployment (`c171672`)

### Fixed

- Fix vLLM+torchrun deadlock and prepare for cluster deployment (`7b84889`)

### Removed

- Remove enforce_eager=True to enable CUDA graph capture for faster vLLM inference (`9daefd4`)

## 2026-01 (January 2026)

### Added

- added better checkpointing and resuming for when training gets cut off (`970b49b`)

### Changed

- first commit, about to prune (`9d41f19`)
- Update .gitignore (`b86909b`)
- figured out pruning, wrote distill_off_policy (`dc953dd`)
- figured out pruning, wrote distill_off_policy (`6b16f66`)
- getting ready to push for benching (`d9c1dbc`)
- Update pyproject.toml (`faa67f1`)
- Update README and project description (`d8f0d42`)
