'''Distilling training loop, using KL Divergence (with packing).'''
import torch
import torch.nn.functional as F
from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerKLDIVLoss
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer
import wandb
from collections import deque
import time
import random
import os
import numpy as np
from pathlib import Path
import bitsandbytes as bnb

# Low-hanging fruit for torch performance
# NOTE: torch.compile requires the new TF32 API (not the old fp32_precision='tf32')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "pruned_models/pruned_olmo3_4096_16_29"
wandb_project = "olmo-distillation"
out_dir = f"checkpoints/{STUDENT.split('/')[1]}"
compile = False  # TODO: fullgraph=True conflicts with flash_attention's data-dependent branching
use_fsdp = True  # Use FSDP instead of DDP for memory efficiency

batch_size = 1 
gradient_accumulation_steps = 2  # must be divisible by world_size
n_epochs = 100
overfit_samples = None  # Set to None for full dataset
preprocess_only = False  # Set True to pack data and exit, False to train
# NOTE: If memory-bound, consider gradient_checkpointing=True in from_pretrained()
# trades ~33% more compute for much lower memory, enabling larger batch sizes  

CHUNK_SIZE = 30000  # Process in smaller chunks to avoid OOM during collating
CACHE_FORMAT_VERSION = 3  # Bump if packed cache layout changes
SAVE_EVERY = 64  # Number of packed sequences to buffer per save

# Store indices in int32 on disk to cut CPU RAM roughly in half.
# Cast to int64 right before gather/embedding on device.
ID_DTYPE = torch.int32
POS_DTYPE = torch.int32
IDX_DTYPE = torch.int32

# Data is pre-packed and uploaded to HF Hub. Download with:
# from huggingface_hub import snapshot_download
# snapshot_download("hbfreed/dolci-distill-packed", repo_type="dataset", local_dir="cache/chunks")
CHUNKS_DIR = "cache/chunks"

# DDP Setup
backend = 'nccl'
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # Explicitly specify device to avoid NCCL hangs with heterogeneous GPU mapping
    init_process_group(backend=backend, device_id=torch.device(device))
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    device = 'cuda'

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1223 + seed_offset)

def tensorize(dataset, max_length=None):
    """Tensorize dataset with explicit dtypes for bf16 training."""
    tensorized = []
    max_len = 0
    skipped = 0
    skipped_long = 0
    # Check if dataset has teacher_logits (new) or teacher_logprobs (legacy)
    has_logits = 'teacher_logits' in dataset.column_names
    logits_key = 'teacher_logits' if has_logits else 'teacher_logprobs'
    for example in tqdm(dataset, desc="Tensorizing"):
        ids = example['input_ids_prompt'] + example['input_ids_completion']
        length = len(ids)
        teacher_len = len(example['teacher_indices'])

        # Skip mismatched examples (data bug)
        if length != teacher_len:
            skipped += 1
            continue

        # Skip examples longer than pack_length
        if max_length and length > max_length:
            skipped_long += 1
            continue

        max_len = max(max_len, length)
        tensorized.append({
            'input_ids': torch.tensor(ids, dtype=ID_DTYPE),
            'position_ids': torch.arange(length, dtype=POS_DTYPE),
            'teacher_indices': torch.tensor(example['teacher_indices'], dtype=IDX_DTYPE),
            'teacher_logits': torch.tensor(example[logits_key], dtype=torch.bfloat16),
            'length': length,
        })
    if skipped > 0:
        print(f"  (skipped {skipped} examples due to length mismatch)")
    if skipped_long > 0:
        print(f"  (skipped {skipped_long} examples exceeding max_length {max_length})")
    return tensorized, max_len

def round_to_multiple(x, multiple=128):
    return ((x + multiple - 1) // multiple) * multiple

def iter_packed_groups(examples, pack_length):
    # O(n) greedy packing after sort: take largest, then fill with smallest that fit.
    # This preserves the prior "largest + smallest" greedy strategy but avoids O(n^2) scans.
    sorted_examples = deque(sorted(examples, key=lambda x: x['length']))
    while sorted_examples:
        current_batch = [sorted_examples.pop()]  # largest remaining
        current_len = current_batch[0]['length']

        # Greedily add smallest that still fit. If the smallest doesn't fit, none will.
        while sorted_examples and current_len + sorted_examples[0]['length'] <= pack_length:
            ex = sorted_examples.popleft()
            current_batch.append(ex)
            current_len += ex['length']

        yield current_batch, current_len


def collate_packed_group(batch_examples, total_len, pack_length, pad_token_id=0):
    k = batch_examples[0]['teacher_indices'].shape[1]
    input_ids = torch.full((pack_length,), pad_token_id, dtype=batch_examples[0]['input_ids'].dtype)
    position_ids = torch.zeros((pack_length,), dtype=batch_examples[0]['position_ids'].dtype)
    teacher_indices = torch.zeros((pack_length, k), dtype=batch_examples[0]['teacher_indices'].dtype)
    teacher_logits = torch.zeros((pack_length, k), dtype=batch_examples[0]['teacher_logits'].dtype)

    offset = 0
    for ex in batch_examples:
        length = ex['length']
        end = offset + length
        input_ids[offset:end] = ex['input_ids']
        position_ids[offset:end] = ex['position_ids']
        teacher_indices[offset:end] = ex['teacher_indices']
        teacher_logits[offset:end] = ex['teacher_logits']
        offset = end

    pad_mask = torch.zeros(pack_length, dtype=torch.bool)
    pad_mask[:total_len] = True

    return {
        'input_ids': input_ids[None],
        'position_ids': position_ids[None],
        'teacher_indices': teacher_indices[None],
        'teacher_logits': teacher_logits[None],
        'pad_mask': pad_mask[None],  # True = real token, False = pad
    }


def _print_packing_stats(num_batches, num_examples, total_real_tokens, total_padded_tokens):
    efficiency = total_real_tokens / (total_real_tokens + total_padded_tokens) * 100
    print(f"Packing stats:")
    print(f"  Total batches: {num_batches}")
    print(f"  Real tokens: {total_real_tokens:,}")
    print(f"  Padding tokens: {total_padded_tokens:,}")
    print(f"  Efficiency: {efficiency:.1f}%")
    print(f"  Avg sequences per batch: {num_examples / max(1, num_batches):.1f}")


def pack_examples(examples, pack_length, pad_token_id=0, pin_memory=True):
    total_real_tokens = 0
    total_padded_tokens = 0

    collated = []
    for batch_examples, total_len in tqdm(iter_packed_groups(examples, pack_length), desc="Collating"):
        batch = collate_packed_group(batch_examples, total_len, pack_length, pad_token_id)
        if pin_memory:
            batch = {k: v.pin_memory() for k, v in batch.items()}
        collated.append(batch)
        total_real_tokens += total_len
        total_padded_tokens += pack_length - total_len

    _print_packing_stats(len(collated), len(examples), total_real_tokens, total_padded_tokens)
    return collated


def save_packed_batches(examples, pack_length, pad_token_id, out_path, save_every=SAVE_EVERY):
    """Stream packed batches to disk to avoid holding the full list in RAM."""
    total_real_tokens = 0
    total_padded_tokens = 0
    num_batches = 0
    top_k = None
    buffer = []

    with open(out_path, "wb") as f:
        for batch_examples, total_len in tqdm(iter_packed_groups(examples, pack_length), desc="Collating+Saving"):
            batch = collate_packed_group(batch_examples, total_len, pack_length, pad_token_id)
            buffer.append(batch)
            num_batches += 1
            total_real_tokens += total_len
            total_padded_tokens += pack_length - total_len
            if top_k is None:
                top_k = batch['teacher_indices'].shape[-1]
            if len(buffer) >= save_every:
                torch.save(buffer, f)
                buffer = []

        if buffer:
            torch.save(buffer, f)

    meta = {
        "num_sequences": num_batches,
        "num_batches": num_batches,
        "num_examples": len(examples),
        "pack_length": pack_length,
        "top_k": top_k,
        "total_real_tokens": total_real_tokens,
        "total_padded_tokens": total_padded_tokens,
    }
    torch.save(meta, out_path + ".meta.pt")
    _print_packing_stats(num_batches, len(examples), total_real_tokens, total_padded_tokens)
    return meta


def iter_packed_batches_from_file(path, pin_memory=False):
    """Load batches from a .pt file. Supports both list format and legacy streaming format."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        for batch in data:
            if pin_memory:
                batch = {k: v.pin_memory() for k, v in batch.items()}
            yield batch
    else:
        # Legacy format: single batch
        if pin_memory:
            data = {k: v.pin_memory() for k, v in data.items()}
        yield data


def stack_sequences(batch_group, pin_memory=False):
    stacked = {
        key: torch.cat([seq[key] for seq in batch_group], dim=0)
        for key in batch_group[0].keys()
    }
    if pin_memory:
        stacked = {k: v.pin_memory() for k, v in stacked.items()}
    return stacked


def iter_batched_sequences_from_files(chunk_files, batch_size, pin_memory=False, shuffle_files=False, shuffle_seed=None):
    files = list(chunk_files)
    if shuffle_files:
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            rng.shuffle(files)
        else:
            random.shuffle(files)

    buffer = []
    for path in files:
        for seq in iter_packed_batches_from_file(path, pin_memory=False):
            buffer.append(seq)
            if len(buffer) == batch_size:
                yield stack_sequences(buffer, pin_memory=pin_memory)
                buffer = []

    if buffer:
        yield stack_sequences(buffer, pin_memory=pin_memory)


def get_num_batches_for_file(path):
    meta_path = path + ".meta.pt"
    if os.path.exists(meta_path):
        try:
            meta = torch.load(meta_path)
            if isinstance(meta, dict) and meta.get("num_batches") is not None:
                return meta["num_batches"]
        except Exception:
            pass

    count = 0
    for _ in iter_packed_batches_from_file(path, pin_memory=False):
        count += 1
    torch.save({"num_batches": count}, meta_path)
    return count


def pack_dataset_cached(examples, pack_length, pad_token_id, cache_path, pin_memory=True):
    """Pack dataset with caching to avoid recomputation."""
    if os.path.exists(cache_path):
        print(f"Loading cached batches from {cache_path}")
        batches = torch.load(cache_path)
        if pin_memory:
            batches = [{k: v.pin_memory() for k, v in b.items()} for b in batches]
        return batches

    print(f"Packing batches (will cache to {cache_path})...")
    batches = pack_examples(examples, pack_length, pad_token_id, pin_memory=False)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(batches, cache_path)

    if pin_memory:
        batches = [{k: v.pin_memory() for k, v in b.items()} for b in batches]
    return batches


def batch_packed_sequences(packed_sequences, batch_size, pin_memory=True):
    batched = []
    for batch_start in range(0, len(packed_sequences), batch_size):
        batch_group = packed_sequences[batch_start : batch_start + batch_size]

        stacked = {
            key: torch.cat([seq[key] for seq in batch_group], dim=0)
            for key in batch_group[0].keys()
        }

        if pin_memory:
            stacked = {k: v.pin_memory() for k, v in stacked.items()}
        batched.append(stacked)

    actual_batch_sizes = [b['input_ids'].shape[0] for b in batched]
    if actual_batch_sizes[-1] != batch_size:
        print(f"Note: Last batch has {actual_batch_sizes[-1]} sequences (others have {batch_size})")

    return batched


def move_batch_to_device(batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    if batch['input_ids'].dtype != torch.long:
        batch['input_ids'] = batch['input_ids'].long()
    if batch['position_ids'].dtype != torch.long:
        batch['position_ids'] = batch['position_ids'].long()
    if batch['teacher_indices'].dtype != torch.long:
        batch['teacher_indices'] = batch['teacher_indices'].long()
    return batch


def eval(student, val_batches, kld_loss, device):
    """Evaluate on validation set using packed batches."""
    student.eval()
    val_loss = 0
    for batch in val_batches:
        batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            student_logprobs = F.log_softmax(
                student(batch['input_ids'], position_ids=batch['position_ids']).logits,
                dim=-1
            )
            pad_mask = batch['pad_mask']
            student_logprobs_at_topk = student_logprobs.gather(-1, batch['teacher_indices'])
            student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
            teacher_logits = batch['teacher_logits'][pad_mask]
            # Normalize raw logits to log probabilities over top-k
            teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(teacher_logits.dtype)
            val_loss += kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float()).item()
    student.train()
    return val_loss / len(val_batches)

tokenizer = AutoTokenizer.from_pretrained(STUDENT)
pad_token_id = tokenizer.pad_token_id or 0
pack_length = 6144  # Max length in dataset is 6094

# Load pre-packed chunks from cache/chunks/
import glob
train_chunk_files = sorted(glob.glob(f"{CHUNKS_DIR}/train_*.pt"))
val_path = f"{CHUNKS_DIR}/val.pt"

if not train_chunk_files or not os.path.exists(val_path):
    print(f"ERROR: Pre-packed data not found in {CHUNKS_DIR}/")
    print("Download it first:")
    print('  from huggingface_hub import snapshot_download')
    print('  snapshot_download("hbfreed/dolci-distill-packed", repo_type="dataset", local_dir="cache/chunks")')
    import sys; sys.exit(1)

print(f"Loading pre-packed data from {CHUNKS_DIR}/...")
val_packed = torch.load(val_path, weights_only=False)
print(f"Found {len(train_chunk_files)} train chunks, {len(val_packed)} val batches")

# Build chunk entries with batch counts
train_chunk_entries = []
for path in train_chunk_files:
    batches = torch.load(path, weights_only=False)
    train_chunk_entries.append({"path": path, "num_batches": len(batches)})
    del batches

if preprocess_only:
    print(f"\nData already preprocessed! Files in {CHUNKS_DIR}/")
    import sys; sys.exit(0)

# Val batches stay in memory (small), train chunks loaded on-demand
val_batches = batch_packed_sequences(val_packed, batch_size, pin_memory=False)
print(f"Created {len(val_batches)} validation batches, will stream {len(train_chunk_files)} train chunks")

rank_chunk_entries = train_chunk_entries[ddp_rank::ddp_world_size]
rank_batch_counts = [
    sum(entry["num_batches"] for entry in train_chunk_entries[r::ddp_world_size])
    for r in range(ddp_world_size)
]
steps_per_epoch = min(rank_batch_counts) if rank_batch_counts else 0
if master_process:
    print(f"Per-rank batch counts: {rank_batch_counts}; using {steps_per_epoch} steps/epoch")
if steps_per_epoch == 0:
    raise RuntimeError("No training batches available. Check chunk files and DDP world size.")

total_steps = steps_per_epoch * n_epochs
eval_every = max(1, int(total_steps * 0.1))

tokens_per_step = gradient_accumulation_steps * ddp_world_size * batch_size * pack_length
print(f"Distilling {STUDENT} with {TEACHER} as teacher with {tokens_per_step} tokens per step ...")
student = AutoLigerKernelForCausalLM.from_pretrained(
    STUDENT,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

if use_fsdp and ddp:
    # FSDP with per-layer sharding for memory efficiency
    from functools import partial
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Olmo3DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    student = FSDP(
        student,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )
elif ddp:
    student = student.to(device)
    student = DDP(student, device_ids=[ddp_local_rank], static_graph=True)
else:
    student = student.to(device)

if compile:
    student = torch.compile(student, mode="max-autotune", fullgraph=True)

# Note: 8-bit Adam doesn't play well with FSDP, use regular AdamW
optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, fused=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=4.5e-7)
kld_loss = LigerKLDIVLoss(log_target=True)

# Resume from checkpoint if exists
resume_step = 0
resume_path = None  # Set to checkpoint path to resume, e.g. "checkpoints/model_step_1000"
if resume_path and os.path.exists(f"{resume_path}/training_state.pt"):
    print(f"Resuming from {resume_path}")
    checkpoint = torch.load(f"{resume_path}/training_state.pt", weights_only=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    random.setstate(checkpoint['python_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    resume_step = checkpoint['step'] + 1
    print(f"Resumed at step {resume_step}")

if master_process:
    wandb.init(
        project=wandb_project,
        config={
            "teacher": TEACHER,
            "student": STUDENT,
            "pack_length": pack_length,
            "batch_size": batch_size,
            "num_batches": steps_per_epoch,
            "n_epochs": n_epochs,
            "accumulation_steps": gradient_accumulation_steps,
            "lr_start": 2e-4,
            "lr_end": 4.5e-7,
            "dataset_size": len(train_data),
        }
    )

global_step = 0
for epoch in range(n_epochs):
    epoch_chunks = list(rank_chunk_entries)
    rng = random.Random(1223 + epoch + ddp_rank)
    rng.shuffle(epoch_chunks)
    train_iter = iter_batched_sequences_from_files(
        [entry["path"] for entry in epoch_chunks],
        batch_size,
        pin_memory=False,
        shuffle_files=False,
    )
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{n_epochs}", disable=not master_process)
    for _ in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            if master_process:
                print("Warning: ran out of batches before steps_per_epoch")
            break
        # Skip to resume point if resuming
        if global_step < resume_step:
            global_step += 1
            continue

        step_start_time = time.time()
        batch = move_batch_to_device(batch, device)

        # Only sync gradients on the last micro-step (saves communication overhead)
        # Note: FSDP uses no_sync() context manager instead, but we skip that optimization for simplicity
        if ddp and not use_fsdp:
            student.require_backward_grad_sync = ((global_step + 1) % gradient_accumulation_steps == 0)

        student_logprobs = F.log_softmax(
            student(batch['input_ids'], position_ids=batch['position_ids']).logits,
            dim=-1
        )

        pad_mask = batch['pad_mask']
        student_logprobs_at_topk = student_logprobs.gather(-1, batch['teacher_indices'])
        student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
        teacher_logits = batch['teacher_logits'][pad_mask]
        # Normalize raw logits to log probabilities over top-k
        teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(teacher_logits.dtype)

        loss = kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float()) / gradient_accumulation_steps
        loss.backward()

        if (global_step + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float('inf'))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - step_start_time
            num_tokens = pad_mask.sum().item()
            throughput = num_tokens / step_time

            if master_process:
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": throughput,
                    "train/step": global_step,
                    "train/epoch": epoch,
                })
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (global_step + 1) % eval_every == 0 and master_process:
            val_loss = eval(student, val_batches, kld_loss, device)
            wandb.log({"eval/loss": val_loss, "train/step": global_step})
            pbar.set_postfix(val_loss=f"{val_loss:.4f}")
            path = f"{out_dir}_step_{global_step}"
            student.save_pretrained(path)
            torch.save({
                'step': global_step,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_loss,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
            }, path + "/training_state.pt")

        global_step += 1

if master_process:
    wandb.finish()

if ddp:
    destroy_process_group()
