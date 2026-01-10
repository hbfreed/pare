'''Distilling training loop, using KL Divergence (with packing).'''
import torch
import torch.nn.functional as F
from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerKLDIVLoss
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
import time
import random
import os
import numpy as np

# Low-hanging fruit for torch performance
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.benchmark = True

TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "pruned_models/pruned_olmo3_4096_16_29"
wandb_project = "olmo-distillation"
out_dir = f"checkpoints/{STUDENT.split('/')[1]}"
compile = True

batch_size = 4
gradient_accumulation_steps = 3  # will def need to turn this up!
# NOTE: If memory-bound, consider gradient_checkpointing=True in from_pretrained()
# trades ~33% more compute for much lower memory, enabling larger batch sizes  

train_test = load_dataset("hbfreed/Dolci-Instruct-RL-Completions", split='train').train_test_split(test_size=0.02, seed=1223)
train_data = train_test['train']
val_data = train_test['test']

# DDP Setup
backend = 'nccl'
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
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

def tensorize(dataset):
    """Tensorize dataset with explicit dtypes for bf16 training."""
    tensorized = []
    max_len = 0
    for example in tqdm(dataset, desc="Tensorizing"):
        ids = example['input_ids_prompt'] + example['input_ids_completion']
        length = len(ids)
        max_len = max(max_len, length)
        tensorized.append({
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'position_ids': torch.arange(length, dtype=torch.long),
            'teacher_indices': torch.tensor(example['teacher_indices'], dtype=torch.long),
            'teacher_logprobs': torch.tensor(example['teacher_logprobs'], dtype=torch.bfloat16),
            'length': length,
        })
    return tensorized, max_len

def round_to_multiple(x, multiple=128):
    return ((x + multiple - 1) // multiple) * multiple

def pack_examples(examples, pack_length, pad_token_id=0, pin_memory=True):
    sorted_examples = sorted(examples, key=lambda x: x['length'], reverse=True)

    packed_batches = []
    used = [False] * len(sorted_examples)

    total_real_tokens = 0
    total_padded_tokens = 0

    for primary_idx, primary_example in enumerate(sorted_examples):
        if used[primary_idx]:
            continue

        current_batch = [primary_example]
        current_len = primary_example['length']
        used[primary_idx] = True

        # Greedily find more examples that fit (search from smallest up)
        for candidate_idx in range(len(sorted_examples) - 1, primary_idx, -1):
            if used[candidate_idx]:
                continue
            if current_len + sorted_examples[candidate_idx]['length'] <= pack_length:
                current_batch.append(sorted_examples[candidate_idx])
                current_len += sorted_examples[candidate_idx]['length']
                used[candidate_idx] = True

        packed_batches.append((current_batch, current_len))
        total_real_tokens += current_len
        total_padded_tokens += pack_length - current_len

    efficiency = total_real_tokens / (total_real_tokens + total_padded_tokens) * 100
    print(f"Packing stats:")
    print(f"  Total batches: {len(packed_batches)}")
    print(f"  Real tokens: {total_real_tokens:,}")
    print(f"  Padding tokens: {total_padded_tokens:,}")
    print(f"  Efficiency: {efficiency:.1f}%")
    print(f"  Avg sequences per batch: {len(examples) / len(packed_batches):.1f}")

    collated = []
    for batch_examples, total_len in tqdm(packed_batches, desc="Collating"):
        pad_len = pack_length - total_len

        input_ids = torch.cat([ex['input_ids'] for ex in batch_examples])
        position_ids = torch.cat([ex['position_ids'] for ex in batch_examples])
        teacher_indices = torch.cat([ex['teacher_indices'] for ex in batch_examples])
        teacher_logprobs = torch.cat([ex['teacher_logprobs'] for ex in batch_examples])

        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            position_ids = F.pad(position_ids, (0, pad_len), value=0)
            teacher_indices = F.pad(teacher_indices, (0, 0, 0, pad_len), value=0)
            teacher_logprobs = F.pad(teacher_logprobs, (0, 0, 0, pad_len), value=0.0)

        batch = {
            'input_ids': input_ids[None],
            'position_ids': position_ids[None],
            'teacher_indices': teacher_indices[None],
            'teacher_logprobs': teacher_logprobs[None],
            'pad_mask': torch.cat([
                torch.ones(total_len, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])[None],  # True = real token, False = pad
        }
        if pin_memory:
            batch = {k: v.pin_memory() for k, v in batch.items()}
        collated.append(batch)

    return collated


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


def eval(student, val_batches, kld_loss, device):
    """Evaluate on validation set using packed batches."""
    student.eval()
    val_loss = 0
    for batch in val_batches:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            student_logprobs = F.log_softmax(
                student(batch['input_ids'], position_ids=batch['position_ids']).logits,
                dim=-1
            )
            pad_mask = batch['pad_mask']
            student_logprobs_at_topk = student_logprobs.gather(-1, batch['teacher_indices'])
            student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
            teacher_logprobs = batch['teacher_logprobs'][pad_mask]
            val_loss += kld_loss(student_logprobs_at_topk, teacher_logprobs).item()
    student.train()
    return val_loss / len(val_batches)

print("Tensorizing datasets...")
train_examples, train_max_len = tensorize(train_data)
val_examples, val_max_len = tensorize(val_data)

lengths = [ex['length'] for ex in train_examples]
print(f"Length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

pack_length = round_to_multiple(train_max_len, 128)
print(f"Pack length: {pack_length}")

tokenizer = AutoTokenizer.from_pretrained(STUDENT)
pad_token_id = tokenizer.pad_token_id or 0

train_packed = pack_dataset_cached(
    train_examples,
    pack_length,
    pad_token_id,
    cache_path=f"cache/packed_dolci_train_{pack_length}.pt",
    pin_memory=False
)
val_packed = pack_dataset_cached(
    val_examples,
    pack_length,
    pad_token_id,
    cache_path=f"cache/packed_dolci_val_{pack_length}.pt",
    pin_memory=False
)

print(f"Batching {len(train_packed)} packed sequences into batches of {batch_size}...")
train_batches = batch_packed_sequences(train_packed, batch_size, pin_memory=True)
val_batches = batch_packed_sequences(val_packed, batch_size, pin_memory=False)
print(f"Created {len(train_batches)} training batches, {len(val_batches)} validation batches")

num_steps = len(train_batches)
eval_every = max(1, int(num_steps * 0.1))

tokens_per_step = gradient_accumulation_steps * ddp_world_size * batch_size * train_max_len
print(f"Distilling {STUDENT} with {TEACHER} as teacher with {tokens_per_step} tokens per step ...")
student = AutoLigerKernelForCausalLM.from_pretrained(
    STUDENT,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

student = student.to(device)

if ddp:
    student = DDP(student, device_ids=[ddp_local_rank], static_graph=True)

if compile:
    student = torch.compile(student, mode="max-autotune", fullgraph=True)

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, fused=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=4.5e-7)
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
            "num_batches": num_steps,
            "accumulation_steps": gradient_accumulation_steps,
            "lr_start": 2e-4,
            "lr_end": 4.5e-7,
            "dataset_size": len(train_examples),
        }
    )

random.shuffle(train_batches)
# Shard batches across ranks
train_batches = train_batches[ddp_rank::ddp_world_size]
pbar = tqdm(train_batches, disable=not master_process)
for step, batch in enumerate(pbar):
    # Skip to resume point if resuming
    if step < resume_step:
        continue

    step_start_time = time.time()
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    # Only sync gradients on the last micro-step (saves communication overhead)
    if ddp:
        student.require_backward_grad_sync = ((step + 1) % gradient_accumulation_steps == 0)

    student_logprobs = F.log_softmax(
        student(batch['input_ids'], position_ids=batch['position_ids']).logits,
        dim=-1
    )

    pad_mask = batch['pad_mask']
    student_logprobs_at_topk = student_logprobs.gather(-1, batch['teacher_indices'])
    student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
    teacher_logprobs = batch['teacher_logprobs'][pad_mask]

    loss = kld_loss(student_logprobs_at_topk, teacher_logprobs) / gradient_accumulation_steps
    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
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
                "train/step": step,
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    if (step + 1) % eval_every == 0 and master_process:
        val_loss = eval(student, val_batches, kld_loss, device)
        wandb.log({"eval/loss": val_loss, "train/step": step})
        pbar.set_postfix(val_loss=f"{val_loss:.4f}")
        path = f"{out_dir}_step_{step}"
        student.save_pretrained(path)
        torch.save({
            'step': step,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
        }, path + "/training_state.pt")

if master_process:
    wandb.finish()

if ddp:
    destroy_process_group()