"""On-policy distillation with DDP support.

GPU layout:
  GPU 0: vLLM student (generation)
  GPU 1: HF teacher (inference)
  GPU 2+: DDP student ranks (training)

Launch with: torchrun --nproc_per_node=N distill_on_policy.py [args]
"""

import argparse
import datetime
import io
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Queue

import bitsandbytes as bnb
import cloudpickle
from huggingface_hub import HfApi, hf_hub_download
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from vllm import LLM, SamplingParams

import wandb
# from evals import run_evals  # disabled: lm-eval incompatible with datasets on cluster

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

# DATASET = "allenai/Dolci-Think-RL-7B"
DATASET = "allenai/Dolci-Instruct-RL"
TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "hbfreed/pruned_olmo3_4096_16_29"
HUB_REPO = "hbfreed/pruned_olmo3_4096_16_29_distilled_on_policy"
WANDB_PROJECT = "pare-on-policy-distillation"
RUN_NAME = None  # set to a string to override auto naming

TEACHER_DEVICE = "cuda:1"
DDP_GPU_OFFSET = 2  # rank i -> cuda:{i + DDP_GPU_OFFSET}

BATCH_SIZE = 1
N_EPOCHS = 1
GROUP_SIZE = 4  # number of rollouts per prompt
GRAD_ACCUM_STEPS = 256
MAX_CONTEXT_LENGTH = 512
LR = 1e-5
CLIP_EPS = 0.2
MAX_GRAD_NORM = 3.0
WARMUP_STEPS = 50
SYNC_EVERY_N_STEPS = 4
SYNC_MIN = 1
SYNC_MAX = 4

RESUME_FROM = None
DEBUG_MODE = False
N_SAMPLE_PROMPTS = 4
SAMPLE_EVERY_N_STEPS = 50
EVAL_EVERY_N_STEPS = 50
EVAL_N_SAMPLES = 200
EVAL_TASKS = ["gsm8k_cot", "arc_easy", "truthfulqa_mc2", "ifeval"]

SAVE_EVERY = 500  # permanent milestone checkpoint every N steps
SAVE_EVERY_N_STEPS = 50  # rolling checkpoint + HF push every N steps
CHECKPOINT_BASE = "checkpoints/onpolicy-from-baseline"

torch.manual_seed(1223)


def get_logprobs_at_tokens(logits, tokens, vocab_size=None):
    if vocab_size is not None:
        logits = logits[:, :, :vocab_size]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return -F.cross_entropy(
        shift_logits.view(B * T, V), shift_labels.view(B * T),
        reduction="none",
    ).view(B, T)


def run_teacher_pipeline(teacher, sequences, attention_mask, chunk_size,
                         device, student_device, queue, consumer_chunk_size=None):
    """Producer: compute teacher logprobs in chunks, emit exact consumer-sized pieces."""
    if consumer_chunk_size is None:
        consumer_chunk_size = chunk_size
    try:
        buffer = []
        buffered_rows = 0
        for i in range(0, len(sequences), chunk_size):
            chunk_seq = sequences[i:i + chunk_size].to(device, non_blocking=True)
            chunk_mask = attention_mask[i:i + chunk_size].to(device, non_blocking=True)
            try:
                with torch.inference_mode():
                    t_out = teacher(input_ids=chunk_seq, attention_mask=chunk_mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Teacher OOM with batch_size={chunk_seq.shape[0]}, "
                    f"seq_len={chunk_seq.shape[1]}. Reduce --teacher-micro-batch-size "
                    f"(currently {chunk_size})."
                )
            logprobs = get_logprobs_at_tokens(t_out.logits, chunk_seq)
            logprobs = logprobs.to(student_device).detach()
            buffer.append(logprobs)
            buffered_rows += logprobs.shape[0]
            while buffered_rows >= consumer_chunk_size:
                combined = torch.cat(buffer, dim=0)
                queue.put(combined[:consumer_chunk_size])
                remainder = combined[consumer_chunk_size:]
                buffer = [remainder] if remainder.shape[0] > 0 else []
                buffered_rows = remainder.shape[0]
        queue.put(None)  # sentinel
    except Exception as e:
        queue.put(e)
        raise


def generate_rollouts(
    vllm_student, prompts, pad_token_id, group_size=1, max_context_length=4096, vocab_size=None
):
    """Generate rollouts from student model using vLLM, returning sequences and prompt length."""
    prompt_lens = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lens)
    max_new_tokens = max_context_length - max_prompt_len

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=max_new_tokens,
        n=group_size, logprobs=1,
    )

    token_prompts = [{"prompt_token_ids": p} for p in prompts]
    outputs = vllm_student.generate(
        prompts=token_prompts, sampling_params=sampling_params, use_tqdm=False,
    )

    all_sequences = []
    all_logprobs = []
    for req_output, prompt_len in zip(outputs, prompt_lens):
        prompt_ids = req_output.prompt_token_ids
        for completion in req_output.outputs:
            full_seq = list(prompt_ids) + list(completion.token_ids)
            all_sequences.append(full_seq)
            seq_logprobs = [0.0] * prompt_len
            for idx, logprob_dict in enumerate(completion.logprobs):
                token_id = completion.token_ids[idx]
                seq_logprobs.append(logprob_dict[token_id].logprob)
            all_logprobs.append(seq_logprobs)

    max_seq_len = max(len(seq) for seq in all_sequences)
    padded = [seq + [pad_token_id] * (max_seq_len - len(seq)) for seq in all_sequences]
    padded_logprobs = [
        logprob + [0.0] * (max_seq_len - len(logprob)) for logprob in all_logprobs
    ]

    sequences = torch.tensor(padded)
    if vocab_size is not None:
        sequences[sequences >= vocab_size] = pad_token_id
    attention_mask = (sequences != pad_token_id).long()
    old_logprobs = torch.tensor(padded_logprobs)
    expanded_prompt_lens = [pl for pl in prompt_lens for _ in range(group_size)]
    return sequences, expanded_prompt_lens, old_logprobs, attention_mask


def prepare_prompts(opt_step_idx, all_batches, tokenizer, grad_accum_steps):
    """Get pretokenized prompts for a given optimizer step index."""
    chunk_start = opt_step_idx * grad_accum_steps
    chunk_end = chunk_start + grad_accum_steps
    # .iter(batch_size=N) wraps values in a list; unwrap to get flat list[int] per prompt
    prompts = []
    for i in range(chunk_start, chunk_end):
        batch = all_batches[i]["input_ids_prompt"]
        if isinstance(batch[0], list):
            prompts.extend(batch)
        else:
            prompts.append(batch)
    return prompts


def timed_generate_rollouts(*args, **kwargs):
    t0 = time.time()
    result = generate_rollouts(*args, **kwargs)
    return (*result, time.time() - t0)


def build_loss_mask(sequences, prompt_lens, pad_token_id):
    """Build mask: 1.0 for completion tokens, 0.0 for prompt/padding. Returns [B, T-1]."""
    batch_size, seq_len = sequences.shape
    positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
    prompt_lens_t = torch.tensor(prompt_lens, device=sequences.device).unsqueeze(1)
    mask = (positions >= prompt_lens_t).float()
    mask[sequences == pad_token_id] = 0.0
    return mask[:, 1:]


def generate_samples(vllm_student, eval_prompts, tokenizer, max_context_length=4096):
    """Generate completions for eval prompts and return a wandb.Table."""
    prompt_lens = [len(p) for p in eval_prompts]
    max_prompt_len = max(prompt_lens)
    sampling_params = SamplingParams(
        temperature=0.7, max_tokens=max_context_length - max_prompt_len, n=1,
    )
    token_prompts = [{"prompt_token_ids": [int(x) for x in p]} for p in eval_prompts]
    outputs = vllm_student.generate(
        prompts=token_prompts, sampling_params=sampling_params, use_tqdm=False,
    )
    table = wandb.Table(columns=["prompt", "completion"])
    for req_output in outputs:
        prompt_text = tokenizer.decode(req_output.prompt_token_ids, skip_special_tokens=True)
        completion_text = tokenizer.decode(req_output.outputs[0].token_ids, skip_special_tokens=True)
        table.add_data(prompt_text, completion_text)
    return table


def save_checkpoint(student, tokenizer, optimizer, global_step, hub_repo=None):
    """Save rolling 'latest'/'prev' checkpoints, plus a permanent one every SAVE_EVERY steps."""
    latest_dir = f"{CHECKPOINT_BASE}/latest"
    prev_dir = f"{CHECKPOINT_BASE}/prev"

    if os.path.exists(latest_dir):
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        os.rename(latest_dir, prev_dir)

    os.makedirs(latest_dir, exist_ok=True)
    student.save_pretrained(latest_dir)
    tokenizer.save_pretrained(latest_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": global_step},
        f"{latest_dir}/training_state.pt",
    )
    print(f"Saved latest checkpoint (step {global_step}) to {latest_dir}")

    if global_step % SAVE_EVERY == 0:
        milestone_dir = f"{CHECKPOINT_BASE}/step_{global_step}"
        os.makedirs(milestone_dir, exist_ok=True)
        student.save_pretrained(milestone_dir)
        tokenizer.save_pretrained(milestone_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(), "step": global_step},
            f"{milestone_dir}/training_state.pt",
        )
        print(f"Saved milestone checkpoint to {milestone_dir}")

    if hub_repo:
        try:
            HfApi().upload_folder(
                folder_path=latest_dir, repo_id=hub_repo,
                commit_message=f"Step {global_step}",
            )
            print(f"Pushed checkpoint to {hub_repo}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")


def sync_weights_to_vllm(hf_model, vllm_llm):
    """Sync weights from HF model to vLLM engine for on-policy learning.

    Sends weights in chunks to avoid msgspec's 4GB encoding limit.
    """
    hf_state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}

    # Split state dict into chunks that serialize under 4GB each
    chunks = []
    current_chunk = {}
    current_size = 0
    max_chunk_bytes = 2 * (1024 ** 3)  # 2GB per chunk for safety margin

    for k, v in hf_state_dict.items():
        tensor_bytes = v.nelement() * v.element_size()
        if current_size + tensor_bytes > max_chunk_bytes and current_chunk:
            buf = io.BytesIO()
            torch.save(current_chunk, buf)
            chunks.append(buf.getvalue())
            current_chunk = {}
            current_size = 0
        current_chunk[k] = v
        current_size += tensor_bytes

    if current_chunk:
        buf = io.BytesIO()
        torch.save(current_chunk, buf)
        chunks.append(buf.getvalue())

    def load_weights_on_worker(worker, serialized_weights):
        buf = io.BytesIO(serialized_weights)
        weights_dict = torch.load(buf, weights_only=True)
        weights = list(weights_dict.items())
        worker.model_runner.model.load_weights(weights=weights)

    method_bytes = cloudpickle.dumps(load_weights_on_worker)
    for chunk in chunks:
        vllm_llm.llm_engine.collective_rpc(method_bytes, args=(chunk,))


def timed_sync_weights_to_vllm(hf_model, vllm_llm):
    start = time.time()
    sync_weights_to_vllm(hf_model, vllm_llm)
    return time.time() - start


def load_checkpoint(checkpoint_path, student, optimizer, vllm_student=None):
    """Load optimizer state and return the step to resume from."""
    state_path = f"{checkpoint_path}/training_state.pt"
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
    else:
        try:
            downloaded = hf_hub_download(checkpoint_path, "training_state.pt")
            state = torch.load(downloaded, weights_only=False)
        except Exception:
            return 0
    optimizer.load_state_dict(state["optimizer"])
    if vllm_student is not None:
        sync_weights_to_vllm(student, vllm_student)
    return state["step"]

# Adaptive sync state (per-process, only rank 0 uses it)
steps_since_decrease = 0


def get_sync_interval(step, mean_ratio, approx_drift, current_interval):
    global steps_since_decrease
    if step < 5:
        return 1
    if abs(mean_ratio - 1.0) > 0.2 or approx_drift > 0.25:
        steps_since_decrease = 0
        return max(SYNC_MIN, current_interval // 2)
    steps_since_decrease += 1
    if (abs(mean_ratio - 1.0) < 0.05 and approx_drift < 0.08
            and steps_since_decrease > 20):
        return min(SYNC_MAX, current_interval + 1)
    return current_interval


def broadcast_rollout_data(rank, world_size, device, sequences, attention_mask,
                           old_student_logprobs, teacher_logprobs_all,
                           hit_eos, prompt_lens_t):
    """Broadcast all rollout tensors from rank 0 to all ranks via NCCL.

    Tensors come in on CPU from rank 0; moved to CUDA for broadcast, returned on CPU.
    """
    # 1) Broadcast shape info so non-zero ranks can allocate buffers
    if rank == 0:
        shape_info = torch.tensor([sequences.shape[0], sequences.shape[1]],
                                  dtype=torch.long, device=device)
    else:
        shape_info = torch.empty(2, dtype=torch.long, device=device)
    dist.broadcast(shape_info, src=0)
    n_seq, seq_len = shape_info.tolist()

    # 2) Build tensors on CUDA for NCCL broadcast
    def _to_device(t, shape, dtype):
        if rank == 0:
            return t.to(device)
        return torch.empty(shape, dtype=dtype, device=device)

    sequences = _to_device(sequences, (n_seq, seq_len), torch.long)
    attention_mask = _to_device(attention_mask, (n_seq, seq_len), torch.long)
    old_student_logprobs = _to_device(old_student_logprobs, (n_seq, seq_len), torch.float32)
    teacher_logprobs_all = _to_device(teacher_logprobs_all, (n_seq, seq_len - 1), torch.float32)
    # NCCL doesn't support bool; cast to uint8 for broadcast
    if rank == 0:
        hit_eos = hit_eos.to(torch.uint8).to(device)
    else:
        hit_eos = torch.empty(n_seq, dtype=torch.uint8, device=device)
    prompt_lens_t = _to_device(prompt_lens_t, (n_seq,), torch.long)

    # 3) Broadcast all tensors
    dist.broadcast(sequences, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(old_student_logprobs, src=0)
    dist.broadcast(teacher_logprobs_all, src=0)
    dist.broadcast(hit_eos, src=0)
    dist.broadcast(prompt_lens_t, src=0)

    # Return on CPU to avoid double-storing on GPU (we'll .to(device) selectively later)
    return (sequences.cpu(), attention_mask.cpu(), old_student_logprobs.cpu(),
            teacher_logprobs_all.cpu(), hit_eos.cpu().bool(), prompt_lens_t.cpu())


def collect_teacher_logprobs(teacher, sequences, attention_mask,
                             teacher_micro_batch_size, student_device):
    """Run teacher pipeline synchronously, return full [N, seq_len-1] tensor."""
    queue = Queue(maxsize=4)
    # Set CUDA device to teacher's GPU so Triton kernels launch on the right device
    prev_device = torch.cuda.current_device()
    torch.cuda.set_device(TEACHER_DEVICE)
    run_teacher_pipeline(teacher, sequences, attention_mask,
                         teacher_micro_batch_size, TEACHER_DEVICE,
                         student_device, queue,
                         consumer_chunk_size=len(sequences))
    result = queue.get()
    torch.cuda.set_device(prev_device)
    if isinstance(result, BaseException):
        raise result
    sentinel = queue.get()
    assert sentinel is None
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="DDP on-policy distillation")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--micro-batch-size", type=int, default=2,
                        help="Sequences per student forward pass")
    parser.add_argument("--teacher-micro-batch-size", type=int, default=6,
                        help="Sequences per teacher forward pass")
    parser.add_argument("--sweep", type=int, default=None,
                        help="Stop after N optimizer steps (for quick sweeps)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume")
    return parser.parse_args()


def main_ddp():
    args = parse_args()
    lr = args.lr
    micro_batch_size = args.micro_batch_size
    teacher_micro_batch_size = args.teacher_micro_batch_size
    sweep_steps = args.sweep
    wandb_run_id = args.wandb_run_id

    # Init vLLM on rank 0 BEFORE DDP, since DDP's CUDA init breaks vLLM's subprocess spawn.
    # Also clear torchrun env vars so vLLM's EngineCore subprocess doesn't try to join DDP.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    vllm_student = None
    if local_rank == 0:
        print("Loading vLLM student on cuda:0...")
        # Clear ALL torchrun/torchelastic env vars so vLLM's EngineCore subprocess
        # doesn't try to join DDP or use the agent store for rendezvous
        torchrun_prefixes = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                             "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "GROUP_WORLD_SIZE",
                             "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_", "TORCH_NCCL_",
                             "NCCL_")
        saved_env = {}
        for k in list(os.environ):
            if any(k == p or k.startswith(p) for p in torchrun_prefixes):
                saved_env[k] = os.environ.pop(k)
        vllm_student = LLM(
            STUDENT, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
            gpu_memory_utilization=0.98,
        )
        os.environ.update(saved_env)

    # Set CUDA device before DDP init so NCCL doesn't default to cuda:0
    # (which vLLM's EngineCore subprocess needs)
    if local_rank != 0:
        torch.cuda.set_device(f"cuda:{local_rank + DDP_GPU_OFFSET}")

    # DDP init (long timeout for rank 0 model loading)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank + DDP_GPU_OFFSET}"
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"DDP: {world_size} ranks, devices cuda:{DDP_GPU_OFFSET}..cuda:{DDP_GPU_OFFSET + world_size - 1}")

    # Tokenizer + dataset (all ranks)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds = load_dataset(DATASET, split="train")
    dataset = (
        ds.select_columns(["prompt", "input_ids_prompt"])
        .filter(lambda x: len(x["input_ids_prompt"]) < MAX_CONTEXT_LENGTH)
        .shuffle(seed=1223)
    )

    if DEBUG_MODE:
        from datasets import concatenate_datasets
        single = dataset.select(range(1))
        dataset = concatenate_datasets([single] * GRAD_ACCUM_STEPS)
        if rank == 0:
            print(f"DEBUG MODE: 1 prompt repeated {GRAD_ACCUM_STEPS}x")

    n_epochs = 20 if DEBUG_MODE else N_EPOCHS
    group_size = GROUP_SIZE
    max_context = MAX_CONTEXT_LENGTH

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * n_epochs

    # Student model (all ranks, each on its own GPU)
    if rank == 0:
        print(f"Loading student model from {RESUME_FROM or STUDENT}...")
    student_src = RESUME_FROM if RESUME_FROM else STUDENT
    student = AutoLigerKernelForCausalLM.from_pretrained(
        student_src, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    student.gradient_checkpointing_enable()
    SHARED_VOCAB_SIZE = student.config.vocab_size

    student = DDP(student, device_ids=[rank + DDP_GPU_OFFSET])

    optimizer = bnb.optim.AdamW8bit(
        student.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8
    )
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    # Rank 0 only: teacher, wandb, eval prompts
    teacher = None
    vllm_executor = None
    checkpoint_executor = None
    teacher_executor = None
    eval_prompts = None

    if rank == 0:
        print(f"Loading teacher model from {TEACHER}...")
        teacher = AutoLigerKernelForCausalLM.from_pretrained(
            TEACHER, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(TEACHER_DEVICE)
        teacher.eval()
        print(f"Student vocab: {SHARED_VOCAB_SIZE}, Teacher vocab: {teacher.config.vocab_size}")

        eval_prompts = [
            dataset[i]["input_ids_prompt"]
            for i in range(min(N_SAMPLE_PROMPTS, len(dataset)))
        ]

        vllm_executor = ThreadPoolExecutor(max_workers=1)
        checkpoint_executor = ThreadPoolExecutor(max_workers=1)
        teacher_executor = ThreadPoolExecutor(max_workers=1)

    # Resume from checkpoint
    start_step = 0
    if RESUME_FROM and rank == 0:
        start_step = load_checkpoint(RESUME_FROM, student.module, optimizer, vllm_student)
        for _ in range(start_step):
            scheduler.step()
        print(f"Resuming from step {start_step}")

    # Broadcast start_step from rank 0
    start_step_t = torch.tensor([start_step], dtype=torch.long, device=device)
    dist.broadcast(start_step_t, src=0)
    start_step = start_step_t.item()
    if start_step > 0 and rank != 0:
        for _ in range(start_step):
            scheduler.step()

    # Wandb init (rank 0 only)
    if rank == 0:
        def short_name(model_name: str) -> str:
            return model_name.split("/")[-1]

        run_name = RUN_NAME
        if run_name is None:
            student_short = short_name(STUDENT).lower()
            lr_str = f"{lr:.0e}".replace("-0", "-")
            sweep_tag = f"-sweep{sweep_steps}" if sweep_steps else ""
            run_name = (
                f"{student_short}-onpolicy-ddp{world_size}"
                f"-lr{lr_str}-clip{CLIP_EPS}-sync{SYNC_EVERY_N_STEPS}{sweep_tag}"
            )

        wandb.init(
            project=WANDB_PROJECT,
            id=wandb_run_id,
            name=run_name,
            config={
                "teacher": TEACHER, "student": STUDENT,
                "batch_size": BATCH_SIZE, "group_size": group_size,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "micro_batch_size": micro_batch_size,
                "teacher_micro_batch_size": teacher_micro_batch_size,
                "ddp_world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
                "n_epochs": n_epochs, "total_steps": total_steps,
                "lr": lr, "sweep_steps": sweep_steps,
                "clip_eps": CLIP_EPS, "max_grad_norm": MAX_GRAD_NORM,
                "warmup_steps": WARMUP_STEPS, "max_context_length": max_context,
            },
            resume="must" if wandb_run_id else "allow",
        )

        baseline_table = generate_samples(vllm_student, eval_prompts, tokenizer, max_context)
        wandb.log({"eval/samples": baseline_table}, step=0)

    # Training loop
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=steps_per_epoch * n_epochs - start_step, desc="Training",
                disable=(rank != 0))

    # Rank 0 async state
    checkpoint_future = None
    sync_future = None
    sample_future = None
    eval_future = None
    gen_future = None
    last_sync_duration = None
    sync_interval = SYNC_EVERY_N_STEPS
    n_sequences_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * group_size
    n_micro_batches = n_sequences_per_step // micro_batch_size
    assert n_micro_batches % world_size == 0, (
        f"n_micro_batches ({n_micro_batches}) must be divisible by world_size ({world_size}). "
        f"Adjust --micro-batch-size or GRAD_ACCUM_STEPS."
    )
    mbs_per_rank = n_micro_batches // world_size

    for epoch in range(n_epochs):
        all_batches = list(dataset.iter(batch_size=BATCH_SIZE))

        for opt_step_idx in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + opt_step_idx
            if current_step < start_step:
                continue

            # === Rank 0: generate rollouts + teacher logprobs ===
            sequences = None
            attention_mask = None
            old_student_logprobs = None
            teacher_logprobs_all = None
            hit_eos = None
            prompt_lens_t = None

            if rank == 0:
                # Drain in-flight vLLM work
                if sync_future is not None:
                    last_sync_duration = sync_future.result()
                    sync_future = None
                if sample_future is not None:
                    wandb.log({"eval/samples": sample_future.result()})
                    sample_future = None
                if eval_future is not None:
                    wandb.log(eval_future.result())
                    eval_future = None

                opt_step_start_time = time.time()

                # Use prefetched generation or generate synchronously
                if gen_future is not None:
                    sequences, prompt_lens, old_student_logprobs, attention_mask, gen_time = (
                        gen_future.result()
                    )
                    gen_future = None
                else:
                    prompts = prepare_prompts(opt_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS)
                    gen_start = time.time()
                    sequences, prompt_lens, old_student_logprobs, attention_mask = (
                        generate_rollouts(
                            vllm_student, prompts, PAD_TOKEN_ID,
                            group_size, max_context, SHARED_VOCAB_SIZE,
                        )
                    )
                    gen_time = time.time() - gen_start

                # Prefetch next step
                next_step_idx = opt_step_idx + 1
                will_sync = (global_step + 1) % sync_interval == 0
                can_prefetch = (
                    next_step_idx < steps_per_epoch
                    and sync_interval > 1
                    and not will_sync
                )
                if can_prefetch:
                    next_prompts = prepare_prompts(
                        next_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS,
                    )
                    gen_future = vllm_executor.submit(
                        timed_generate_rollouts,
                        vllm_student, next_prompts, PAD_TOKEN_ID,
                        group_size, max_context, SHARED_VOCAB_SIZE,
                    )

                if opt_step_idx < 2:
                    print(f"[step {opt_step_idx}] Rollout: {tokenizer.decode(sequences[0].tolist()[:200])}")

                # hit_eos detection
                positions = torch.arange(sequences.shape[1]).unsqueeze(0)
                prompt_lens_tensor = torch.tensor(prompt_lens).unsqueeze(1)
                completion_mask = (positions >= prompt_lens_tensor) & (sequences != PAD_TOKEN_ID)
                hit_eos = ((sequences == tokenizer.eos_token_id) & completion_mask).any(dim=1)
                prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long)

                old_student_logprobs = old_student_logprobs.float()

            if world_size > 1:
                # Multi-rank: collect all teacher logprobs, broadcast, then train
                if rank == 0:
                    teacher_logprobs_all = collect_teacher_logprobs(
                        teacher, sequences, attention_mask,
                        teacher_micro_batch_size, device,
                    )
                    teacher_logprobs_all = teacher_logprobs_all.cpu().float()

                (sequences, attention_mask, old_student_logprobs,
                 teacher_logprobs_all, hit_eos, prompt_lens_t) = broadcast_rollout_data(
                    rank, world_size, device, sequences, attention_mask,
                    old_student_logprobs, teacher_logprobs_all,
                    hit_eos, prompt_lens_t,
                )
                prompt_lens = prompt_lens_t.tolist()

            # Pre-compute masks on each rank's device
            old_logprobs_shifted_all = old_student_logprobs[:, 1:].to(device)
            loss_mask_all = build_loss_mask(sequences.to(device), prompt_lens, PAD_TOKEN_ID)
            total_generated_tokens = loss_mask_all.sum().item()

            # === Micro-batch loop (DDP) ===
            rank_start = rank * mbs_per_rank

            if world_size == 1:
                # Stream teacher logprobs: teacher runs in background thread,
                # student consumes chunks as they arrive (overlapping GPU work)
                teacher_queue = Queue(maxsize=4)
                torch.cuda.set_device(TEACHER_DEVICE)
                teacher_future = teacher_executor.submit(
                    run_teacher_pipeline, teacher, sequences, attention_mask,
                    teacher_micro_batch_size, TEACHER_DEVICE, device, teacher_queue,
                    micro_batch_size,
                )
                torch.cuda.set_device(device)
            else:
                teacher_logprobs_all = teacher_logprobs_all.to(device)

            for local_idx in range(mbs_per_rank):
                mb_idx = rank_start + local_idx
                seq_start = mb_idx * micro_batch_size
                seq_end = seq_start + micro_batch_size

                if world_size == 1:
                    teacher_lp = teacher_queue.get()
                    if isinstance(teacher_lp, BaseException):
                        raise teacher_lp
                else:
                    teacher_lp = teacher_logprobs_all[seq_start:seq_end]

                mb_old_lp = old_logprobs_shifted_all[seq_start:seq_end]
                mb_loss_mask = loss_mask_all[seq_start:seq_end]
                mb_advantage = -(mb_old_lp - teacher_lp).detach()

                student_input = sequences[seq_start:seq_end].to(device, non_blocking=True)
                student_mask = attention_mask[seq_start:seq_end].to(device, non_blocking=True)

                student_out = student(
                    input_ids=student_input,
                    attention_mask=student_mask,
                )
                current_logprobs = get_logprobs_at_tokens(
                    student_out.logits, student_input, SHARED_VOCAB_SIZE,
                )

                if global_step < 3 and local_idx == 0:
                    print(f"[rank {rank}] --- Gradient diagnostics (step {global_step}) ---")
                    print(f"  logits.grad_fn: {student_out.logits.grad_fn}")
                    print(f"  current_logprobs.grad_fn: {current_logprobs.grad_fn}")
                    print(f"  loss_mask sum: {mb_loss_mask.sum().item()}")

                ratio = torch.exp(current_logprobs - mb_old_lp)
                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                pg_loss1 = -ratio * mb_advantage
                pg_loss2 = -clipped_ratio * mb_advantage
                per_token_loss = torch.max(pg_loss1, pg_loss2)
                masked_loss = (per_token_loss * mb_loss_mask).sum() / mb_loss_mask.sum()

                if global_step < 3 and local_idx == 0:
                    ms = max(mb_loss_mask.sum().item(), 1)
                    print(f"  ratio mean: {(ratio * mb_loss_mask).sum().item() / ms:.6f}")
                    print(f"  masked_loss: {masked_loss.item():.6f}")

                # Each rank divides by mbs_per_rank; DDP averages across world_size
                # -> total division = mbs_per_rank * world_size = n_micro_batches
                scaled_loss = masked_loss / mbs_per_rank
                ctx = nullcontext() if (local_idx == mbs_per_rank - 1) else student.no_sync()
                with ctx:
                    scaled_loss.backward()
                accumulated_loss += scaled_loss.item()

            if world_size == 1:
                # Drain sentinel and propagate any teacher exception
                assert teacher_queue.get() is None
                teacher_future.result()

            # === Optimizer step ===
            if global_step < 3:
                has_grads = any(
                    p.grad is not None for p in student.parameters() if p.requires_grad
                )
                print(f"[rank {rank}] Step {global_step + 1}: gradients exist = {has_grads}")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # === Logging + sync (rank 0 only) ===
            if rank == 0:
                opt_step_time = time.time() - opt_step_start_time
                avg_loss = accumulated_loss
                tokens_per_sec = total_generated_tokens / gen_time if gen_time > 0 else 0

                mask_sum_mb = mb_loss_mask.sum()
                seq_lens_all = attention_mask.to(device).sum(dim=1)
                prompt_lens_all_t = prompt_lens_t.to(device)
                avg_gen_len = (seq_lens_all - prompt_lens_all_t).float().mean()
                mb_kl = ((mb_old_lp - teacher_lp) * mb_loss_mask).sum() / mask_sum_mb

                metrics_tensor = torch.stack([
                    grad_norm,
                    (mb_advantage * mb_loss_mask).sum() / mask_sum_mb,
                    (ratio * mb_loss_mask).sum() / mask_sum_mb,
                    mb_kl,
                    ((ratio > 1.0 + CLIP_EPS) | (ratio < 1.0 - CLIP_EPS)).float().sum() / mask_sum_mb,
                    ((pg_loss2 > pg_loss1) * mb_loss_mask).sum() / mask_sum_mb,
                    ((current_logprobs - mb_old_lp).abs() * mb_loss_mask).sum() / mask_sum_mb,
                ])
                (grad_norm_val, mean_advantage, mean_ratio, mean_kl,
                 ratio_clipped_frac, clip_active_frac, approx_policy_drift) = metrics_tensor.tolist()

                log_payload = {
                    "train/loss": avg_loss,
                    "train/grad_norm": grad_norm_val,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/gen_time_sec": gen_time,
                    "train/optimizer_step_time_sec": opt_step_time,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/mean_advantage": mean_advantage,
                    "train/mean_ratio": mean_ratio,
                    "train/mean_kl": mean_kl,
                    "train/ratio_clipped_frac": ratio_clipped_frac,
                    "train/clip_active_frac": clip_active_frac,
                    "train/approx_policy_drift": approx_policy_drift,
                    "train/avg_gen_length": avg_gen_len.item(),
                    "train/no_eos_frac": (~hit_eos).float().mean().item(),
                }
                if last_sync_duration is not None:
                    log_payload["train/sync_duration_sec"] = last_sync_duration
                    last_sync_duration = None
                sync_interval = get_sync_interval(
                    global_step, mean_ratio, approx_policy_drift, sync_interval,
                )
                log_payload["train/sync_every_n_steps"] = sync_interval
                wandb.log(log_payload)

            accumulated_loss = 0.0
            global_step += 1
            pbar.update(1)

            # Rank 0: sync weights, samples, evals, checkpoints
            if rank == 0:
                if global_step % sync_interval == 0:
                    if gen_future is not None:
                        gen_future = None  # discard stale prefetch
                    if sync_future is None or sync_future.done():
                        sync_future = vllm_executor.submit(
                            timed_sync_weights_to_vllm, student.module, vllm_student,
                        )

                if global_step % SAMPLE_EVERY_N_STEPS == 0 and sample_future is None:
                    sample_future = vllm_executor.submit(
                        generate_samples, vllm_student, eval_prompts, tokenizer, max_context,
                    )

                # if global_step % EVAL_EVERY_N_STEPS == 0 and eval_future is None:
                #     eval_future = vllm_executor.submit(
                #         run_evals, vllm_student, tokenizer, STUDENT,
                #         tasks=EVAL_TASKS, limit=EVAL_N_SAMPLES,
                #     )

                hub_repo = None if DEBUG_MODE else HUB_REPO
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    if checkpoint_future is not None:
                        checkpoint_future.result()
                    checkpoint_future = checkpoint_executor.submit(
                        save_checkpoint, student.module, tokenizer, optimizer,
                        global_step, hub_repo,
                    )

            if sweep_steps and global_step >= sweep_steps:
                if rank == 0:
                    print(f"Sweep: stopping after {sweep_steps} steps")
                break

        if sweep_steps and global_step >= sweep_steps:
            break

    pbar.close()

    # Cleanup
    if rank == 0:
        if sync_future is not None:
            sync_future.result()
        if sample_future is not None:
            wandb.log({"eval/samples": sample_future.result()})
        if eval_future is not None:
            wandb.log(eval_future.result())
        if checkpoint_future is not None:
            checkpoint_future.result()

        vllm_executor.shutdown(wait=True)
        checkpoint_executor.shutdown(wait=True)

        hub_repo = None if DEBUG_MODE else HUB_REPO
        save_checkpoint(student.module, tokenizer, optimizer, global_step, hub_repo)
        wandb.finish()
        if teacher_executor is not None:
            teacher_executor.shutdown(wait=False)
        print("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main_ddp()
