"""
Generate teacher logprobs using HuggingFace directly.
Simpler, more controllable than vLLM for this use case.
"""

import argparse
import json
import torch
from pathlib import Path

# Enable TF32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Disable CUDA graphs for dynamic shapes (variable seq lengths cause issues)
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TOP_K = 128


def get_done_indices(output_file: Path) -> set:
    """Get indices already processed."""
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("teacher_indices") is not None:
                        done.add(row["idx"])
                except json.JSONDecodeError:
                    continue
    return done


@torch.inference_mode()
def process_batch(model, batch_input_ids, batch_indices, device):
    """Process a batch of samples with padding and extract top-k logprobs."""
    lengths = [len(ids) for ids in batch_input_ids]
    max_len = max(lengths)

    # Pad sequences and create attention mask
    padded = []
    attention_mask = []
    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        padded.append(ids + [0] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(attention_mask, device=device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Get top-k logits directly (skip full log_softmax to save memory)
    # topk on logits gives same indices as topk on logprobs (monotonic)
    # We'll normalize at training time over just the top-k
    top_logits, top_indices = logits.topk(TOP_K, dim=-1)  # [batch, seq_len, TOP_K]
    del logits  # free immediately

    # Extract results, removing padding
    results = []
    for i, (idx, seq_len) in enumerate(zip(batch_indices, lengths)):
        results.append({
            "idx": idx,
            "teacher_indices": top_indices[i, :seq_len].cpu().tolist(),
            "teacher_logits": top_logits[i, :seq_len].cpu().tolist(),  # raw logits, normalize at training
            "error": None,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-3-7B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="dolci_tokenized")
    parser.add_argument("--output-file", type=str, default="logprobs_hf.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--start-idx", type=int, default=0, help="Start from this dataset index")
    parser.add_argument("--end-idx", type=int, default=None, help="End at this dataset index")
    parser.add_argument("--indices-file", type=str, default=None, help="JSON file with list of indices to process")
    args = parser.parse_args()

    output_file = Path(args.output_file)

    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)

    # Get already done indices
    done = get_done_indices(output_file)
    print(f"Already done: {len(done)}")

    # Determine indices to process
    if args.indices_file:
        with open(args.indices_file) as f:
            all_indices = json.load(f)
        indices = [i for i in all_indices if i not in done]
        print(f"Loaded {len(all_indices)} indices from {args.indices_file}")
    else:
        end_idx = args.end_idx if args.end_idx is not None else len(ds)
        indices = [i for i in range(args.start_idx, end_idx) if i not in done]
    print(f"Processing {len(indices)} samples")

    if not indices:
        print("Nothing to do!")
        return

    # Process samples in batches
    import time
    total_tokens = 0
    total_samples = 0
    start_time = time.time()

    num_batches = (len(indices) + args.batch_size - 1) // args.batch_size

    with open(output_file, "a") as f:
        for batch_idx in tqdm(range(num_batches), desc="Processing"):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(indices))
            batch_indices_list = indices[batch_start:batch_end]

            # Gather input_ids for batch
            batch_input_ids = []
            for idx in batch_indices_list:
                example = ds[idx]
                input_ids = example["input_ids_prompt"] + example["input_ids_completion"]
                batch_input_ids.append(input_ids)
                total_tokens += len(input_ids)

            try:
                results = process_batch(model, batch_input_ids, batch_indices_list, args.device)
            except Exception as e:
                import traceback
                print(f"Error on batch {batch_idx}: {e}")
                traceback.print_exc()
                results = [
                    {"idx": idx, "teacher_indices": None, "teacher_logprobs": None, "error": str(e)}
                    for idx in batch_indices_list
                ]

            for result in results:
                f.write(json.dumps(result) + "\n")
            f.flush()
            torch.cuda.empty_cache()

            total_samples += len(batch_indices_list)
            if (batch_idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                tps = total_tokens / elapsed
                sps = total_samples / elapsed
                print(f"  [{total_samples} samples] {tps:.1f} tok/s, {sps:.2f} samples/s")


if __name__ == "__main__":
    main()
