"""Generate teacher completions offline using vLLM for off-policy distillation.

Each rank loads its own vLLM LLM on a separate GPU and processes a slice of the dataset.
Run 3 copies for 3 GPUs:
    python generate_off_policy_completions.py --rank 0 &
    python generate_off_policy_completions.py --rank 1 &
    python generate_off_policy_completions.py --rank 2 &

After all ranks finish, tokenize completions into an HF dataset:
    python generate_off_policy_completions.py --build-dataset
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TOKENIZED_OUTPUT = "dolci_tokenized"


def get_done_indices(output_dir: Path, num_ranks: int) -> set:
    """Collect all successfully completed indices across all rank files."""
    done = set()
    for rank in range(num_ranks):
        filepath = output_dir / f"completions_{rank}.jsonl"
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get("completion") is not None:
                            done.add(row["idx"])
                    except json.JSONDecodeError:
                        continue
    return done


def build_tokenized_dataset(output_dir: Path, num_ranks: int, model_name: str):
    """Tokenize all completions into an HF dataset (replaces build_distill_dataset.py).

    Run after all ranks finish generating completions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load all completions across ranks
    all_records = []
    for rank in range(num_ranks):
        filepath = output_dir / f"completions_{rank}.jsonl"
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        try:
                            row = json.loads(line)
                            if row.get("completion") is not None:
                                all_records.append(row)
                        except json.JSONDecodeError:
                            continue

    print(f"Loaded {len(all_records)} completions across {num_ranks} ranks")

    processed = []
    for rec in tqdm(all_records, desc="Tokenizing"):
        messages = [{"role": "user", "content": rec["prompt"]}]
        input_ids_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        completion_text = rec["completion"] + "<|im_end|>"
        input_ids_completion = tokenizer.encode(completion_text, add_special_tokens=False)
        processed.append({
            "idx": rec["idx"],
            "input_ids_prompt": input_ids_prompt,
            "input_ids_completion": input_ids_completion,
        })

    dataset = Dataset.from_list(processed)
    dataset.save_to_disk(TOKENIZED_OUTPUT)
    print(f"Saved tokenized dataset ({len(dataset)} examples) to {TOKENIZED_OUTPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--num-ranks", type=int, default=3)
    parser.add_argument("--build-dataset", action="store_true",
                        help="Tokenize all completions into HF dataset (run after all ranks finish)")
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="allenai/Dolci-Instruct-RL")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of prompts per vLLM generate() call")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.build_dataset:
        build_tokenized_dataset(output_dir, args.num_ranks, args.model)
        return

    if args.rank is None:
        parser.error("--rank is required when not using --build-dataset")

    # Load dataset and tokenizer
    print(f"[Rank {args.rank}] Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    total = len(ds)

    # Resume support: skip already-done indices
    done = get_done_indices(output_dir, args.num_ranks)

    # Split work across ranks
    all_remaining = sorted(i for i in range(total) if i not in done)
    my_indices = [idx for i, idx in enumerate(all_remaining) if i % args.num_ranks == args.rank]
    print(f"[Rank {args.rank}] Total: {total}, Done: {len(done)}, My work: {len(my_indices)}")

    if not my_indices:
        print(f"[Rank {args.rank}] Nothing to do!")
        return

    # Load vLLM on this rank's GPU
    print(f"[Rank {args.rank}] Loading vLLM on cuda:{args.rank}...")
    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        device=f"cuda:{args.rank}",
    )

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096,
    )

    output_file = output_dir / f"completions_{args.rank}.jsonl"

    # Process in batches
    for batch_start in range(0, len(my_indices), args.batch_size):
        batch_indices = my_indices[batch_start:batch_start + args.batch_size]
        batch_rows = [ds[idx] for idx in batch_indices]

        # Apply chat template
        prompts = []
        for row in batch_rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        outputs = llm.generate(prompts, sampling_params)

        with open(output_file, "a") as f:
            for idx, output in zip(batch_indices, outputs):
                completion = output.outputs[0].text
                result = {
                    "idx": idx,
                    "prompt": ds[idx]["prompt"],
                    "completion": completion,
                }
                f.write(json.dumps(result) + "\n")
            f.flush()

        done_so_far = batch_start + len(batch_indices)
        print(f"[Rank {args.rank}] {done_so_far}/{len(my_indices)} completed")


if __name__ == "__main__":
    main()
