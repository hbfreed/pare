"""
Generate teacher logprobs for distillation dataset.

Mirrors generate_off_policy_completions.py pattern for multi-rank processing.
Reads tokenized dataset (from build_distill_dataset.py) and adds top-128 logprobs.
"""
import json
import argparse
import asyncio
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_from_disk

TOP_K = 128  # number of top logprobs to store


async def get_teacher_logprobs(client, token_ids, idx, semaphore, model):
    """Get top-k logprobs for each position in the sequence."""
    async with semaphore:
        try:
            response = await client.completions.create(
                model=model,
                prompt=token_ids,  # list of token IDs
                max_tokens=0,  # no generation, just prompt logprobs
                extra_body={"prompt_logprobs": TOP_K}
            )

            # Parse prompt_logprobs from response
            # vLLM returns list of dicts, one per position
            # Each dict maps token_id (str) -> logprob (float)
            prompt_logprobs = response.prompt_logprobs

            teacher_indices = []
            teacher_logprobs = []

            for position_logprobs in prompt_logprobs:
                if position_logprobs is None:
                    # First position has no logprobs (nothing to predict from)
                    # Use zeros as placeholder
                    teacher_indices.append([0] * TOP_K)
                    teacher_logprobs.append([0.0] * TOP_K)
                else:
                    # Sort by logprob descending, take top-k
                    sorted_items = sorted(
                        position_logprobs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:TOP_K]

                    indices = [int(tok_id) for tok_id, _ in sorted_items]
                    logprobs = [lp for _, lp in sorted_items]

                    # Pad if fewer than TOP_K (shouldn't happen, but be safe)
                    while len(indices) < TOP_K:
                        indices.append(0)
                        logprobs.append(float('-inf'))

                    teacher_indices.append(indices)
                    teacher_logprobs.append(logprobs)

            return {
                'idx': idx,
                'teacher_indices': teacher_indices,
                'teacher_logprobs': teacher_logprobs,
                'error': None
            }

        except Exception as e:
            print(f"Error on idx {idx}: {e}")
            return {
                'idx': idx,
                'teacher_indices': None,
                'teacher_logprobs': None,
                'error': str(e)
            }


def get_done_indices(output_dir: Path, num_ranks: int) -> set:
    """Collect all successfully completed indices across all rank files."""
    done = set()
    for rank in range(num_ranks):
        filepath = output_dir / f'logprobs_{rank}.jsonl'
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get('teacher_indices') is not None:
                            done.add(row['idx'])
                    except json.JSONDecodeError:
                        continue
    return done


def get_remaining_indices(total: int, done: set) -> list:
    """Get indices that still need processing."""
    return sorted([i for i in range(total) if i not in done])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--num-ranks', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--port-base', type=int, default=8000)
    parser.add_argument('--model', type=str, default='allenai/Olmo-3-7B-Instruct')
    parser.add_argument('--dataset-path', type=str, default='dolci_tokenized')
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    port = args.port_base + args.rank
    client = AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="none",
        timeout=args.timeout
    )

    # Load tokenized dataset
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)
    total = len(ds)

    # Figure out what's done across ALL ranks
    done = get_done_indices(output_dir, args.num_ranks)
    remaining = get_remaining_indices(total, done)
    print(f"Total: {total}, Done: {len(done)}, Remaining: {len(remaining)}")

    # Split remaining work evenly across ranks
    chunks = [remaining[i::args.num_ranks] for i in range(args.num_ranks)]
    my_indices = chunks[args.rank]
    print(f"Rank {args.rank}: processing {len(my_indices)} samples")

    if not my_indices:
        print("Nothing to do!")
        return

    output_file = output_dir / f'logprobs_{args.rank}.jsonl'
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process in batches
    for batch_start in tqdm(range(0, len(my_indices), args.concurrency), desc=f"Rank {args.rank}"):
        batch_indices = my_indices[batch_start:batch_start + args.concurrency]

        # Prepare token sequences (prompt + completion)
        batch_data = []
        for idx in batch_indices:
            example = ds[idx]
            token_ids = example['input_ids_prompt'] + example['input_ids_completion']
            batch_data.append((idx, token_ids))

        batch_results = await asyncio.gather(*[
            get_teacher_logprobs(client, token_ids, idx, semaphore, args.model)
            for idx, token_ids in batch_data
        ])

        with open(output_file, 'a') as f:
            for out in batch_results:
                f.write(json.dumps(out) + '\n')
            f.flush()


if __name__ == '__main__':
    asyncio.run(main())
