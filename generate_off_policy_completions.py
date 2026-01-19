import json
import argparse
import asyncio
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset

async def generate_one(client, row, idx, semaphore, model):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": row['prompt']}],
                max_tokens=4096
            )
            completion = response.choices[0].message.content
        except Exception as e:
            print(f"Error on idx {idx}: {e}")
            completion = None

        return {
            'idx': idx,
            'prompt': row['prompt'],
            'completion': completion
        }

def get_done_indices(output_dir: Path, num_ranks: int) -> set:
    """Collect all successfully completed indices across all rank files."""
    done = set()
    for rank in range(num_ranks):
        filepath = output_dir / f'completions_{rank}.jsonl'
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get('completion') is not None:
                            done.add(row['idx'])
                    except json.JSONDecodeError:
                        continue
    return done

def get_remaining_indices(total: int, done: set, indices_file: str = None) -> list:
    """Get indices that still need processing."""
    if indices_file and Path(indices_file).exists():
        with open(indices_file) as f:
            indices = json.load(f)
        return sorted([i for i in indices if i not in done])
    return sorted([i for i in range(total) if i not in done])

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--num-ranks', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--port-base', type=int, default=8000)
    parser.add_argument('--model', type=str, default='allenai/Olmo-3-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='allenai/Dolci-Instruct-RL')
    parser.add_argument('--prompt-field', type=str, default='prompt')
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--indices-file', type=str, default=None, help='JSON file with specific indices to retry')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.rank <= 2:
        # Ranks 0-2: local vLLM
        port = args.port_base + args.rank
        client = AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="none",
            timeout=args.timeout
        )
    else:
        # Rank 3: OpenRouter
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key= "REDACTED",
            timeout=args.timeout
        )

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, split='train')
    total = len(ds)

    # Figure out what's done across ALL ranks
    done = get_done_indices(output_dir, args.num_ranks)
    remaining = get_remaining_indices(total, done, args.indices_file)
    print(f"Total: {total}, Done: {len(done)}, Remaining: {len(remaining)}")

    # Split remaining work evenly across ranks
    chunks = [remaining[i::args.num_ranks] for i in range(args.num_ranks)]
    my_indices = chunks[args.rank]
    print(f"Rank {args.rank}: processing {len(my_indices)} samples")

    if not my_indices:
        print("Nothing to do!")
        return

    output_file = output_dir / f'completions_{args.rank}.jsonl'
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process in batches
    for i in tqdm(range(0, len(my_indices), args.concurrency), desc=f"Rank {args.rank}"):
        batch_indices = my_indices[i:i + args.concurrency]
        batch_rows = [(idx, ds[idx]) for idx in batch_indices]

        batch_results = await asyncio.gather(*[
            generate_one(client, row, idx, semaphore, args.model)
            for idx, row in batch_rows
        ])

        with open(output_file, 'a') as f:
            for out in batch_results:
                f.write(json.dumps(out) + '\n')
            f.flush()

if __name__ == '__main__':
    asyncio.run(main())
