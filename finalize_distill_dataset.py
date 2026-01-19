"""
Combine tokenized dataset with teacher logprobs and push to hub.

Streaming approach to handle 470GB+ of logprobs data:
1. Load tokenized dataset into memory (small - just token IDs)
2. Stream through logprobs files, matching with tokenized data
3. Write directly to Arrow format using a generator

Run after:
1. build_distill_dataset.py (creates dolci_tokenized/)
2. generate_logprobs_hf.py (creates logprobs_*.jsonl)

Parallel mode:
    # Run these in separate terminals:
    uv run finalize_distill_dataset.py --file-index 0
    uv run finalize_distill_dataset.py --file-index 1
    uv run finalize_distill_dataset.py --file-index 2
    uv run finalize_distill_dataset.py --file-index 3  # retry file

    # Then merge:
    uv run finalize_distill_dataset.py --merge
"""
import argparse
import json
import glob
from pathlib import Path
from datasets import load_from_disk, Dataset, concatenate_datasets
from tqdm import tqdm

TOKENIZED_PATH = "dolci_tokenized"
LOGPROBS_FILES = ["logprobs_0.jsonl", "logprobs_1.jsonl", "logprobs_2.jsonl", "logprobs_retry.jsonl"]
HUB_DATASET = "hbfreed/Dolci-Instruct-RL-Completions"
OUTPUT_PATH = "/media/henry/MoreFiles/dolci_final"
CACHE_DIR = "/media/henry/MoreFiles/hf_cache"


def stream_merged_examples(tokenized_ds, logprobs_files):
    """
    Generator that streams through logprobs files and yields merged examples.

    Dedupes by idx (keeps first valid record for each idx).
    Uses teacher_logits from the logprobs files (not teacher_logprobs).
    """
    # Build lookup from tokenized dataset (small, fits in memory)
    print("Building tokenized lookup...")
    tokenized_lookup = {}
    for i in tqdm(range(len(tokenized_ds)), desc="Indexing tokenized"):
        example = tokenized_ds[i]
        tokenized_lookup[example['idx']] = {
            'input_ids_prompt': example['input_ids_prompt'],
            'input_ids_completion': example['input_ids_completion'],
        }
    print(f"Indexed {len(tokenized_lookup)} tokenized examples")

    # Track which indices we've already yielded (for deduping)
    seen_indices = set()
    yielded_count = 0
    skipped_count = 0
    missing_tokenized = 0

    for filepath in logprobs_files:
        print(f"\nProcessing {filepath}...")
        try:
            with open(filepath, 'r') as f:
                for line in tqdm(f, desc=f"  {filepath}"):
                    if not line.strip():
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    idx = row['idx']

                    # Skip if no valid data (OOM'd)
                    if row.get('teacher_indices') is None:
                        skipped_count += 1
                        continue

                    # Skip if we already have this idx (dedupe)
                    if idx in seen_indices:
                        continue

                    # Skip if no matching tokenized data
                    if idx not in tokenized_lookup:
                        missing_tokenized += 1
                        continue

                    seen_indices.add(idx)
                    tokenized = tokenized_lookup[idx]

                    yielded_count += 1
                    yield {
                        'idx': idx,
                        'input_ids_prompt': tokenized['input_ids_prompt'],
                        'input_ids_completion': tokenized['input_ids_completion'],
                        'teacher_indices': row['teacher_indices'],
                        'teacher_logits': row['teacher_logits'],  # raw logits, not logprobs
                    }

        except FileNotFoundError:
            print(f"  WARNING: {filepath} not found, skipping")

    print(f"\n--- Stream stats ---")
    print(f"Yielded: {yielded_count}")
    print(f"Skipped (OOM/error): {skipped_count}")
    print(f"Missing tokenized: {missing_tokenized}")


def process_single_file(file_index, tokenized_ds):
    """Process a single logprobs file into a partial dataset."""
    filepath = LOGPROBS_FILES[file_index]
    output_path = f"{OUTPUT_PATH}_part{file_index}"

    print(f"Processing {filepath} -> {output_path}")

    def gen():
        yield from stream_merged_examples(tokenized_ds, [filepath])

    ds = Dataset.from_generator(gen, cache_dir=f"{CACHE_DIR}_{file_index}")
    print(f"Got {len(ds)} examples from {filepath}")

    if len(ds) > 0:
        ds.save_to_disk(output_path)
        print(f"Saved to {output_path}")
    else:
        print(f"No examples from {filepath}, skipping save")

    return len(ds)


def merge_partials():
    """Merge all partial datasets, dedupe by idx, and save final."""
    print("Merging partial datasets...")

    datasets = []
    for i in range(len(LOGPROBS_FILES)):
        part_path = f"{OUTPUT_PATH}_part{i}"
        if Path(part_path).exists():
            print(f"Loading {part_path}...")
            ds = load_from_disk(part_path)
            print(f"  {len(ds)} examples")
            datasets.append(ds)
        else:
            print(f"  {part_path} not found, skipping")

    if not datasets:
        print("ERROR: No partial datasets found!")
        return

    # Concatenate all
    print(f"\nConcatenating {len(datasets)} datasets...")
    combined = concatenate_datasets(datasets)
    print(f"Combined: {len(combined)} examples")

    # Dedupe by idx (keep first occurrence)
    print("Deduplicating by idx...")
    print("  Extracting idx column...")
    all_idxs = combined['idx']  # fast batch read of single column

    seen = set()
    keep_indices = []
    for i, idx in enumerate(tqdm(all_idxs, desc="  Deduping")):
        if idx not in seen:
            seen.add(idx)
            keep_indices.append(i)

    final_ds = combined.select(keep_indices)
    print(f"After dedupe: {len(final_ds)} examples")

    # Sort by idx
    print("Sorting by idx...")
    final_ds = final_ds.sort('idx')

    # Verify sample
    print(f"\nSample keys: {list(final_ds[0].keys())}")
    sample = final_ds[0]
    seq_len = len(sample['input_ids_prompt']) + len(sample['input_ids_completion'])
    teacher_len = len(sample['teacher_indices'])
    top_k = len(sample['teacher_indices'][0])
    print(f"Sample: seq_len={seq_len}, teacher_len={teacher_len}, top_k={top_k}")

    # Save final - use 5GB shards (Hub-friendly, good for downloads)
    print(f"\nSaving to {OUTPUT_PATH}...")
    final_ds.save_to_disk(OUTPUT_PATH, max_shard_size="5GB", num_proc=12)
    print(f"Saved to {OUTPUT_PATH}/")

    # Optionally push to hub
    push = input(f"\nPush to {HUB_DATASET}? [y/N] ").strip().lower()
    if push == 'y':
        print(f"Pushing to {HUB_DATASET}...")
        final_ds.push_to_hub(HUB_DATASET)
        print("Done!")
    else:
        print("Skipped push to hub")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-index", type=int, default=None,
                        help="Process only this file index (0-3)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial datasets into final")
    args = parser.parse_args()

    if args.merge:
        merge_partials()
        return

    # Load tokenized dataset
    print(f"Loading tokenized dataset from {TOKENIZED_PATH}...")
    ds = load_from_disk(TOKENIZED_PATH)
    print(f"Tokenized dataset has {len(ds)} examples")

    if args.file_index is not None:
        # Process single file
        process_single_file(args.file_index, ds)
    else:
        # Sequential mode: process all files
        print("\nCreating final dataset from stream...")

        def gen():
            yield from stream_merged_examples(ds, LOGPROBS_FILES)

        final_ds = Dataset.from_generator(gen, cache_dir=CACHE_DIR)
        print(f"\nFinal dataset has {len(final_ds)} examples")

        if len(final_ds) == 0:
            print("ERROR: No examples in final dataset!")
            return

        # Verify sample
        print(f"Sample keys: {list(final_ds[0].keys())}")
        sample = final_ds[0]
        seq_len = len(sample['input_ids_prompt']) + len(sample['input_ids_completion'])
        teacher_len = len(sample['teacher_indices'])
        top_k = len(sample['teacher_indices'][0])
        print(f"Sample: seq_len={seq_len}, teacher_len={teacher_len}, top_k={top_k}")

        # Save locally
        print(f"\nSaving to {OUTPUT_PATH}...")
        final_ds.save_to_disk(OUTPUT_PATH)
        print(f"Saved to {OUTPUT_PATH}/")

        # Sort by idx for cleaner output
        print("Sorting by idx...")
        final_ds = final_ds.sort('idx')

        # Optionally push to hub
        push = input(f"\nPush to {HUB_DATASET}? [y/N] ").strip().lower()
        if push == 'y':
            print(f"Pushing to {HUB_DATASET}...")
            final_ds.push_to_hub(HUB_DATASET)
            print("Done!")
        else:
            print("Skipped push to hub")


if __name__ == "__main__":
    main()
