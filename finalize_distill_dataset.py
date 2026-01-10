"""
Combine tokenized dataset with teacher logprobs and push to hub.

Run after:
1. build_distill_dataset.py (creates dolci_tokenized/)
2. generate_off_policy_logprobs.py (creates logprobs_*.jsonl)
"""
import json
import glob
from datasets import load_from_disk, Dataset
from tqdm import tqdm

TOKENIZED_PATH = "dolci_tokenized"
LOGPROBS_PATTERN = "logprobs_*.jsonl"
HUB_DATASET = "hbfreed/Dolci-Instruct-RL-Completions"


def load_logprobs(pattern: str = LOGPROBS_PATTERN) -> dict:
    """Load all logprobs JSONL files into a dict keyed by idx."""
    logprobs_by_idx = {}
    for filepath in sorted(glob.glob(pattern)):
        print(f"Loading {filepath}...")
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get('teacher_indices') is not None:
                        logprobs_by_idx[row['idx']] = {
                            'teacher_indices': row['teacher_indices'],
                            'teacher_logprobs': row['teacher_logprobs']
                        }
    print(f"Loaded logprobs for {len(logprobs_by_idx)} examples")
    return logprobs_by_idx


def main():
    # Load tokenized dataset
    print(f"Loading tokenized dataset from {TOKENIZED_PATH}...")
    ds = load_from_disk(TOKENIZED_PATH)
    print(f"Tokenized dataset has {len(ds)} examples")

    # Load logprobs
    logprobs = load_logprobs()

    # Merge
    merged = []
    missing = 0
    for idx in tqdm(range(len(ds)), desc="Merging"):
        example = ds[idx]
        if idx not in logprobs:
            missing += 1
            continue

        merged.append({
            'idx': example['idx'],
            'id': example.get('id'),
            'input_ids_prompt': example['input_ids_prompt'],
            'input_ids_completion': example['input_ids_completion'],
            'teacher_indices': logprobs[idx]['teacher_indices'],
            'teacher_logprobs': logprobs[idx]['teacher_logprobs'],
        })

    if missing > 0:
        print(f"Warning: {missing} examples missing logprobs (skipped)")

    print(f"Final dataset has {len(merged)} examples")

    # Create dataset and push
    final_ds = Dataset.from_list(merged)
    print(f"Sample keys: {final_ds[0].keys()}")

    # Verify shapes
    sample = final_ds[0]
    seq_len = len(sample['input_ids_prompt']) + len(sample['input_ids_completion'])
    teacher_len = len(sample['teacher_indices'])
    top_k = len(sample['teacher_indices'][0])
    print(f"Sample: seq_len={seq_len}, teacher_len={teacher_len}, top_k={top_k}")

    # Save locally first
    final_ds.save_to_disk("dolci_final")
    print("Saved to dolci_final/")

    # Push to hub
    print(f"Pushing to {HUB_DATASET}...")
    final_ds.push_to_hub(HUB_DATASET)
    print("Done!")


if __name__ == "__main__":
    main()
