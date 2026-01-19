"""
Process and merge logprobs JSONL files.

Handles the 470GB of logprobs data by:
1. First pass: Build index of idx -> (file, line_num) for valid records only
2. Identify missing/failed indices for potential retry
3. Report stats

This is a precursor to finalize_distill_dataset.py which will do the actual merge.
"""
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

LOGPROBS_FILES = ["logprobs_0.jsonl", "logprobs_1.jsonl", "logprobs_2.jsonl"]
EXPECTED_COUNT = 169956  # 0 to 169955 inclusive


def build_index():
    """
    Build an index of idx -> (file_idx, line_num) for valid records.
    Only keeps the first valid record for each idx (handles duplicates).
    """
    idx_to_location = {}  # idx -> (file_idx, line_num, byte_offset)
    failed_indices = set()

    for file_idx, filepath in enumerate(LOGPROBS_FILES):
        print(f"\nScanning {filepath}...")
        path = Path(filepath)
        if not path.exists():
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        with open(path, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"  {filepath}")):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    idx = row['idx']
                    has_data = row.get('teacher_indices') is not None

                    if has_data:
                        # Only keep first valid record for each idx
                        if idx not in idx_to_location:
                            idx_to_location[idx] = (file_idx, line_num)
                    else:
                        # Track failed indices (OOM, errors)
                        if idx not in idx_to_location:
                            failed_indices.add(idx)

                except json.JSONDecodeError:
                    print(f"  WARNING: JSON decode error at line {line_num}")
                    continue

    return idx_to_location, failed_indices


def main():
    print("Building index of logprobs files...")
    idx_to_location, failed_indices = build_index()

    # Remove failed indices that were later successful
    failed_indices -= set(idx_to_location.keys())

    # Find completely missing indices
    all_expected = set(range(EXPECTED_COUNT))
    present_indices = set(idx_to_location.keys())
    missing_indices = all_expected - present_indices - failed_indices

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Expected indices: 0 to {EXPECTED_COUNT - 1} ({EXPECTED_COUNT} total)")
    print(f"Valid records found: {len(idx_to_location)}")
    print(f"Failed records (OOM/error): {len(failed_indices)}")
    print(f"Missing indices (never attempted): {len(missing_indices)}")
    print(f"Total accounted: {len(idx_to_location) + len(failed_indices) + len(missing_indices)}")

    # Distribution across files
    file_counts = defaultdict(int)
    for idx, (file_idx, _) in idx_to_location.items():
        file_counts[file_idx] += 1
    print("\nRecords per file:")
    for file_idx, filepath in enumerate(LOGPROBS_FILES):
        print(f"  {filepath}: {file_counts[file_idx]}")

    # Save indices that need retry
    retry_indices = sorted(failed_indices | missing_indices)
    if retry_indices:
        with open("logprobs_retry_indices.json", "w") as f:
            json.dump(retry_indices, f)
        print(f"\nSaved {len(retry_indices)} indices to logprobs_retry_indices.json")

    # Save the index for use by finalize script
    # Convert to serializable format
    index_data = {
        "idx_to_location": {str(k): v for k, v in idx_to_location.items()},
        "files": LOGPROBS_FILES,
    }
    with open("logprobs_index.json", "w") as f:
        json.dump(index_data, f)
    print(f"Saved index to logprobs_index.json")

    return idx_to_location, retry_indices


if __name__ == "__main__":
    main()
