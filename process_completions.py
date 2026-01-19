import json
from collections import OrderedDict

# Read all completions from the 3 files
all_rows = []
for i in range(3):
    filename = f"completions_{i}.jsonl"
    with open(filename, 'r') as f:
        for line in f:
            row = json.loads(line.strip())
            all_rows.append(row)

print(f"Total rows read: {len(all_rows)}")

# Dedupe: for each idx, keep the first non-null completion
# Process in order so "first" is preserved
idx_to_completion = {}
for row in all_rows:
    idx = row['idx']
    completion = row.get('completion')

    if idx not in idx_to_completion:
        # First time seeing this idx
        idx_to_completion[idx] = row
    elif idx_to_completion[idx].get('completion') is None and completion is not None:
        # We had a null, now we have a non-null - replace
        idx_to_completion[idx] = row

print(f"Unique indices after dedup: {len(idx_to_completion)}")

# Count indices with null completions after deduping
null_indices = [idx for idx, row in idx_to_completion.items() if row.get('completion') is None]
print(f"Indices with null completions after dedup: {len(null_indices)}")

# Find missing indices (0-169963)
expected_indices = set(range(169964))
present_indices = set(idx_to_completion.keys())
missing_indices = sorted(expected_indices - present_indices)
print(f"Missing indices (never attempted): {len(missing_indices)}")

# Combine nulls + missing into retry list
indices_to_retry = sorted(set(null_indices) | set(missing_indices))
print(f"Total indices needing retry (null + missing): {len(indices_to_retry)}")

# Save retry list
with open('indices_to_retry.json', 'w') as f:
    json.dump(indices_to_retry, f)
print(f"Saved {len(indices_to_retry)} indices to indices_to_retry.json")

# Filter to non-null completions only and sort by idx
clean_completions = [row for row in idx_to_completion.values() if row.get('completion') is not None]
clean_completions.sort(key=lambda x: x['idx'])
print(f"Clean completions (non-null): {len(clean_completions)}")

# Save clean completions
with open('completions_clean.jsonl', 'w') as f:
    for row in clean_completions:
        f.write(json.dumps(row) + '\n')
print(f"Saved {len(clean_completions)} completions to completions_clean.jsonl")

# Final stats
print("\n" + "="*50)
print("FINAL STATS")
print("="*50)
print(f"Clean completions saved: {len(clean_completions)}")
print(f"Indices needing retry: {len(indices_to_retry)}")
print(f"  - From null completions: {len(null_indices)}")
print(f"  - From missing indices: {len(missing_indices)}")
print(f"Total accounted for: {len(clean_completions) + len(indices_to_retry)} (should be 169964)")
