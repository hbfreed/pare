"""
Build the distillation dataset from completion JSONL files.

IMPORTANT: Remember to add <|im_end|> token to each completion!
"""
import json
import glob
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZER_NAME = "allenai/Olmo-3-7B-Instruct"
OUTPUT_PATH = "dolci_tokenized"
HUB_DATASET = "hbfreed/Dolci-Instruct-RL-Completions"

def load_completions(filepath: str = "completions_final.jsonl") -> list[dict]:
    """Load completions from JSONL file."""
    all_records = []
    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
    print(f"Loaded {len(all_records)} total records")
    return all_records

def build_dataset(records: list[dict]) -> Dataset:
    """
    Build HF dataset with properly formatted completions.

    Tokenizes prompts with chat template and completions with <|im_end|>.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    processed = []
    for rec in tqdm(records, desc="Tokenizing"):
        # Apply chat template to prompt to get input_ids_prompt
        messages = [{"role": "user", "content": rec["prompt"]}]
        # add_generation_prompt=True adds the assistant turn start
        input_ids_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True
        )

        # Tokenize completion WITH <|im_end|> appended
        completion_text = rec["completion"] + "<|im_end|>"
        input_ids_completion = tokenizer.encode(completion_text, add_special_tokens=False)

        processed.append({
            "idx": rec["idx"],
            "input_ids_prompt": input_ids_prompt,
            "input_ids_completion": input_ids_completion,
        })

    return Dataset.from_list(processed)

def main():
    records = load_completions()
    dataset = build_dataset(records)
    print(f"Dataset has {len(dataset)} examples")
    print(f"Sample keys: {dataset[0].keys()}")

    # Save locally (intermediate step before adding logprobs)
    dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Note: Run generate_off_policy_logprobs.py next to add teacher logprobs")

if __name__ == "__main__":
    main()