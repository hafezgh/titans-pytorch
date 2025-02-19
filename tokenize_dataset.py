from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token  # needed if eos_token != pad_token

# Load dataset
raw_dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=False,
    cache_dir="/scratch/users/hafezgh/fineweb-10B"
)

def tokenize_and_chunk(examples):
    """
    Splits each text into 4096-token chunks, then returns them as 
    lists of equal length for 'input_ids' and 'attention_mask'.
    """
    chunk_size = 4096
    all_input_ids = []
    all_attention_masks = []

    for text in examples["text"]:
        tokens = tokenizer(
            text,
            return_tensors="np",
            truncation=False,
            padding=False
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]

        # Break into chunks
        for i in range(0, len(input_ids), chunk_size):
            chunk_ids = input_ids[i : i + chunk_size]
            chunk_mask = attention_mask[i : i + chunk_size]

            # Pad if needed
            if len(chunk_ids) < chunk_size:
                pad_len = chunk_size - len(chunk_ids)
                chunk_ids = np.pad(
                    chunk_ids,
                    (0, pad_len),
                    constant_values=tokenizer.eos_token_id
                )
                chunk_mask = np.pad(
                    chunk_mask,
                    (0, pad_len),
                    constant_values=0
                )

            # Collect the chunk
            all_input_ids.append(chunk_ids.tolist())
            all_attention_masks.append(chunk_mask.tolist())

    # Return a dict of lists (one column per list)
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks
    }

# Remove ALL columns from the original dataset.
# This ensures we don't get a length mismatch on the old columns.
all_cols = raw_dataset.column_names

processed_dataset = raw_dataset.map(
    tokenize_and_chunk,
    batched=True,
    batch_size=1024,
    remove_columns=all_cols,
    desc="Tokenizing Dataset"
)

# Save final dataset
save_path = "/scratch/users/hafezgh/fineweb-10B-tokenized"
processed_dataset.save_to_disk(save_path)                 # HF dataset format
processed_dataset.to_parquet(f"{save_path}.parquet")      # Parquet format
print(f"✅ Processed dataset saved to {save_path} and {save_path}.parquet")