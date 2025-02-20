from datasets import load_dataset
from transformers import AutoTokenizer
import os
os.environ["TMPDIR"] = "/scratch/users/hafezgh/fineweb-10B-tokenized"
os.environ["HF_DATASETS_CACHE"] = "/scratch/users/hafezgh/fineweb-10B-tokenized"
# ----------------------------------------------------------------
# 1. Load the Llama-2 tokenizer
#    (We use Llama-2's vocab, but train a model from scratch.)
# ----------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Set <eos> as padding token (LLaMA doesn’t have a dedicated <pad> token)
tokenizer.pad_token = tokenizer.eos_token

# Set the max context length
CHUNK_SIZE = 4096

# ----------------------------------------------------------------
# 2. Define a function to tokenize and chunk
# ----------------------------------------------------------------
def tokenize_and_concatenate(examples):
    """
    Tokenizes text, concatenates it into 4K chunks, and pads where necessary.
    """
    all_input_ids = []

    for text in examples["text"]:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,  # Ensure no sequence exceeds 4K
            max_length=CHUNK_SIZE
        )["input_ids"]

        # Append EOS manually
        token_ids.append(tokenizer.eos_token_id)
        all_input_ids.extend(token_ids)

    chunked_input_ids = []
    chunked_attention_masks = []

    for i in range(0, len(all_input_ids), CHUNK_SIZE):
        chunk = all_input_ids[i : i + CHUNK_SIZE]

        if len(chunk) < CHUNK_SIZE:
            pad_len = CHUNK_SIZE - len(chunk)
            chunk.extend([tokenizer.pad_token_id] * pad_len)
            attn_mask = [1] * (CHUNK_SIZE - pad_len) + [0] * pad_len
        else:
            attn_mask = [1] * CHUNK_SIZE

        chunked_input_ids.append(chunk)
        chunked_attention_masks.append(attn_mask)

    return {"input_ids": chunked_input_ids, "attention_mask": chunked_attention_masks}

# ----------------------------------------------------------------
# 3. Load and process dataset
# ----------------------------------------------------------------
raw_dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=False
)

processed_dataset = raw_dataset.map(
    tokenize_and_concatenate,
    batched=True,
    num_proc=8,  # Enable multiprocessing for speedup
    remove_columns=raw_dataset.column_names,
    load_from_cache_file=False,  # Avoid cache issues
    desc="Tokenizing and chunking the dataset"
)

# ----------------------------------------------------------------
# 4. Save dataset to disk
# ----------------------------------------------------------------
save_path = "/path/to/fineweb_tokenized"
processed_dataset.save_to_disk(save_path)
processed_dataset.to_parquet(f"{save_path}.parquet")

print(f"✅ Tokenized dataset saved to {save_path} and {save_path}.parquet")