from datasets import load_dataset
from transformers import AutoTokenizer

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
raw_dataset = load_dataset("your_dataset_name")

# Tokenize, chunk, and pad function
def tokenize_and_chunk(examples):
    tokens = tokenizer(
        examples["text"],
        return_tensors="np",
        truncation=False,
        padding=False,
    )

    input_ids = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]

    chunks = []
    chunk_size = 4096

    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i : i + chunk_size]
        chunk_mask = attention_mask[i : i + chunk_size]

        if len(chunk_ids) < chunk_size:
            pad_length = chunk_size - len(chunk_ids)
            chunk_ids = np.pad(chunk_ids, (0, pad_length), constant_values=tokenizer.eos_token_id)
            chunk_mask = np.pad(chunk_mask, (0, pad_length), constant_values=0)

        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": chunk_mask,
        })

    return {"input_ids": [chunk["input_ids"] for chunk in chunks],
            "attention_mask": [chunk["attention_mask"] for chunk in chunks]}

# Tokenize and chunk dataset
processed_dataset = raw_dataset.map(
    tokenize_and_chunk,
    batched=True,
    remove_columns=["text"],
    batch_size=1
).map(lambda batch: batch, batched=True)

print(processed_dataset)
