import json
import os
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

# ✅ Get absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_dataset.json")

# ✅ Check if preprocessed dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Error: Preprocessed dataset not found at {DATA_PATH}")

# ✅ Load preprocessed dataset
with open(DATA_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)

# ✅ Ensure dataset is valid
if not isinstance(dataset, list) or len(dataset) == 0:
    raise ValueError("❌ Error: Preprocessed dataset is empty or not in list format.")

# ✅ Load tokenizer (fix `legacy=True` warning)
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# ✅ Define PyTorch dataset class
class CustomT5Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = self._validate_data(data)  # Validate dataset before use
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # ✅ Construct input text in a more structured format
        input_text = f"Modify the text based on the command below:\nCommand: {entry['command']}\nText: {entry['original_text']}"
        output_text = entry["modified_text"]

        # ✅ Tokenize input and output correctly
        input_encoding = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        output_encoding = self.tokenizer(
            output_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": output_encoding["input_ids"].squeeze()
        }

    def _validate_data(self, data):
        """Ensures all dataset entries have required fields"""
        valid_data = []
        for idx, entry in enumerate(data):
            if not all(k in entry for k in ["command", "original_text", "modified_text"]):
                print(f"⚠️ Warning: Skipping invalid entry {idx} (Missing fields)")
                continue
            valid_data.append(entry)
        
        if len(valid_data) == 0:
            raise ValueError("❌ Error: No valid dataset entries found. Fix `preprocessed_dataset.json`!")

        return valid_data

# ✅ Create dataset instance
t5_dataset = CustomT5Dataset(dataset, tokenizer)

# ✅ Print sample data
print("\n✅ Dataset Successfully Created!")
print("🔹 Sample Data:")
print("📌 Command:", t5_dataset.data[0]["command"])
print("📌 Original Text:", t5_dataset.data[0]["original_text"][:100] + "...")  # Show first 100 chars
print("📌 Modified Text:", t5_dataset.data[0]["modified_text"][:100] + "...")  # Show first 100 chars
