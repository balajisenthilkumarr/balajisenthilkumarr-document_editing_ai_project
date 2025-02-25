import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from tqdm import tqdm

# ✅ Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "train_dataset.json")  
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "t5_finetuned")

# ✅ Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Error: Train dataset not found at {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)

# ✅ Load Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# ✅ Fix Dataset Format
def format_training_data(entry):
    """Ensure dataset is correctly formatted."""
    command = entry["command"].strip().lower()
    original_text = entry["original_text"].strip()
    modified_text = entry["modified_text"].strip()

    # ✅ Ensure Task Instruction is Clear
    formatted_input = f"Task: {command}\nText: {original_text}"
    
    return formatted_input, modified_text

# ✅ Define Dataset Class
class CustomT5Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = [format_training_data(entry) for entry in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, output_text = self.data[idx]

        input_enc = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        output_enc = self.tokenizer(
            output_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        # ✅ Ignore padding in loss
        labels = output_enc["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100  

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }

# ✅ Create DataLoader
batch_size = 4 if torch.cuda.is_available() else 2
train_dataset = CustomT5Dataset(dataset, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ✅ Load Model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)

# ✅ Optimizer & Learning Rate
optimizer = AdamW(model.parameters(), lr=5e-4)  # 🚀 Increased LR for faster learning
num_training_steps = len(train_dataloader) * 50  # 50 epochs

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

# ✅ Training Loop
epochs = 10  # 🚀 Increased for better learning
grad_accum_steps = 2
print(f"🚀 Training T5 model on {device} for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(loop):
        optimizer.zero_grad()

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_dataloader)
    print(f"✅ Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")

    # 🔎 Debugging: Check Model Output Every 5 Epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        
        sample_input = train_dataset.data[0][0]  # Get formatted input
        inputs = tokenizer(sample_input, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_length=150, 
                num_beams=3,
                early_stopping=True,
                temperature=0.7,  
                repetition_penalty=2.0,
                do_sample=True,
                top_p=0.85
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # 🔍 Print Model Output
        print(f"\n🔎 Model Output at Epoch {epoch+1}: {repr(generated_text)}\n{'='*50}")

        # 🔍 Debug: Show Tokenized Output
        print("\n🔍 DEBUG: Tokenized Input")
        print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

        print("\n🔍 DEBUG: Tokenized Output")
        print(tokenizer.convert_ids_to_tokens(output_ids[0]))

# ✅ Save Fine-Tuned Model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"✅ Fine-tuned model saved at {MODEL_SAVE_PATH}")
