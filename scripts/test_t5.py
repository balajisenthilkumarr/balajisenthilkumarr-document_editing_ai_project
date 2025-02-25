import json
import os
import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ✅ Get project root directory dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Define absolute paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "t5_finetuned")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_dataset.json")

# ✅ Ensure model files exist before loading
if not os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
    print(f"❌ Model file not found in {MODEL_PATH}")
    exit()

print(f"✅ Checking model files in {MODEL_PATH}...\n")
print(os.listdir(MODEL_PATH))  # Show all files in model directory

# ✅ Load tokenizer from fine-tuned model
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, legacy=False)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    exit()

# ✅ Load fine-tuned model correctly
try:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ✅ Device setup (ensure inference is on GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Function to normalize text for better matching
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces
    text = text.replace("..", ".")  # Fix double periods
    return text

# ✅ Function to check if model is just copying input
def is_copying_input(predicted_text, original_text):
    return normalize_text(predicted_text) == normalize_text(original_text)

# ✅ Function to generate predictions with correct format
def generate_prediction(command, original_text):
    # ✅ FIX: Use the same format as training
    formatted_input = f"Task: {command}\nText: {original_text}"

    print(f"\n🔍 Debugging Tokenization for Input: {formatted_input}")
    inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True).to(device)

    print(f"🛠 Tokenized Input IDs: {inputs['input_ids']}")
    print(f"🛠 Tokenized Attention Mask: {inputs['attention_mask']}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=512, 
            num_beams=5,  # Increased to improve results
            early_stopping=True
        )

    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not predicted_text:
        print("⚠️ Warning: Model returned an empty output!")

    return predicted_text

# ✅ Load test dataset
if os.path.exists(TEST_DATA_PATH):
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    print(f"✅ Loaded {len(test_data)} test cases from {TEST_DATA_PATH}.")
else:
    print("⚠️ Test dataset not found!")
    exit()

# ✅ Running a manual test before evaluation
print("\n🔎 Running a manual test...")

sample_command = "Replace 'CEO' with 'Managing Director'"
sample_text = "The CEO will address the employees."
predicted_output = generate_prediction(sample_command, sample_text)

print("\n🔎 Model Output:", repr(predicted_output))

# ✅ Evaluate the model
correct_predictions = 0
total_samples = len(test_data)
incorrect_cases = []
copying_cases = []

print("\n✅ Running model evaluation...\n")

for i, entry in enumerate(test_data):
    try:
        command = entry["command"]
        original_text = entry["original_text"]
        expected_output = entry["modified_text"]

        predicted_output = generate_prediction(command, original_text)

        print(f"\n🔹 Test {i+1}/{total_samples}")
        print(f"📝 Command: {command}")
        print(f"📌 Original: {original_text}")
        print(f"✅ Expected: {expected_output}")
        print(f"🔎 Predicted: {predicted_output}")

        # Apply normalization before comparison
        if normalize_text(predicted_output) == normalize_text(expected_output):
            correct_predictions += 1
        else:
            print("❌ Mismatch detected!")
            incorrect_cases.append({
                "command": command,
                "original": original_text,
                "expected": expected_output,
                "predicted": predicted_output
            })

        # Check if the model is copying input text
        if is_copying_input(predicted_output, original_text):
            print("🚨 Model is copying input instead of modifying!")
            copying_cases.append({
                "command": command,
                "original": original_text,
                "expected": expected_output,
                "predicted": predicted_output
            })

    except ValueError:
        print(f"⚠️ Skipping test {i+1}: Invalid format in test dataset")

# ✅ Print evaluation results
accuracy = (correct_predictions / total_samples) * 100
print(f"\n✅ Model Evaluation Complete! Accuracy: {accuracy:.2f}%")

# ✅ Save incorrect cases for debugging
if incorrect_cases:
    debug_path = os.path.join(PROJECT_ROOT, "data", "incorrect_predictions.json")
    with open(debug_path, "w", encoding="utf-8") as debug_file:
        json.dump(incorrect_cases, debug_file, indent=4)
    print(f"⚠️ {len(incorrect_cases)} incorrect predictions saved to {debug_path} for review.")
else:
    print("🎉 All predictions matched! No incorrect cases.")

# ✅ Save cases where the model is copying input
if copying_cases:
    copy_debug_path = os.path.join(PROJECT_ROOT, "data", "copying_predictions.json")
    with open(copy_debug_path, "w", encoding="utf-8") as copy_debug_file:
        json.dump(copying_cases, copy_debug_file, indent=4)
    print(f"🚨 {len(copying_cases)} cases where the model is copying input saved to {copy_debug_path}.")

print("\n✅ Testing complete!")
