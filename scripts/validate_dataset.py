import json
import os

# Get the absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define correct paths
dataset_path = os.path.join(BASE_DIR, "data", "preprocessed_dataset.json")
log_file = os.path.join(BASE_DIR, "logs", "dataset_validation.log")

# Ensure log directory exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Debugging: Check if dataset file exists
if not os.path.exists(dataset_path):
    print(f"❌ Error: Preprocessed dataset not found at {dataset_path}")
    exit()

print(f"✅ Preprocessed dataset found at {dataset_path}")

# Load preprocessed dataset
with open(dataset_path, "r", encoding="utf-8") as file:
    try:
        dataset = json.load(file)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {dataset_path}")
        exit()

# Initialize variables for analysis
invalid_entries = []
command_types = {}

# Validate each entry in the dataset
for i, entry in enumerate(dataset):
    command = entry.get("command", "").strip()
    original_text = entry.get("original_text", "").strip()
    modified_text = entry.get("modified_text", "").strip()

    # Track command distribution dynamically
    command_types[command] = command_types.get(command, 0) + 1

    # Identify invalid entries (empty fields, incorrect transformations)
    if not command or not original_text or not modified_text:
        invalid_entries.append((i, entry))

# Save validation results to log file
with open(log_file, "w", encoding="utf-8") as log:
    log.write("✅ Dataset Validation Complete!\n")
    log.write(f"Total Entries: {len(dataset)}\n")
    log.write(f"Unique Commands: {len(command_types)}\n")
    log.write(f"Command Distribution: {json.dumps(command_types, indent=4)}\n")
    log.write(f"Invalid Entries: {len(invalid_entries)}\n")

    if invalid_entries:
        log.write("⚠️ Warning: Some entries have missing or incorrect data! Below are the first 10 invalid entries:\n")
        for idx, entry in invalid_entries[:10]:  # Show first 10 invalid entries for review
            log.write(f"\nEntry {idx}:\n{json.dumps(entry, indent=4, ensure_ascii=False)}\n")
    else:
        log.write("✅ No missing or corrupt entries found!\n")

print(f"✅ Dataset validation complete! Results saved to {log_file}")
