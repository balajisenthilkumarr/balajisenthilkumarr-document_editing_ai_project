import json
from sklearn.model_selection import train_test_split

# Load dataset
with open("data/preprocessed_dataset.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)

# Split into 80% train, 20% test
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Save the split datasets
with open("data/train_dataset.json", "w", encoding="utf-8") as file:
    json.dump(train_data, file, indent=4)

with open("data/test_dataset.json", "w", encoding="utf-8") as file:
    json.dump(test_data, file, indent=4)

print(f"✅ Train-Test Split Done! Train: {len(train_data)}, Test: {len(test_data)}")
