import json
import os
import re
from docx import Document

def extract_text_from_docx(file_path):
    """Extract text from a .docx file while preserving paragraph structure."""
    try:
        doc = Document(file_path)
        return "\n\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {str(e)}")
        return ""

def load_existing_data(file_path):
    """Load existing extracted dataset if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: preprocessed_dataset.json is corrupted. Resetting file.")
                return []
    return []

def verify_transformation(command, original_text, modified_text):
    """Verify that the transformation was applied correctly."""
    if not modified_text or original_text == modified_text:
        return False  # No change detected

    if command == "Capitalize all words":
        original_words = original_text.split()
        modified_words = modified_text.split()

        if len(original_words) != len(modified_words):
            return False  # Word count mismatch

        return all(m_word == o_word.capitalize() for o_word, m_word in zip(original_words, modified_words))

    elif command == "Remove the first paragraph":
        original_paras = original_text.split("\n\n")
        modified_paras = modified_text.split("\n\n")
        return len(original_paras) > 1 and original_paras[0] not in modified_text

    elif command == "Remove the last sentence":
        original_sentences = len(re.findall(r'[.!?]', original_text))
        modified_sentences = len(re.findall(r'[.!?]', modified_text))
        return original_sentences > modified_sentences

    elif command.startswith("Replace"):
        match = re.search(r"Replace '(.+?)' with '(.+?)'", command)
        if match:
            original_term, replacement = match.groups()
            return (original_term.lower() in original_text.lower() and 
                    replacement.lower() in modified_text.lower())

    elif command == "Remove bullet points":
        return "•" in original_text and "•" not in modified_text

    elif command == "Remove numbers":
        return any(c.isdigit() for c in original_text) and not any(c.isdigit() for c in modified_text)

    elif command == "Convert to uppercase":
        return modified_text.isupper()

    return True  # Default to assuming transformation was applied

def extract_dataset(base_dir):
    """Extract text from documents, validate transformations, and update the dataset."""
    training_dataset_dir = os.path.join(base_dir, 'training_dataset')
    metadata_path = os.path.join(training_dataset_dir, 'metadata.json')  
    output_path = os.path.join(base_dir, 'preprocessed_dataset.json')

    if not os.path.exists(metadata_path):
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        return

    # Load metadata and existing dataset
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    existing_data = load_existing_data(output_path)
    existing_original_texts = {entry["original_text"] for entry in existing_data}

    new_data = []
    skipped = 0
    invalid = 0

    for entry in metadata:
        original_file = os.path.join(training_dataset_dir, entry["original_file"])
        modified_file = os.path.join(training_dataset_dir, entry["modified_file"])
        command = entry["command"]

        if not (os.path.exists(original_file) and os.path.exists(modified_file)):
            print(f"⚠️ Warning: Missing files for '{command}'. Skipping.")
            skipped += 1
            continue

        original_text = extract_text_from_docx(original_file)
        modified_text = extract_text_from_docx(modified_file)

        if not original_text or not modified_text:
            print(f"⚠️ Warning: Empty text extracted for '{command}'. Skipping.")
            skipped += 1
            continue

        if original_text in existing_original_texts:
            print(f"🔄 Skipping duplicate document for '{command}'.")
            skipped += 1
            continue

        if not verify_transformation(command, original_text, modified_text):
            print(f"⚠️ Warning: Invalid transformation for '{command}'. Skipping.")
            invalid += 1
            continue

        new_data.append({
            "command": command,
            "original_text": original_text,
            "modified_text": modified_text
        })

    if new_data:
        updated_data = existing_data + new_data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Added {len(new_data)} new entries. Updated dataset saved at: {output_path}")
    else:
        print("\n✅ No new documents added.")

    print(f"📊 Statistics: {skipped} skipped, {invalid} invalid transformations")

if __name__ == "__main__":
    base_directory = "data"  # Main folder containing training_dataset
    extract_dataset(base_directory)
