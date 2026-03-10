import os
import fitz  # PyMuPDF
import json
import re
from tqdm import tqdm

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = os.path.join(BASE_DIR, "data", "KB_phase2_drive")

output_path = os.path.join(BASE_DIR, "processed_data.json")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,]', '', text)
    return text


def load_data(base_path):
    dataset = []

    for disease in os.listdir(base_path):
        disease_path = os.path.join(base_path, disease)

        if not os.path.isdir(disease_path):
            continue

        print(f"\nProcessing: {disease}")

        for file in tqdm(os.listdir(disease_path)):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(disease_path, file)

                text = extract_text_from_pdf(pdf_path)

                if text.strip():
                    dataset.append({
                        "disease": disease,
                        "file_name": file,
                        "content": clean_text(text)
                    })

    return dataset


def main():
    data = load_data(BASE_PATH)

    print(f"\nTotal docs: {len(data)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved processed_data.json")


if __name__ == "__main__":
    main()