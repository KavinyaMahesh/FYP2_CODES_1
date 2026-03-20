import os
import fitz
import json
import re
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# New dataset root
BASE_PATH = os.path.join(BASE_DIR, "data", "KB_phase2")

OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data.json")


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

    for category in os.listdir(base_path):

        category_path = os.path.join(base_path, category)

        if not os.path.isdir(category_path):
            continue

        print(f"\nCATEGORY: {category}")

        for disease in os.listdir(category_path):

            disease_path = os.path.join(category_path, disease)

            if not os.path.isdir(disease_path):
                continue

            print(f"  Disease: {disease}")

            files = os.listdir(disease_path)

            for file in tqdm(files):

                if file.endswith(".pdf"):

                    pdf_path = os.path.join(disease_path, file)

                    text = extract_text_from_pdf(pdf_path)

                    if text.strip():

                        dataset.append({
                            "category": category,
                            "disease": disease,
                            "file_name": file,
                            "content": clean_text(text)
                        })

    return dataset


def main():

    data = load_data(BASE_PATH)

    print(f"\nTotal documents processed: {len(data)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("processed_data.json created successfully")


if __name__ == "__main__":
    main()