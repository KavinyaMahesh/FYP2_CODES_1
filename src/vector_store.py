import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load data
with open(os.path.join(BASE_DIR, "processed_data.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to documents
documents = []
for item in data:
    documents.append(
        Document(
            page_content=item["content"],
            metadata={
                "disease": item["disease"],
                "source": item["file_name"]
            }
        )
    )

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Vector DB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=os.path.join(BASE_DIR, "vector_db")
)

vector_db.persist()

print("Vector DB created successfully!")