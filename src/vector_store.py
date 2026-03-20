import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "processed_data.json")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# Load processed data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data:
    documents.append(
        Document(
            page_content=item["content"],
            metadata={
                "category": item["category"],
                "disease": item["disease"],
                "source": item["file_name"]
            }
        )
    )

print("Documents loaded:", len(documents))

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create empty DB
vector_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model
)

# Insert chunks in batches
BATCH_SIZE = 1000

for i in range(0, len(chunks), BATCH_SIZE):

    batch = chunks[i:i+BATCH_SIZE]

    vector_db.add_documents(batch)

    print(f"Inserted batch {i} → {i + len(batch)}")

# Persist database
vector_db.persist()

print("Vector database created successfully!")