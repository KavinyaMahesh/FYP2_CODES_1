import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector DB
vector_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "vector_db"),
    embedding_function=embedding_model
)

print("✅ Vector DB loaded successfully!\n")

# Test queries
queries = [
    "I have fever and cough",
    "shortness of breath and wheezing",
    "chest pain and difficulty breathing"
]

# Run retrieval
for query in queries:
    print("=" * 60)
    print(f"🔍 Query: {query}\n")

    results = vector_db.similarity_search(query, k=3)

    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print(res.page_content[:300])  # show first 300 chars
        print("Metadata:", res.metadata)
        print("-" * 40)