import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "vector_db"),
    embedding_function=embedding_model
)

print("Vector DB loaded\n")

queries = [
    "I have chest pain and shortness of breath",
    "persistent cough and fever",
    "abdominal pain and vomiting"
]

for query in queries:

    print("=" * 60)
    print("Query:", query)

    results = vector_db.similarity_search(query, k=3)

    for i, res in enumerate(results):

        print(f"\nResult {i+1}")
        print(res.page_content[:300])
        print("Category:", res.metadata["category"])
        print("Disease:", res.metadata["disease"])
        print("Source:", res.metadata["source"])