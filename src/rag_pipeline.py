import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate


# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Load vector database
vector_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model
)


# LLM (lightweight model for low RAM systems)
llm = Ollama(model="gemma:2b")


# Prompt Template
prompt_template = """
You are a medical assistant.

The knowledge base contains medical information about multiple disease categories including:

- Respiratory diseases
- Cardiovascular diseases
- Gastrointestinal diseases
- Infectious diseases

Use ONLY the information provided in the context.

Context:
{context}

User symptoms:
{question}

Instructions:

1. Identify the most likely disease from the context
2. Use the disease category given in the context
3. Explain why the symptoms match the disease

Answer format:

Disease:
Category:
Explanation:
"""


prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# -----------------------------
# Retrieve relevant documents
# -----------------------------
def retrieve_context(query):

    results = vector_db.similarity_search(query, k=4)

    context = ""

    for doc in results:

        context += f"""
Category: {doc.metadata['category']}
Disease: {doc.metadata['disease']}
Source: {doc.metadata['source']}

Content:
{doc.page_content}

---
"""

    return context


# -----------------------------
# RAG Query
# -----------------------------
def rag_query(question):

    context = retrieve_context(question)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    return response


# -----------------------------
# CLI Interface
# -----------------------------
def main():

    print("\n🧠 Medical Diagnosis Assistant\n")

    while True:

        query = input("Enter symptoms (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag_query(query)

        print("\n🔎 Diagnosis Suggestion:\n")
        print(answer)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()