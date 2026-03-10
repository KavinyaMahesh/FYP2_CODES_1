import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector DB
vector_db = Chroma(
    persist_directory=os.path.join(BASE_DIR, "vector_db"),
    embedding_function=embedding_model
)

# LLM
llm = Ollama(model="gemma:2b")

# Prompt Template
prompt_template = """
You are a medical assistant specialized in respiratory diseases.

Use the context below to answer the user's symptoms.

Context:
{context}

User Symptoms:
{question}

Based on the information:
1. Suggest the most likely respiratory disease
2. Explain why
3. Mention important symptoms

Answer clearly.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def retrieve_context(query):
    results = vector_db.similarity_search(query, k=4)

    context = "\n\n".join([doc.page_content for doc in results])
    return context


def rag_query(question):

    context = retrieve_context(question)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    return response


def main():

    print("\n🫁 Respiratory Disease Assistant\n")

    while True:

        query = input("Enter symptoms (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag_query(query)

        print("\n🧠 Diagnosis Suggestion:\n")
        print(answer)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()