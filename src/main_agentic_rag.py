import os

from agents.ontology_agent import normalize_symptoms
from agents.ner_agent import extract_symptoms
from agents.category_agent import classify_category

from diagnosis.differential_diagnosis import differential_diagnosis
from diagnosis.verification_agent import verify_diagnosis
from diagnosis.confidence_agent import compute_confidence

from knowledge_graph.build_graph import build_graph
from knowledge_graph.graph_query import query_graph

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embedding_model = HuggingFaceEmbeddings(
model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
persist_directory=os.path.join(BASE_DIR,"vector_db"),
embedding_function=embedding_model
)


def run_pipeline(query):

    normalized = normalize_symptoms(query)

    symptoms = extract_symptoms(normalized)

    category = classify_category(symptoms)

    retrieved = vector_db.similarity_search(query,k=10)

    graph = build_graph(retrieved)

    graph_candidates = query_graph(graph,symptoms)

    top_diseases = differential_diagnosis(retrieved)

    best_disease = top_diseases[0][0]

    verified = verify_diagnosis(best_disease,str(retrieved))

    confidence = compute_confidence(top_diseases[0][1],len(retrieved))

    return {
        "symptoms":symptoms,
        "category":category,
        "differential":top_diseases,
        "diagnosis":best_disease,
        "confidence":confidence,
        "verified":verified
    }


def main():

    print("\nAdvanced Agentic Medical Diagnosis System\n")

    while True:

        q=input("Enter symptoms (exit to quit): ")

        if q=="exit":
            break

        result = run_pipeline(q)

        print("\nSymptoms:",result["symptoms"])
        print("Category:",result["category"])

        print("\nDifferential Diagnosis:")

        for d in result["differential"]:
            print(d)

        print("\nFinal Diagnosis:",result["diagnosis"])
        print("Confidence:",result["confidence"])
        print("Verified:",result["verified"])

        print("\n---------------------------")


if __name__=="__main__":
    main()