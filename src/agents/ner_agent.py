import re

medical_symptoms = [
    "fever","cough","chest pain","shortness of breath",
    "abdominal pain","vomiting","diarrhea","fatigue",
    "wheezing","palpitations"
]

def extract_symptoms(text):

    text = text.lower()

    found = []

    for symptom in medical_symptoms:

        if re.search(rf"\b{symptom}\b",text):
            found.append(symptom)

    return found