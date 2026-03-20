symptom_ontology = {
    "tummy ache": "abdominal pain",
    "stomach ache": "abdominal pain",
    "breathlessness": "shortness of breath",
    "high temperature": "fever",
    "throwing up": "vomiting"
}

def normalize_symptoms(text):

    text = text.lower()

    for k,v in symptom_ontology.items():
        text = text.replace(k,v)

    return text