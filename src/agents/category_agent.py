category_rules = {
"Respiratory_diseases":["cough","wheezing","shortness of breath"],
"Cardiovascular_diseases":["chest pain","palpitations"],
"Gastrointestinal_diseases":["abdominal pain","vomiting","diarrhea"]
}

def classify_category(symptoms):

    for category,values in category_rules.items():

        for s in symptoms:
            if s in values:
                return category

    return "Infectious_diseases"