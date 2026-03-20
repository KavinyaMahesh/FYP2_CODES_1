def verify_diagnosis(disease,evidence):

    if disease.lower() in evidence.lower():
        return True

    return False