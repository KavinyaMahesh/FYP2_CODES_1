def differential_diagnosis(retrieved_docs):

    scores = {}

    for doc in retrieved_docs:

        disease = doc.metadata["disease"]

        if disease not in scores:
            scores[disease]=0

        scores[disease]+=1

    ranked = sorted(scores.items(),key=lambda x:x[1],reverse=True)

    return ranked[:3]