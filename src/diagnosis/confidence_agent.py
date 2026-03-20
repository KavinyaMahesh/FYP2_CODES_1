def compute_confidence(score,evidence_count):

    confidence = (score*0.6)+(evidence_count*0.4)

    if confidence>1:
        confidence=1

    return round(confidence,2)