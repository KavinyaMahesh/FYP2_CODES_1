from rank_bm25 import BM25Okapi
import numpy as np

def build_bm25(documents):

    corpus = [doc.page_content.split() for doc in documents]

    bm25 = BM25Okapi(corpus)

    return bm25


def hybrid_search(query,vector_results,bm25,documents):

    tokenized = query.split()

    bm25_scores = bm25.get_scores(tokenized)

    hybrid_scores = []

    for i,doc in enumerate(documents):

        vector_score = 1

        score = 0.6*vector_score + 0.4*bm25_scores[i]

        hybrid_scores.append((doc,score))

    hybrid_scores.sort(key=lambda x:x[1],reverse=True)

    return hybrid_scores[:5]