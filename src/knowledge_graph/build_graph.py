import networkx as nx

G = nx.Graph()

def build_graph(documents):

    for doc in documents:

        disease = doc.metadata["disease"]
        category = doc.metadata["category"]

        G.add_node(disease,type="disease")
        G.add_node(category,type="category")

        G.add_edge(disease,category)

    return G