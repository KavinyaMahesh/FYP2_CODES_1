def query_graph(graph,symptoms):

    related_diseases = []

    for node,data in graph.nodes(data=True):

        if data["type"]=="disease":

            for s in symptoms:

                if s in node.lower():
                    related_diseases.append(node)

    return related_diseases