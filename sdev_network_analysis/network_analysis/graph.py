import networkx as nx

def create_network(iterable_series, G):
    previous = "Na"
    id = {}
    for idx, i in enumerate(iterable_series):
        G.add_node(i, hostname=i)
        id[idx] = i
        G.add_edge(i, previous, weight=1)
        previous = i
    return G, id


