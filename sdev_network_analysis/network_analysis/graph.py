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


def nx_simple_random_sample(G, num_sample_nodes):
    import random

    return random.sample(G.nodes(), num_sample_nodes)


def nx_degree_summary(G):
    # Get degree summary
    degree_sequence = list(G.degree())
    avg_degree = np.mean(np.array(degree_sequence)[:, 1].astype(np.float))
    med_degree = np.median(np.array(degree_sequence)[:, 1].astype(np.float))
    max_degree = max(np.array(degree_sequence)[:, 1].astype(np.float))
    min_degree = np.min(np.array(degree_sequence)[:, 1].astype(np.float))

    print("Number of nodes : " + str(G.number_of_nodes()))
    print("Number of edges : " + str(G.number_of_edges()))
    print("Maximum degree : " + str(max_degree))
    print("Minimum degree : " + str(min_degree))
    print("Average degree : " + str(avg_degree))
    print("Median degree : " + str(med_degree))
    return None


def nx_plot_degree_dist(G):
    """
    # Visualize distribution of degrees per node
    The distribution of the number of degrees per node should be really close to the mean.
    The probability to observe a high number of nodes decreases exponentially.
    """
    import matplotlib

    degree_freq = np.array(nx.degree_histogram(G)).astype("float")
    plt.figure(figsize=(12, 8))
    plt.stem(degree_freq, use_line_collection=True)
    # plt.bar(degree_freq[:2], degree_freq[:2])
    plt.ylabel("Frequence")
    plt.xlabel("Degree")
    plt.show()
    return None


def overlap(H, edge):
    """
    Function for computing edge overlap , defined for non - isolated edges .
    examine the so-called weak tie hypothesis
    “Structure and tie strengths in mobile communication networks” by
    J.-P. Onnela, J. Saramaki 2007
    Oij = nij /((ki − 1) + (kj − 1) − nij)

    """
    import numpy as np

    node_i = edge[0]
    node_j = edge[1]
    degree_i = H.degree(node_i)
    degree_j = H.degree(node_j)
    neigh_i = set(H.neighbors(node_i))
    neigh_j = set(H.neighbors(node_j))
    neigh_ij = neigh_i & neigh_j
    num_cn = len(neigh_ij)
    if degree_i > 1 or degree_j > 1:
        return float(num_cn) / (degree_i + degree_j - num_cn - 2)
    else:
        return np.NaN


def nx_overlap_values(G):
    # Compute some overlap values for different ER networks .
    overlap_values = np.array([overlap(G, edge) for edge in G.edges()])
    overlap_values = overlap_values[~np.isnan(overlap_values)]
    return overlap_values


def nx_plot_edge_overlap(G):
    # Compute some overlap values for different ER networks .
    ps = np.arange(0, 1.001, 0.05)
    fig = plt.figure(figsize=(10, 10))
    line1 = plt.plot(ps, nx_overlap_values(G), marker="o", markersize=10)
    line2 = plt.plot(ps, ps)
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Graph p- parameter ", fontsize=15)
    plt.ylabel(" Average overlap <O>", fontsize=15)
    plt.show()
    return None


def hits(graph, iter_count=20):
    # Given input graph, this method is the implementation of hits algorithms.
    # It returns the hubs and authorities score
    # Approach using the power iteration method
    nodes = graph.nodes()
    nodes_count = len(nodes)
    matrix = nx.to_numpy_matrix(graph, nodelist=nodes)
    hubs_score = np.ones(nodes_count)
    auth_score = np.ones(nodes_count)
    H = matrix * matrix.T
    A = matrix.T * matrix
    for i in range(iter_count):
        hubs_score = hubs_score * H
        auth_score = auth_score * A
        hubs_score = hubs_score / np.linalg.norm(hubs_score)
        auth_score = auth_score / np.linalg.norm(auth_score)
    hubs_score = np.array(hubs_score).reshape(
        -1,
    )
    auth_score = np.array(auth_score).reshape(
        -1,
    )
    hubs = dict(zip(nodes, hubs_score))
    authorities = dict(zip(nodes, auth_score))
    return hubs, authorities


def nx_top_k_hubs(graph, k=10):
    # Given a graph, this method returns top k hubs

    import operator

    hubs = hits(graph)[0]
    return sorted(hubs.items(), key=operator.itemgetter(1), reverse=True)[:k]


def nx_top_k_authorities(graph, k=10):
    # Given a graph, this method returns top k authorities
    import operator

    auth = hits(graph)[1]
    return sorted(auth.items(), key=operator.itemgetter(1), reverse=True)[:k]
