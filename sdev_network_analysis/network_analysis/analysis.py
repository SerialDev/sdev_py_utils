import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab
import networkx as nx


def network_degree_distribution(D, out=False):
    if out == False:
        degrees = D.in_degree()
    else:
        degrees = D.out_degree()
    in_values = sorted(set(dict(degrees).values()))
    hist = {}
    for i in in_values:
        hist[i] = 0
        for j in dict(degrees).values():
            if j == i:
                hist[i] = hist[i] + 1
    return hist


def network_degree_distribution_viz(D, filename=None):
    in_hist = network_degree_distribution(D)
    out_hist = network_degree_distribution(D, True)
    plt.figure()
    plt.plot(in_hist.keys(), in_hist.values(), "ro-")  # in_degree
    plt.plot(out_hist.keys(), out_hist.values(), "bo-")  # out_degree
    plt.legend(["In-degree", "Out-degree"])
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("In Out degree distribution")
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)


def network_degree_distribution_viz_log(D, filename=None):
    in_hist = network_degree_distribution(D)
    out_hist = network_degree_distribution(D, True)
    plt.figure()
    plt.loglog(in_hist.keys(), in_hist.values(), "ro-")  # in_degree
    plt.loglog(out_hist.keys(), out_hist.values(), "bo-")  # out_degree
    plt.legend(["In-degree", "Out-degree"])
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("In Out degree distribution")
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)


def network_c_coefficients(G):
    cluster_coefficients = nx.clustering(G)
    return cluster_coefficients


def avg_cluster(G):
    ccs = nx.clustering(G)
    avg_clust = sum(ccs.values()) / len(ccs)

    return avg_clust


def network_components(G):
    components = nx.connected_component_subgraphs(G)
    graph_mc = components.__next__()
    return graph_mc


def network_betweenness_centrality(graph_mc):
    # betweenness centrality
    bet_cen = nx.betweenness_centrality(graph_mc)
    return bet_cen


def network_closeness_centrality(graph_mc):
    # closeness centrality
    clo_cen = nx.closeness_centrality(graph_mc)
    return clo_cen


def network_eigenvector_centrality(graph_mc):
    # eigenvector centrality
    eig_cen = nx.eigenvector_centrality(graph_mc)
    return eig_cen


def network_centrality_measures(G):
    components = network_components(G)
    bet_cen = network_betweenness_centrality(G)
    clo_cen = network_closeness_centrality(G)
    eig_cen = network_eigenvector_centrality(G)
    return (bet_cen, clo_cen, eig_cen)


def highest_centrality(cent_dict):
    """Returns a tuple (node,value) with the node
    with largest value from Networkx centrality dictionary."""
    # Create ordered tuple of centrality data
    cent_items = [(b, a) for (a, b) in cent_dict.items()]
    # Sort in descending order
    cent_items.sort()
    cent_items.reverse()
    return tuple(reversed(cent_items[0]))


def centrality_scatter(dict1, dict2, path=None, ylab="", xlab="", title="", line=False):
    # Create figure and drawing axis
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # Create items and extract centralities
    items1 = sorted(dict1.items())
    items2 = sorted(dict2.items())
    xdata = [b for a, b in items1]
    ydata = [b for a, b in items2]

    # Add each actor to the plot by ID
    for p in range(len(items1)):
        ax1.text(x=xdata[p], y=ydata[p], s=str(items1[p][0]), color="b")
    if line:
        # use NumPy to calculate the best fit
        slope, yint = plt.polyfit(xdata, ydata, 1)
        xline = plt.xticks()[0]
        yline = map(lambda x: slope * x + yint, xline)
        ax1.plot(xline, yline, ls="--", color="b")
    # Set new x- and y-axis limits
    plt.xlim((0.0, max(xdata) + (0.15 * max(xdata))))
    plt.ylim((0.0, max(ydata) + (0.15 * max(ydata))))
    # Add labels and save
    ax1.set_title(title)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    if path == None:
        plt.show()
    else:
        plt.savefig(path)


from collections import deque


def breadth_first_search(g, source):
    queue = deque([(None, source)])
    enqueued = set([source])
    while queue:
        parent, n = queue.popleft()
        yield parent, n
        new = set(g[n]) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])


def get_network_triad(g):
    for n1 in g.nodes:
        neighbors1 = set(g[n1])
        for n2 in filter(lambda x: x > n1, g.nodes):
            neighbors2 = set(g[n2])
            common = neighbors1 & neighbors2
            for n3 in filter(lambda x: x > n2, common):
                yield n1, n2, n3


# def avg_neigh_degree(g):
#     data = {}
#     for n in g.nodes():
#         if g.degree(n):
#             data[n] = float(sum(g.degree(i) for i in g[n])) / g.degree(n)
#     return data


def avg_neigh_degree(g):
    return dict(
        (n, float(sum(g.degree(i) for i in g[n])) / g.degree(n))
        for n in g.nodes()
        if g.degree(n)
    )


def get_top_keys(dictionary, top):
    items = dictionary.items()
    # items.sort(reverse=True, key=lambda x: x[1])
    items = sorted(list(items))
    return map(lambda x: x[0], items[:top])


def create_network_nodes(iterable_series, session_id, G):
    previous = "Na"
    id = {}

    for idx, i in enumerate(iterable_series):
        G.add_node(session_id, session=session_id, hostname=i)
        nodes = [x for x in i.split("/") if x is not ""]
        for j in nodes:
            G.add_node(j, hostname=i)
            id[idx] = j
            G = edge_increasew_or_create(G, session_id, j)
            G = edge_increasew_or_create(G, j, previous)
            previous = j
    return G, id


def edge_increasew_or_create(G, subject_id, object_id, add_w=1):
    if G.has_edge(subject_id, object_id):
        G[subject_id][object_id]["weight"] += add_w
    else:
        G.add_edge(subject_id, object_id, weight=add_w)
    return G


def prune_nx_by_weight(D, left_bound=1, right_bound=10):
    for node in D.nodes():
        edges = D.in_edges(node, data=True)
        if len(edges) > 0:  # some nodes have zero edges going into it
            # min_weight = min([edge[2]['weight'] for edge in edges])
            # for edge in list(edges):
            #     if edge[2]['weight'] > min_weight:
            #         D.remove_edge(edge[0], edge[1])
            for edge in list(edges):
                if left_bound <= edge[2]["weight"] <= right_bound:
                    D.remove_edge(edge[0], edge[1])
    return D


def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
        - G is a netorkx graph
        - node_order (optional) is a list of nodes where
              every
    node in G appears exactly once     - is a list of node lists, the where each node in the G Appears
              in exactly one node list
        - colors is a list of strings Indicating what color each
              partition Should Be
        If partitions is Specified, the cloud number of the colors needs to be
        Specified.
    """
    from matplotlib import pyplot, patches

    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adjacency_matrix, cmap="Blues", interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(
                patches.Rectangle(
                    (current_idx, current_idx),
                    LEN(module),  # Width
                    len(module),  # Height
                    face_color="none",
                    edge_color=color,
                    linewidth="1",
                )
            )
            current_idx += len(module)


def plot_coloured_edges(D, ids, filename="edges_width.png"):
    from matplotlib import colors
    from matplotlib import cm

    f = plt.figure()

    # minLineWidth = 1e-4
    minLineWidth = 3e2

    # for u, v, d in D.edges(data=True):
    #     d['weight'] = D.edges[u, v]['weight']*minLineWidth
    # edges,weights = zip(*nx.get_edge_attributes(D,'weight').items())
    # # Set Edge Color based on weight

    values = range(len(D.edges()))

    jet = plt.get_cmap("YlOrRd")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    colorList = []

    for i in range(len(D.edges())):
        colorVal = scalarMap.to_rgba(values[i])
        colorList.append(colorVal)

    nx.draw_networkx_edges(
        D,
        pos=nx.spring_layout(D),
        ax=f.add_subplot(111),
        edge_color=colorList,
        arrows=True,
        width=[
            np.log(d["weight"] * minLineWidth) * 0.3 for _, _, d in D.edges(data=True)
        ]
        # width=[d['weight'] * minLineWidth  for _, _, d in D.edges(data=True)]
    )
    nx.draw_networkx_nodes(
        D,
        pos=nx.spring_layout(D),
        ax=f.add_subplot(111),
        node_color=colorList,
        with_labels=True,
        labels=ids,
        # node_size=[np.log(d['weight']) for _, _, d in D.edges(data=True)]
        node_size=50
        # node_size=[d['weight'] for _, _, d in D.edges(data=True)]
    )
    f.savefig(filename)


def network_c_coefficients(G):
    cluster_coefficients = nx.clustering(G)
    return cluster_coefficients
