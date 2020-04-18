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
    plt.plot(in_hist.keys(),in_hist.values(),'ro-')#in_degree
    plt.plot(out_hist.keys(),out_hist.values(),'bo-')#out_degree
    plt.legend(['In-degree', 'Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('In Out degree distribution')
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)


def network_degree_distribution_viz_log(D, filename=None):
    in_hist = network_degree_distribution(D)
    out_hist = network_degree_distribution(D, True)
    plt.figure()
    plt.loglog(in_hist.keys(),in_hist.values(),'ro-')#in_degree
    plt.loglog(out_hist.keys(),out_hist.values(),'bo-')#out_degree
    plt.legend(['In-degree', 'Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('In Out degree distribution')
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
    cent_items=[(b,a) for (a,b) in cent_dict.items()]
    # Sort in descending order
    cent_items.sort()
    cent_items.reverse()
    return tuple(reversed(cent_items[0]))

def centrality_scatter(dict1,dict2,path=None,
                       ylab="",xlab="",title="",line=False):
    # Create figure and drawing axis
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)
    # Create items and extract centralities
    items1 = sorted(dict1.items())
    items2 = sorted(dict2.items())
    xdata=[b for a,b in items1]
    ydata=[b for a,b in items2]

    # Add each actor to the plot by ID
    for p in range(len(items1)):
        ax1.text(x=xdata[p], y=ydata[p],s=str(items1[p][0]), color="b")
    if line:
        # use NumPy to calculate the best fit
        slope, yint = plt.polyfit(xdata,ydata,1)
        xline = plt.xticks()[0]
        yline = map(lambda x: slope*x+yint,xline)
        ax1.plot(xline,yline,ls='--',color='b')
    # Set new x- and y-axis limits
    plt.xlim((0.0,max(xdata)+(.15*max(xdata))))
    plt.ylim((0.0,max(ydata)+(.15*max(ydata))))
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
        parent,n = queue.popleft()
        yield parent,n
        new = set(g[n]) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])


def get_network_triad(g):
    for n1 in g.nodes:
        neighbors1 = set(g[n1])
        for n2 in filter(lambda x: x>n1, g.nodes):
            neighbors2 = set(g[n2])
            common = neighbors1 & neighbors2
            for n3 in filter(lambda x: x>n2, common):
                yield n1,n2,n3


# def avg_neigh_degree(g):
#     data = {}
#     for n in g.nodes():
#         if g.degree(n):
#             data[n] = float(sum(g.degree(i) for i in g[n])) / g.degree(n)
#     return data


def avg_neigh_degree(g):
    return dict((n,float(sum(g.degree(i) for i in g[n]))/
              g.degree(n)) for n in g.nodes() if g.degree(n))


def get_top_keys(dictionary, top):
    items = dictionary.items()
    #items.sort(reverse=True, key=lambda x: x[1])
    items = sorted(list(items))
    return map(lambda x: x[0], items[:top])

