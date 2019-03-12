import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab
import networkx as nx
import pandas as pd
from bokeh.io import show, output_file, output_notebook, push_notebook
from bokeh.plotting import save, figure
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models.graphs import from_networkx

import datashader as ds
import datashader.transfer_functions as tf
from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
from itertools import chain

def plot_labelled_bokeh_ni(G, id, filename="test.html"):
    plot = figure(title="Title", tools="", x_range=(-1.5, 1.5),
                  y_range=(-1.5, 1.5), toolbar_location=None)
    graph = from_networkx(G, nx.spring_layout)
    plot.renderers.append(graph)

    x, y = zip(*graph.layout_provider.graph_layout.values())

    node_labels = nx.get_node_attributes(G, 'hostname')

    source = ColumnDataSource({'x': x, 'y': y,
     'label': [node_labels[id[i]] for i in range(len(x))]})


    labels = LabelSet(x='x', y='y', text='label', source=source,
                          background_fill_color='white')

    plot.renderers.append(labels)

    save(plot, filename)

def plot_bokeh_network_i(G,id, layout=nx.circular_layout, output ="testing.html"):
    plot = Plot(plot_width=400, plot_height=400,
                x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    plot.title.text = "Graph Interaction Demonstration"
    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool(), WheelZoomTool())


    graph_renderer = from_networkx(G, layout, scale=1, center=(0,0))



    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])


    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()


    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())

    node_labels = nx.get_node_attributes(G, 'hostname')

    source = ColumnDataSource({'x': x, 'y': y,
     'label': [node_labels[id[i]] for i in range(len(x))]})


    labels = LabelSet(x='x', y='y', text='label', source=source,
                          background_fill_color='white')


    plot.renderers.append(graph_renderer)

    output_file(output)
    save(plot)



def viz_network(graph):
    edge_labels=dict([((u,v,),d['weight'])
                     for u,v,d in graph.edges(data=True)])
    red_edges = [('C','D'),('D','A')]
    edge_colors = ['black' if not edge in red_edges else 'red' for edge in graph.edges()]

    pos=nx.spring_layout(graph)

    nx.draw_networkx_edge_labels(graph,pos,edge_labels=edge_labels)
    nx.draw(graph,pos, node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)


def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    # cut = 1.00
    # xmax = cut * max(xx for xx, yy in pos.values())
    # ymax = cut * max(yy for xx, yy in pos.values())
    # plt.xlim(0, xmax)
    # plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig

def plot_holoviews_i(G, filename="graph.html"):
    import holoviews as hv

    extension = hv.extension('bokeh')
    renderer = hv.renderer('bokeh')
    graph_plot = hv.Graph.from_networkx(G, nx.layout.spring_layout)
    renderer.save(graph_plot, 'graph.html')

def ng(graph,name):
    graph.name = name
    return graph

cvsopts = dict(plot_height=400, plot_width=400)

def nodesplot(nodes, name=None, canvas=None, cat=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    aggregator=None if cat is None else ds.count_cat(cat)
    agg=canvas.points(nodes,'x','y',aggregator)
    return tf.spread(tf.shade(agg, cmap=["#FF3333"]), px=3, name=name)

def edgesplot(edges, name=None, canvas=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)

def nx_layout(graph):
    layout = nx.circular_layout(graph)
    data = [[node]+layout[node].tolist() for node in graph.nodes]

    nodes = pd.DataFrame(data, columns=['id', 'x', 'y'])
    nodes.set_index('id', inplace=True)

    edges = pd.DataFrame(list(graph.edges), columns=['source', 'target'])
    return nodes, edges


def graphplot(nodes, edges, name="", canvas=None, cat=None):
    if canvas is None:
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)

    np = nodesplot(nodes, name + " nodes", canvas, cat)
    ep = edgesplot(edges, name + " edges", canvas)
    return tf.stack(ep, np, how="over", name=name)

def nx_plot(graph, name=""):
    print(graph.name, len(graph.edges))
    nodes, edges = nx_layout(graph)

    direct = connect_edges(nodes, edges)
    bundled_bw005 = hammer_bundle(nodes, edges)
    bundled_bw030 = hammer_bundle(nodes, edges, initial_bandwidth=0.30)

    return [graphplot(nodes, direct,         graph.name),
            graphplot(nodes, bundled_bw005, "Bundled bw=0.05"),
            graphplot(nodes, bundled_bw030, "Bundled bw=0.30")]


def save_datashader_triplet(G, filename="datashader.html"):
    plots = [nx_plot(G)]
    u = tf.Images(*chain.from_iterable(plots)).cols(3)
    with open(filename, 'w') as f:
        f.write(u._repr_html_())


import json

def to_flare_json(df, filename):
    """Convert dataframe into nested JSON as in flare files used for D3.js"""
    flare = dict()
    d = {"name":"flare", "children": []}

    for index, row in df.iterrows():
        parent = row[0]
        child = row[1]
        child_size = row[2]

        # Make a list of keys
        key_list = []
        for item in d['children']:
            key_list.append(item['name'])

        #if 'parent' is NOT a key in flare.JSON, append it
        if not parent in key_list:
            d['children'].append({"name": parent, "children":[{"value": child_size, "name": child}]})
        # if parent IS a key in flare.json, add a new child to it
        else:
            d['children'][key_list.index(parent)]['children'].append({"value": child_size, "name": child})
    flare = d
    # export the final result to a json file
    with open(filename +'.json', 'w') as outfile:
        json.dump(flare, outfile, indent=4)
    return ("Done")
