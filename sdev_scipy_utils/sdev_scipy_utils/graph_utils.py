import networkx as nx

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def bundle_nodes(G, prefixes):
    '''
    * ---------------Function---------------
    * Bundles nodes in a graph based on their prefixes
    * ----------------Returns---------------
    * -> NetworkX graph object :: The modified graph with bundled nodes
    * ----------------Params----------------
    * G :: NetworkX graph object :: The input graph
    * prefixes :: list of str :: The prefixes to bundle nodes by
    * ----------------Usage-----------------
    * ```
    * G = nx.Graph()
    * G.add_node("node1", Labels=["prefix1_label"])
    * G.add_node("node2", Labels=["prefix2_label"])
    * prefixes = ["prefix1", "prefix2"]
    * G = bundle_nodes(G, prefixes)
    * ```
    * ----------------Notes-----------------
    * This function modifies the input graph by removing nodes that match the given prefixes and replacing them with a master node for each prefix. It then reconnects the edges to the master nodes.
    >>>
    # Example usage
    G = nx.DiGraph()
    G.add_node(1, Labels="Node A")
    G.add_node(2, Labels="Node B")
    G.add_node(3, Labels="Other C")
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    prefixes = ["Node ", "Other"]
    G = bundle_nodes(G, prefixes)
    # Display the graph
    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges())

    '''
    # Create a master node for each prefix
    prefix_master_nodes = {prefix: f"{prefix.rstrip()}_master" for prefix in prefixes}
    for master_node in prefix_master_nodes.values():
        G.add_node(master_node, label=master_node)

    # Maintain edges that should be bundled
    bundled_edges = {prefix: [] for prefix in prefixes}
    connections_between_prefixes = set()

    # Iterate through nodes and bundle them based on their prefixes
    for node, data in list(G.nodes(data=True)):
        labels = data.get('Labels', [])
        if not isinstance(labels, list):
            labels = [labels]
        for prefix in prefixes:
            if any(label.startswith(prefix) for label in labels) and node not in prefix_master_nodes.values():
                for pred in G.predecessors(node):
                    bundled_edges[prefix].append((pred, prefix_master_nodes[prefix]))
                for succ in G.successors(node):
                    bundled_edges[prefix].append((prefix_master_nodes[prefix], succ))
                for other_prefix in prefixes:
                    if other_prefix != prefix:
                        if any(G.has_edge(pred, node) for pred in G.predecessors(node)) or any(G.has_edge(node, succ) for succ in G.successors(node)):
                            connections_between_prefixes.add((prefix_master_nodes[prefix], prefix_master_nodes[other_prefix]))
                G.remove_node(node)

    # Add the bundled edges to the graph
    for edges in bundled_edges.values():
        for edge in edges:
            G.add_edge(*edge)

    # Connect master nodes if needed
    for edge in connections_between_prefixes:
        G.add_edge(*edge)

    return G


def flatten_graph(G):
    '''
    * ---------------Function---------------
    * This function flattens a graph into a new graph where each attribute becomes a node.
    * ----------------Returns---------------
    * -> nx.DiGraph: A new directed graph where each attribute is a node.
    * ----------------Params----------------
    * G: <any> The input graph to be flattened.
    * ----------------Usage-----------------
    * ```
    * G = nx.Graph()
    * G.add_node(1, Props={'color': 'red', 'shape': 'circle'})
    * G.add_edge(1, 2, Props={'label': 'edge label'})
    * flat_G = flatten_graph(G)
    * ```
    * ----------------Notes-----------------
    * This function assumes that the input graph is a networkx graph.
    * It creates nodes for each attribute in the original graph and edges to represent the relationships between them.
    * The resulting graph can be used for visualization or further processing.
    '''
    flat_G = nx.DiGraph()

    # Create nodes for each attribute and edges for each attribute key
    for node, data in G.nodes(data=True):
        # Use the actual node ID as the label
        flat_G.add_node(node, label=f"Node {node}")

        # Create nodes and edges for each attribute in the 'Props' dictionary
        if 'Props' in data and data['Props']:  # Skip if 'Props' is empty
            for key, value in data['Props'].items():
                attr_node = f"{node}_{key}"
                value_node = f"{node}_{key}_value"
                value_label = str(value)  # Convert value to string
                flat_G.add_node(attr_node, label=key)
                flat_G.add_node(value_node, label=value_label)
                flat_G.add_edge(node, attr_node, label=key)
                flat_G.add_edge(attr_node, value_node, label='value')

        # Create nodes and edges for each attribute outside 'Props'
        for key, value in data.items():
            if key != 'Props':
                attr_node = f"{node}_{key}"
                value_node = f"{node}_{key}_value"
                value_label = str(value)  # Convert value to string
                flat_G.add_node(attr_node, label=key)
                flat_G.add_node(value_node, label=value_label)
                flat_G.add_edge(node, attr_node, label=key)
                flat_G.add_edge(attr_node, value_node, label='value')

    # Create edges for the original edges with attributes as nodes
    for u, v, data in G.edges(data=True):
        edge_node = f"{u}_{v}_edge"
        flat_G.add_node(edge_node, label=f"Edge {u} -> {v}")
        flat_G.add_edge(u, edge_node, label='source')
        flat_G.add_edge(edge_node, v, label='target')

        # Create nodes and edges for each edge attribute if 'Props' is not empty
        if data and data.get('Props'):  # Skip if 'Props' is empty
            for key, value in data.items():
                attr_node = f"{edge_node}_{key}"
                value_node = f"{edge_node}_{key}_value"
                value_label = str(value)  # Convert value to string
                flat_G.add_node(attr_node, label=key)
                flat_G.add_node(value_node, label=value_label)
                flat_G.add_edge(edge_node, attr_node, label=key)
                flat_G.add_edge(attr_node, value_node, label='value')

    return flat_G

def convert_to_topological(multidigraph):
    # Step 2: Initialize the new graph as a MultiDiGraph
    H = nx.MultiDiGraph()

    # Step 3 & 4: Add nodes for original nodes and attributes, and add directed edges
    for node, attrs in multidigraph.nodes(data=True):
        H.add_node(node)  # Add original node
        for attr, value in attrs.items():
            attr_node = f"{node}_{attr}_{value}"  # Create a unique identifier for the attribute node
            H.add_node(attr_node)
            H.add_edge(
                node, attr_node
            )  # Link node to its attribute node with a directed edge
            H.add_edge(
                attr_node, node
            )  # Optionally, add reverse edge if bidirectional relationship is needed

    # Step 5: Copy original directed edges, considering multiple edges
    for source, target, key in multidigraph.edges(keys=True):
        H.add_edge(source, target, key=key)  # Preserve the edge key for multi-edges

    return H


def add_edge_label(graph, label):
    for u, v, data in graph.edges(data=True):
        data['ulabels'] = label

def print_edge_labels(graph):
    for u, v, data in graph.edges(data=True):
        if 'ulabels' in data:
            print(f"Edge ({u}, {v}) has attribute 'ulabels': {data['ulabels']}")
        else:
            print(f"Edge ({u}, {v}) has no attribute 'ulabels'")



def eliminate_in_graph(G, keys):
    """
    Eliminate attributes or props based on the list of keys from edges and nodes in the graph.

    Parameters:
    G (nx.Graph): The input graph.
    keys (list): The list of keys to eliminate from nodes and edges.
    """
    # Remove the keys from node attributes
    for node, data in G.nodes(data=True):
        for key in keys:
            if key in data:
                del data[key]
            if 'Props' in data and key in data['Props']:
                del data['Props'][key]

    # Remove the keys from edge attributes
    for u, v, data in G.edges(data=True):
        for key in keys:
            if key in data:
                del data[key]
            if 'Props' in data and key in data['Props']:
                del data['Props'][key]



def introspect_nx_graph_attributes(graph):
    # Initialize sets to hold unique node and edge attribute names
    node_attr_names = set()
    edge_attr_names = set()

    # Introspect node attributes
    for _, node_data in graph.nodes(data=True):
        node_attr_names.update(node_data.keys())

    # Introspect edge attributes
    # Assuming a MultiDiGraph, we consider the possibility of multiple edges between any two nodes
    for _, _, edge_data in graph.edges(data=True):
        edge_attr_names.update(edge_data.keys())

    # Convert sets to sorted lists for consistency
    node_attr_names = sorted(list(node_attr_names))
    edge_attr_names = sorted(list(edge_attr_names))

    return node_attr_names, edge_attr_names




def get_attributes_and_categories(graph, target="node"):
    """
    Collects attributes and their unique categories from nodes or edges of a networkx graph.

    :param graph: The networkx graph to analyze.
    :param target: Specify 'node' for node attributes or 'edge' for edge attributes.
    :return: A dictionary where keys are attribute names and values are sets of unique values.
    """
    attributes_categories = {}

    # Define a helper function to process attributes
    def process_attributes(attrs):
        for attr_name, attr_value in attrs.items():
            if attr_name not in attributes_categories:
                attributes_categories[attr_name] = set()
            if attr_value is not None:  # Consider excluding None values
                attributes_categories[attr_name].add(attr_value)

    if target == "node":
        for _, attrs in graph.nodes(data=True):
            process_attributes(attrs)
    elif target == "edge":
        for _, _, attrs in graph.edges(data=True):
            process_attributes(attrs)
    else:
        raise ValueError("The target must be either 'node' or 'edge'.")

    return attributes_categories





def visualize_graph(
    graph,
    title="Network Graph Visualization",
    node_size_range=(5, 15),
    width=800,
    height=600,
):
    """
    Visualizes a NetworkX graph using Plotly, including node and edge attributes.

    Parameters:
    - graph: NetworkX graph object to be visualized.
    - title: Title of the graph visualization.
    - node_size_range: Tuple indicating the minimum and maximum sizes of nodes.
    - width: Width of the visualization.
    - height: Height of the visualization.
    """
    pos = nx.spring_layout(graph, seed=42)  # Position nodes

    # Edge traces
    edge_x, edge_y, edge_text = [], [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        # Edge attributes text
        edge_attr = ", ".join(
            [f"{key}: {value}" for key, value in graph[edge[0]][edge[1]].items()]
        )
        edge_text += [f"{edge[0]} <-> {edge[1]}\n{edge_attr}", "", ""]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="text",
        mode="lines",
        text=edge_text,  # Add edge attributes to hover text
    )

    # Node traces
    node_x, node_y = [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Normalize node sizes
    node_degrees = np.array([graph.degree(n) for n in graph.nodes()])
    min_size, max_size = node_size_range
    sizes = min_size + (max_size - min_size) * (node_degrees - np.min(node_degrees)) / (
        np.max(node_degrees) - np.min(node_degrees)
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            size=sizes,
            color=node_degrees,
            colorbar=dict(
                thickness=15, title="Node Degree", xanchor="left", titleside="right"
            ),
            line=dict(width=2, color="DarkSlateGrey"),
        ),
    )

    # Adding hover text to nodes including attributes
    node_text = []
    for node in graph.nodes():
        attr_text = ", ".join(
            [f"{key}: {value}" for key, value in graph.nodes[node].items()]
        )
        node_text.append(f"{node}\n{attr_text}\n{graph.degree(node)} connections")
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height,
        ),
    )

    fig.show()


def visualize_graph_with_edge_attributes(
    graph,
    title="Network Graph Visualization with Edge Attributes",
    node_size_range=(5, 15),
    width=800,
    height=600,
):
    pos = nx.spring_layout(graph, seed=42)  # Position nodes using a spring layout

    # Initialize containers for edge lines and hover points
    edge_lines = []
    hover_points_x, hover_points_y, hover_points_text = [], [], []

    # Define edge types and corresponding line styles
    edge_styles = {
        "direct_transition": "solid",
        "time_to_transition": "dash",
        "time_interval": "dot",
    }

    # Iterate over edges to create line traces and collect hover points
    for edge_type, line_style in edge_styles.items():
        x_edge, y_edge = [], []
        for u, v, data in graph.edges(data=True):
            if data.get("type") == edge_type:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                x_edge.extend([x0, x1, None])
                y_edge.extend([y0, y1, None])

                # Collect midpoints for hover info
                hover_points_x.append((x0 + x1) / 2)
                hover_points_y.append((y0 + y1) / 2)
                hover_info = f"{u}-{v}: {data.get('type')}"
                for key, value in data.items():
                    if key != "type":  # Add additional data to hover text
                        hover_info += f", {key}: {value}"
                hover_points_text.append(hover_info)

        # Add edge line trace
        edge_trace = go.Scatter(
            x=x_edge,
            y=y_edge,
            line=dict(width=1.5, color="#888", dash=line_style),
            hoverinfo="none",
            mode="lines",
        )
        edge_lines.append(edge_trace)

    # Add hover points trace
    hover_points_trace = go.Scatter(
        x=hover_points_x,
        y=hover_points_y,
        mode="markers",
        marker=dict(size=8, color="#888", opacity=0),  # Make markers invisible
        hoverinfo="text",
        text=hover_points_text,
    )

    # Node trace setup remains the same
    node_x, node_y, node_text = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}: {graph.degree(node)} connections")

    node_degrees = np.array([graph.degree(n) for n in graph.nodes()])
    min_size, max_size = node_size_range
    sizes = min_size + (max_size - min_size) * (node_degrees - np.min(node_degrees)) / (
        np.max(node_degrees) - np.min(node_degrees)
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=sizes,
            color=node_degrees,
            colorbar=dict(thickness=15, title="Node Connections"),
            line_width=2,
        ),
    )

    # Compile the figure with node, edge lines, and hover points traces
    fig = go.Figure(
        data=[*edge_lines, hover_points_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height,
        ),
    )

    fig.show()



def visualize_network_with_sequential_subgraph(
    master_graph, subgraph, title="Network and Sequence Visualization"
):
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Master Network with Subgraph Overlay",
            "Sequential Visualization",
        ),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # Generate layout for the master graph and subgraph
    master_pos = nx.spring_layout(master_graph, seed=42)

    # Function to add hover info for nodes and edges
    def add_hover_info(trace, node_or_edge, attr_type="node"):
        if attr_type == "node":
            trace.hoverinfo = "text"
            trace.text = f"{node_or_edge}"
        else:  # Edge with attributes
            hover_texts = [f"{k}: {v}" for k, v in node_or_edge.items()]
            trace.hoverinfo = "text"
            trace.text = [", ".join(hover_texts)]

    # Master graph and subgraph visualization
    for edge in master_graph.edges(data=True):
        x0, y0 = master_pos[edge[0]]
        x1, y1 = master_pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color="grey"),
            showlegend=False,
        )
        add_hover_info(edge_trace, edge[2], "edge")
        fig.add_trace(edge_trace, row=1, col=1)

    for node in master_graph.nodes():
        x, y = master_pos[node]
        node_trace = go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=8, color="blue"),
            showlegend=False,
        )
        add_hover_info(node_trace, node)
        fig.add_trace(node_trace, row=1, col=1)

    # Subgraph overlay with dashed-line nodes for attributes and direct transitions
    for edge in subgraph.edges(data=True):
        x0, y0 = master_pos[edge[0]]
        x1, y1 = master_pos[edge[1]]
        # Direct transition
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Attribute subnodes visualization
        attr_x, attr_y = (x0 + x1) / 2, (y0 + y1) / 2  # Midpoint for attribute nodes
        for attr, value in edge[2].items():
            subnode_trace = go.Scatter(
                x=[attr_x],
                y=[attr_y],
                mode="markers",
                marker=dict(size=4, color="purple", symbol="x"),
                showlegend=False,
            )
            add_hover_info(subnode_trace, {attr: value}, "edge")
            fig.add_trace(subnode_trace, row=1, col=1)
            # Connecting attribute nodes with dashed lines
            fig.add_trace(
                go.Scatter(
                    x=[x0, attr_x, x1],
                    y=[y0, attr_y, y1],
                    mode="lines",
                    line=dict(color="purple", width=1, dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # Sequence Visualization with attributes
    sequence_nodes = list(subgraph.nodes())
    sequence_pos = {node: (index, 0) for index, node in enumerate(sequence_nodes)}

    for edge in subgraph.edges(data=True):
        x0, y0 = sequence_pos[edge[0]]
        x1, y1 = sequence_pos[edge[1]]
        # Sequence direct transition
        direct_trace = go.Scatter(
            x=[x0, x1],
            y=[0, 0],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False,
        )
        add_hover_info(direct_trace, edge[2], "edge")
        fig.add_trace(direct_trace, row=2, col=1)

        # Attribute visualization in sequence
        attr_x = (
            x0 + x1
        ) / 2  # Attribute nodes placed in the middle of the direct transition
        for attr, value in edge[2].items():
            seq_subnode_trace = go.Scatter(
                x=[attr_x],
                y=[-0.1],
                mode="markers",
                marker=dict(size=4, color="purple", symbol="x"),
                showlegend=False,
            )
            add_hover_info(seq_subnode_trace, {attr: value}, "edge")
            fig.add_trace(seq_subnode_trace, row=2, col=1)
            # Dashed line for attribute in sequence
            fig.add_trace(
                go.Scatter(
                    x=[attr_x, x1],
                    y=[-0.1, 0],
                    mode="lines",
                    line=dict(color="purple", width=1, dash="dash"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    # Drawing nodes for the sequence
    for node in sequence_nodes:
        x, y = sequence_pos[node]
        node_trace = go.Scatter(
            x=[x],
            y=[0],
            mode="markers+text",
            text=[node],
            textposition="top center",
            marker=dict(size=10, color="blue"),
            showlegend=False,
        )
        add_hover_info(node_trace, node)
        fig.add_trace(node_trace, row=2, col=1)

    # Update Layout for better visibility
    fig.update_layout(height=1000, width=1200, title_text=title, showlegend=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(showticklabels=True, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)

    fig.show()


def compare_graphs(G1, G2):
    try:
        # Number of Nodes
        num_nodes_G1 = G1.number_of_nodes()
        num_nodes_G2 = G2.number_of_nodes()
        print("Number of Nodes G1:", num_nodes_G1)
        print("Number of Nodes G2:", num_nodes_G2)
    except Exception as e:
        print(f"Error comparing number of nodes: {e}")

    try:
        # Number of Edges
        num_edges_G1 = G1.number_of_edges()
        num_edges_G2 = G2.number_of_edges()
        print("Number of Edges G1:", num_edges_G1)
        print("Number of Edges G2:", num_edges_G2)
    except Exception as e:
        print(f"Error comparing number of edges: {e}")

    try:
        # Nodes Present in One Graph but Not the Other
        nodes_diff_G1 = sorted(set(map(str, G1.nodes)) - set(map(str, G2.nodes)))
        nodes_diff_G2 = sorted(set(map(str, G2.nodes)) - set(map(str, G1.nodes)))
        print("Nodes in G1 but not in G2:", nodes_diff_G1)
        print("Nodes in G2 but not in G1:", nodes_diff_G2)
    except Exception as e:
        print(f"Error comparing nodes: {e}")
    try:
        # Convert edges to a canonical form where each edge is represented as a sorted tuple of strings
        edges_G1 = set((str(edge[0]), str(edge[1])) if edge[0] < edge[1] else (str(edge[1]), str(edge[0])) for edge in G1.edges)
        edges_G2 = set((str(edge[0]), str(edge[1])) if edge[0] < edge[1] else (str(edge[1]), str(edge[0])) for edge in G2.edges)

        # Compute the differences
        edges_diff_G1 = sorted(edges_G1 - edges_G2)
        edges_diff_G2 = sorted(edges_G2 - edges_G1)

        # Determine the length of the longest list for proper alignment
        max_len = max(len(edges_diff_G1), len(edges_diff_G2))

        # Pad the shorter list with empty strings for alignment
        edges_diff_G1 += [('', '')] * (max_len - len(edges_diff_G1))
        edges_diff_G2 += [('', '')] * (max_len - len(edges_diff_G2))

        # Print the header
        print(f"{'Edges in G1 but not in G2':<25} {'Edges in G2 but not in G1':<25}")

        # Print the edges side by side
        for edge1, edge2 in zip(edges_diff_G1, edges_diff_G2):
            print(f"{str(edge1):<25} {str(edge2):<25}")

    except Exception as e:
        print(f"Error comparing edges: {e}")

    try:
        # Degree Sequence
        degree_sequence_G1 = sorted([d for n, d in G1.degree()], reverse=True)
        degree_sequence_G2 = sorted([d for n, d in G2.degree()], reverse=True)
        print("Degree sequence G1:", degree_sequence_G1)
        print("Degree sequence G2:", degree_sequence_G2)
    except Exception as e:
        print(f"Error comparing degree sequences: {e}")

    try:
        # Clustering Coefficient
        clustering_G1 = nx.average_clustering(G1)
        clustering_G2 = nx.average_clustering(G2)
        print("Clustering Coefficient G1:", clustering_G1)
        print("Clustering Coefficient G2:", clustering_G2)
    except Exception as e:
        print(f"Error comparing clustering coefficients: {e}")

    try:
        # Connected Components
        components_G1 = sorted([len(c) for c in nx.connected_components(G1)])
        components_G2 = sorted([len(c) for c in nx.connected_components(G2)])
        print("Connected Components G1:", components_G1)
        print("Connected Components G2:", components_G2)
    except Exception as e:
        print(f"Error comparing connected components: {e}")

    try:
        # Graph Density
        density_G1 = nx.density(G1)
        density_G2 = nx.density(G2)
        print("Graph Density G1:", density_G1)
        print("Graph Density G2:", density_G2)
    except Exception as e:
        print(f"Error comparing graph densities: {e}")

    try:
        # Average Shortest Path Length
        avg_shortest_path_length_G1 = nx.average_shortest_path_length(G1) if nx.is_connected(G1) else float('inf')
        avg_shortest_path_length_G2 = nx.average_shortest_path_length(G2) if nx.is_connected(G2) else float('inf')
        print("Average Shortest Path Length G1:", avg_shortest_path_length_G1)
        print("Average Shortest Path Length G2:", avg_shortest_path_length_G2)
    except Exception as e:
        print(f"Error comparing average shortest path lengths: {e}")

    try:
        # Diameter
        diameter_G1 = nx.diameter(G1) if nx.is_connected(G1) else float('inf')
        diameter_G2 = nx.diameter(G2) if nx.is_connected(G2) else float('inf')
        print("Diameter G1:", diameter_G1)
        print("Diameter G2:", diameter_G2)
    except Exception as e:
        print(f"Error comparing diameters: {e}")



# ------------------------------------------------------------------------- #
#                       GRAPH generic feature creation                      #
# ------------------------------------------------------------------------- #


def make_hashable(value):
    if isinstance(value, (list, set)):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value

def most_common_attributes(graph):
    from collections import Counter

    node_attributes = Counter()
    edge_attributes = Counter()

    for node, data in graph.nodes(data=True):
        for attr, value in data.items():
            hashable_value = make_hashable(value)
            node_attributes[(attr, hashable_value)] += 1

    for u, v, data in graph.edges(data=True):
        for attr, value in data.items():
            hashable_value = make_hashable(value)
            edge_attributes[(attr, hashable_value)] += 1

    most_common_node_attrs = {}
    most_common_edge_attrs = {}

    for (attr, value), count in node_attributes.items():
        if attr not in most_common_node_attrs or count > most_common_node_attrs[attr][1]:
            most_common_node_attrs[attr] = (value, count)

    for (attr, value), count in edge_attributes.items():
        if attr not in most_common_edge_attrs or count > most_common_edge_attrs[attr][1]:
            most_common_edge_attrs[attr] = (value, count)

    most_common_node_attrs = {k: v[0] for k, v in most_common_node_attrs.items()}
    most_common_edge_attrs = {k: v[0] for k, v in most_common_edge_attrs.items()}

    return most_common_node_attrs, most_common_edge_attrs



def add_reachability_info(graph):
    root_nodes = [node for node in graph.nodes() if str(node).endswith('_0')]

    reachability = {node: 0 for node in graph.nodes()}

    for root in root_nodes:
        descendants = nx.descendants(graph, root)
        reachability[root] = len(descendants)
        for descendant in descendants:
            reachability[descendant] = len(nx.descendants(graph, descendant))

    nx.set_node_attributes(graph, reachability, 'reachability')
    return graph


def extract_node_features(graph, default_attributes):
    nodes = list(graph.nodes)
    node_features = []
    for node in nodes:
        data = graph.nodes[node]
        feature = {f'node_{attr}': data.get(attr, default_attributes.get(attr, None)) for attr in default_attributes}
        feature.update({
            'node_id': node,
            'node_degree': graph.degree(node),
            'node_in_degree': graph.in_degree(node),
            'node_out_degree': graph.out_degree(node)
        })
        node_features.append(feature)

    feature_keys = list(node_features[0].keys())
    for feature in node_features:
        for key in feature_keys:
            if key not in feature:
                feature[key] = default_attributes.get(key[5:], None)

    return pd.DataFrame(node_features)


def extract_edge_features(graph, nodes, default_edge_attributes):
    edges = []
    for u in nodes:
        for v in nodes:
            if graph.has_edge(u, v):
                data = graph[u][v]
                feature = {f'edge_{attr}': data.get(attr, default_edge_attributes.get(attr, None)) for attr in default_edge_attributes}
            else:
                feature = {f'edge_{attr}': default_edge_attributes.get(attr, None) for attr in default_edge_attributes}
            feature.update({
                'edge_source': u,
                'edge_target': v,
                'edge_has_edge': graph.has_edge(u, v)
            })
            edges.append(feature)

    return pd.DataFrame(edges)



def extract_graph_features(graph):
    features = {
        'graph_number_of_nodes': graph.number_of_nodes(),
        'graph_number_of_edges': graph.number_of_edges(),
        'graph_density': nx.density(graph),
        'graph_number_of_connected_components': nx.number_connected_components(graph.to_undirected()) if not graph.is_multigraph() else np.nan,
        'graph_average_degree_centrality': np.mean(list(nx.degree_centrality(graph).values()))
    }

    if not graph.is_multigraph():
        features['graph_average_clustering'] = nx.average_clustering(graph.to_undirected())
        features['graph_eccentricity'] = np.mean(list(nx.eccentricity(graph).values())) if nx.is_connected(graph.to_undirected()) else np.nan

    if nx.is_directed(graph):
        try:
            largest_scc = max(nx.strongly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_scc)
            features['graph_average_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
            features['graph_diameter'] = nx.diameter(subgraph.to_undirected())
        except Exception as e:
            features['graph_average_shortest_path_length'] = np.nan
            features['graph_diameter'] = np.nan
    else:
        if nx.is_connected(graph):
            features['graph_average_shortest_path_length'] = nx.average_shortest_path_length(graph)
            features['graph_diameter'] = nx.diameter(graph)
        else:
            features['graph_average_shortest_path_length'] = np.nan
            features['graph_diameter'] = np.nan

    return pd.DataFrame([features])


def aggregate_features(df, prefix):
    from scipy.stats import skew, kurtosis
    aggregation_functions = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
        'max': np.max,
        'min': np.min,
        'range': lambda x: np.max(x) - np.min(x),
        'variance': np.var,
        'skewness': skew,
        'kurtosis': kurtosis
    }
    aggregated_features = {}
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # Only aggregate numeric columns
            for agg_name, agg_func in aggregation_functions.items():
                agg_value = agg_func(df[column].dropna())
                aggregated_features[f'{prefix}_{column}_{agg_name}'] = agg_value
        else:
            if df[column].dtype == object:
                # Convert lists and dictionaries to tuples for handling non-hashable types
                converted_column = df[column].apply(lambda x: tuple(sorted(x.items())) if isinstance(x, dict) else tuple(x) if isinstance(x, list) else x)
                mode_value = converted_column.mode()[0] if not converted_column.mode().empty else None
                unique_count = converted_column.nunique()
                aggregated_features[f'{prefix}_{column}_mode'] = mode_value
                aggregated_features[f'{prefix}_{column}_unique_count'] = unique_count
    return aggregated_features



def vectorize_process_graph_features(df, column_name):
    feature_dfs = df[column_name].apply(process_graph_features)
    result_df = pd.concat(feature_dfs.tolist(), axis=0).reset_index(drop=True)
    return result_df


def print_unique_types(G):
    """
    Print unique types of node and edge attributes in the graph.

    Parameters:
    G (nx.Graph): The input graph.
    """
    node_attr_types = set()
    edge_attr_types = set()

    # Collect unique node attribute types
    for _, data in G.nodes(data=True):
        for key, value in data.items():
            node_attr_types.add((key, type(value)))

    # Collect unique edge attribute types
    for _, _, data in G.edges(data=True):
        for key, value in data.items():
            edge_attr_types.add((key, type(value)))

    print("Unique Node Attribute Types:", node_attr_types)
    print("Unique Edge Attribute Types:", edge_attr_types)

    
def process_graph_features(G):
    most_common_node_attrs, most_common_edge_attrs = most_common_attributes(G)

    node_features = extract_node_features(G, most_common_node_attrs)
    edge_features = extract_edge_features(G, list(G.nodes), most_common_edge_attrs)
    graph_features = extract_graph_features(G)

    features = pd.concat([node_features, edge_features, graph_features], axis=1)
    feature_columns = set(node_features.columns).union(set(edge_features.columns)).union(set(graph_features.columns))
    features = features.reindex(columns=feature_columns, fill_value=np.nan)

    node_aggregated_features = aggregate_features(node_features, 'node')
    edge_aggregated_features = aggregate_features(edge_features, 'edge')

    graph_features_flat = graph_features.iloc[0].to_dict()
    graph_features_flat['node_count'] = G.number_of_nodes()
    graph_features_flat['edge_count'] = G.number_of_edges()

    all_features = {**node_aggregated_features, **edge_aggregated_features, **graph_features_flat}

    flat_features_df = pd.DataFrame([all_features])
    return flat_features_df

#                                                                           #
# ------------------------------------------------------------------------- #
