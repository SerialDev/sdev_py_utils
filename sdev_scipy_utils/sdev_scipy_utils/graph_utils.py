import networkx as nx

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
