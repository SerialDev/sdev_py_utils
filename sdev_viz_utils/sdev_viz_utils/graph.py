class SlicePlot:
    """
    * ---------------Function---------------
    *
    SlicePlot: A class to create a temporal slice plot of a graph
    *
    *
    * ----------------Returns---------------
    *
    -> result :: None
    *
    * ----------------Params----------------
    *
    graph :: networkx.graph : The input graph
    title :: str : The title of the plot (default: "slice")
    use_monotonic_time :: bool : Whether to use monotonically increasing time (default: False)
    height :: int : The height of the plot (default: 800)
    *
    * ----------------Usage----------------
    *
    Create a SlicePlot object with a graph, title, and other parameters. Call the create_edges method to create the edges of the graph. Finally, call the draw_plotly method to draw the plot.
    *
    * ----------------Notes----------------
    * This class uses Plotly to create an interactive plot of a graph with temporal information. It can be used to visualize the temporal relationships between nodes in a graph.
    * >>>
    # Example usage:
    G = nx.DiGraph()
    G.add_edge(1, 2, label="A", Time=1, duration=2)
    G.add_edge(2, 3, label="B", Time=3, duration=2)
    G.add_edge(3, 4, label="C", Time=5, duration=2)
    G.add_edge(4, 5, label="D", Time=7, duration=2)
    G.add_edge(5, 6, label="E", Time=9, duration=2)
    G.add_edge(6, 7, label="F", Time=11, duration=2)
    G.add_edge(7, 8, label="G", Time=13, duration=2)
    G.add_edge(8, 9, label="H", Time=15, duration=2)
    G.add_edge(9, 10, label="I", Time=17, duration=2)
    G.add_edge(10, 11, label="J", Time=19, duration=2)
    G.add_edge(11, 12, label="K", Time=21, duration=2)
    G.add_edge(12, 13, label="L", Time=23, duration=2)
    G.add_edge(13, 14, label="M", Time=25, duration=2)
    G.add_edge(14, 1, label="N", Time=27, duration=2)  # Loopback

    plot = SlicePlot(G, title="Master Network - Temporal Slice Plot", use_monotonic_time=True)
    plot.create_edges()
    plot.draw_plotly()

    plot = SlicePlot(G, title="Master Network - Temporal Slice Plot", use_monotonic_time=False)
    plot.create_edges()
    plot.draw_plotly()

    """

    import networkx as nx
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    class_name = "slice"

    def __init__(self, graph, title=None, use_monotonic_time=False, height=800):
        self.graph = graph
        self.title = title if title else self.class_name
        self.use_monotonic_time = use_monotonic_time
        self.labels = []
        self.edges = []
        self.connections = []
        self.height = height

    def create_edges(self):
        step = 1
        for edge in self.graph.edges(data=True):
            if "label" in edge[2]:
                label = edge[2]["label"]
            else:
                label = f"{edge[0]}-{edge[1]}"
            if label not in self.labels:
                self.labels.append(label)

        times = [0]
        for i, (u, v, data) in enumerate(self.graph.edges(data=True)):
            if self.use_monotonic_time:
                time = i  # Use monotonically increasing time
            else:
                time = data.get("Time", i)  # Use the 'Time' attribute if available
            label = data.get("label", f"{u}-{v}")
            duration = data.get("duration", 1)  # Use 'duration' if available
            j = self.labels.index(label)
            plot_edge = {
                "u": u,
                "v": v,
                "label": label,
                "x": time,
                "y": j * step,
                "start": time,
                "end": time + duration,
            }
            self.edges.append(plot_edge)
            self.connections.append((u, v, label))
            times.append(time)

        times = sorted(set(times))
        time_to_index = {time: index for index, time in enumerate(times)}
        for edge in self.edges:
            edge["time_index"] = time_to_index[edge["x"]]

    def draw_arc(self, x_start, y_start, x_end, y_end, u, v):
        control_x = (x_start + x_end) / 2
        control_y = max(y_start, y_end) + 1
        path = f"M {x_start},{y_start} Q {control_x},{control_y} {x_end},{y_end}"
        hover_text = f"Loopback: {u} -> {v}"

        return dict(
            type="path", path=path, line=dict(color="black", dash="dash")
        ), go.Scatter(
            x=[x_start, control_x, x_end],
            y=[y_start, control_y, y_end],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="text",
            text=[hover_text] * 3,
        )

    def draw_plotly(self):
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=("Network", "Edges Stack"),
            specs=[[{"type": "scatter"}, {"type": "table"}]],
        )

        edge_x = [edge["x"] for edge in self.edges]
        edge_y = [edge["y"] for edge in self.edges]
        edge_text = [f"{edge['u']}-{edge['v']}" for edge in self.edges]

        # Check if the number of edges exceeds 20 to decide whether to show labels
        show_labels = len(self.edges) <= 20

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="markers+text" if show_labels else "markers",
            marker=dict(color=edge_y, colorscale="Viridis", size=10),
            text=edge_text if show_labels else None,
            textposition="top center",
            name="Edges",
            hoverinfo="text",
            hovertext=[
                f"Edge: {edge['u']} -> {edge['v']}<br>Label: {edge['label']}<br>Time: {edge['x']}"
                for edge in self.edges
            ],
        )

        fig.add_trace(edge_trace, row=1, col=1)

        table_header = dict(
            values=[
                "Inserted Node",
                "Linked Node",
                "Loopback",
                "Node -- Node connections",
            ],
            align="left",
        )
        table_cells = dict(values=[[], [], [], []], align="left")

        fig.add_trace(go.Table(header=table_header, cells=table_cells), row=1, col=2)

        # Create frames for the slider to show the construction of the graph step by step
        frames = []
        shapes = []
        hover_traces = []
        inserted_nodes = set()
        max_x = max(edge["end"] for edge in self.edges)  # Find the maximum end time

        for t in range(max(edge["time_index"] for edge in self.edges) + 1):
            frame_data = [
                go.Scatter(
                    x=[edge["x"] for edge in self.edges if edge["time_index"] <= t],
                    y=[edge["y"] for edge in self.edges if edge["time_index"] <= t],
                    mode="markers+text" if show_labels else "markers",
                    marker=dict(
                        color=[
                            edge["y"] for edge in self.edges if edge["time_index"] <= t
                        ],
                        colorscale="Viridis",
                        size=10,
                    ),
                    text=(
                        [
                            f"{edge['u']}-{edge['v']}"
                            for edge in self.edges
                            if edge["time_index"] <= t
                        ]
                        if show_labels
                        else None
                    ),
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=[
                        f"Edge: {edge['u']} -> {edge['v']}<br>Label: {edge['label']}<br>Time: {edge['x']}"
                        for edge in self.edges
                        if edge["time_index"] <= t
                    ],
                )
            ]

            current_shapes = shapes.copy()
            current_hover_traces = hover_traces.copy()
            current_connections = [[], [], [], []]
            for i, (u, v, label) in enumerate(self.connections):
                edge = next(
                    edge for edge in self.edges if edge["u"] == u and edge["v"] == v
                )
                if edge["time_index"] <= t:
                    try:
                        x_start = edge["x"]
                        y_start = edge["y"]
                        x_end = next(
                            edge["x"]
                            for edge in self.edges
                            if edge["u"] == v and edge["time_index"] <= t
                        )
                        y_end = next(
                            edge["y"]
                            for edge in self.edges
                            if edge["u"] == v and edge["time_index"] <= t
                        )
                    except StopIteration:
                        continue  # Skip if the edge is not found or not yet introduced
                    if u not in inserted_nodes:
                        current_connections[0].insert(0, f"Node {u}")
                        current_connections[1].insert(0, f"{u} -> {v}")
                        current_connections[3].insert(0, f"{u} -> {v}")
                        inserted_nodes.add(u)
                    if v not in inserted_nodes:
                        current_connections[0].insert(0, f"Node {v}")
                        current_connections[1].insert(0, f"{v} -> {u}")
                        current_connections[3].insert(0, f"{v} -> {u}")
                        inserted_nodes.add(v)
                    if (
                        edge["time_index"] == t
                    ):  # Add the shape only in the frame it is introduced
                        if x_end < x_start:
                            arc_shape, arc_hover_trace = self.draw_arc(
                                x_start, y_start, x_end, y_end, u, v
                            )
                            current_shapes.append(arc_shape)
                            current_hover_traces.append(arc_hover_trace)
                            current_connections[2].insert(0, f"Loopback: {u} -> {v}")
                            current_connections[1].insert(0, "")
                            current_connections[3].insert(0, f"Loopback: {u} -> {v}")
                        else:
                            current_shapes.append(
                                dict(
                                    type="line",
                                    x0=x_start,
                                    y0=y_start,
                                    x1=x_end,
                                    y1=y_end,
                                    line=dict(color="black", dash="dash"),
                                )
                            )
                            # Add a visible scatter trace for hover information
                            current_hover_traces.append(
                                go.Scatter(
                                    x=[x_start, x_end],
                                    y=[y_start, y_end],
                                    mode="lines",
                                    line=dict(color="rgba(0,0,0,0.2)", dash="dash"),
                                    hoverinfo="text",
                                    text=f"{u} -> {v}",
                                )
                            )
                            current_connections[1].insert(0, f"{u} -> {v}")
                            current_connections[2].insert(0, "")
                            current_connections[3].insert(0, f"{u} -> {v}")
            shapes = current_shapes.copy()
            hover_traces = current_hover_traces.copy()

            # Update the table cells
            for col in range(4):
                table_cells["values"][col] = (
                    current_connections[col] + table_cells["values"][col]
                )[:20]

            frame_data.append(
                go.Table(
                    header=table_header,
                    cells=dict(values=table_cells["values"], align="left"),
                )
            )

            frames.append(
                go.Frame(
                    data=frame_data + hover_traces,
                    name=str(t),
                    layout=dict(shapes=current_shapes),
                )
            )

        fig.update_layout(
            title=self.title,
            xaxis_title="Step",
            yaxis_title="Edge",
            height=self.height,  # Make the plot area taller
            xaxis=dict(
                range=[0, max_x + 5]
            ),  # Set a larger initial range for the x-axis
            yaxis=dict(tickvals=list(range(len(self.labels))), ticktext=self.labels),
            xaxis2=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False),
            sliders=[
                {
                    "steps": [
                        {
                            "label": str(t),
                            "method": "animate",
                            "args": [
                                [str(t)],
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                        }
                        for t in range(
                            max(edge["time_index"] for edge in self.edges) + 1
                        )
                    ],
                    "transition": {"duration": 0},
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": -0.2,
                    "yanchor": "top",
                }
            ],
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": -0.2,
                    "yanchor": "top",
                }
            ],
        )

        fig.frames = frames

        fig.show()


# Absolutely the best for directed graphs
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from collections import defaultdict
from functools import lru_cache
from rtree import index as rtree_index  # Make sure to import rtree_index


def add_edge_to_rtree(spatial_index, edge, positions, edge_id):
    """Adds an edge to the R-tree with a unique ID and a flattened bounding box."""
    source, target = edge
    x0, y0 = positions[source]
    x1, y1 = positions[target]
    # Flatten the coordinates tuple to (minx, miny, maxx, maxy)
    minx, miny = min(x0, x1), min(y0, y1)
    maxx, maxy = max(x0, x1), max(y0, y1)
    # Insert the edge ID (integer) into the R-tree
    spatial_index.insert(edge_id, (minx, miny, maxx, maxy), obj=edge_id)


def query_proximity(edge, spatial_index, positions, threshold=0.3):
    """Queries for nearby edges using the spatial index."""
    source, target = edge
    x0, y0 = positions[source]
    x1, y1 = positions[target]
    # Define the bounding box around the edge with some proximity threshold
    minx, miny = min(x0, x1) - threshold, min(y0, y1) - threshold
    maxx, maxy = max(x0, x1) + threshold, max(y0, y1) + threshold
    # Query the spatial index for all nearby edges within the bounding box
    return list(spatial_index.intersection((minx, miny, maxx, maxy), objects=True))


def create_bundled_edges(
    G, positions, bundle_strength=0.95, num_curve_points=100, max_iters=10
):
    """Edge bundling using a procedure similar to the MINGLE algorithm."""
    # Initialize variables for edge coordinates and text
    edge_x, edge_y, edge_text = [], [], []
    UNGROUPED = -1
    total_gain = 0

    # Create spatial index for edge proximity
    spatial_index = rtree_index.Index()
    edges = list(G.edges(data=True))

    # Assign a unique ID for each edge
    edge_ids = list(range(len(edges)))
    edge_map = {edge_id: edge[:2] for edge_id, edge in enumerate(edges)}

    # Build the initial proximity graph Gamma
    edge_proximity = defaultdict(set)
    for edge_id, edge in edge_map.items():
        add_edge_to_rtree(spatial_index, edge, positions, edge_id)

    # Build initial edge proximity graph
    for edge_id, edge in edge_map.items():
        neighbors = query_proximity(edge, spatial_index, positions)
        for neighbor in neighbors:
            neighbor_id = neighbor.object
            if neighbor_id != edge_id:
                edge_proximity[edge_id].add(neighbor_id)

    # Initialize grouping
    group = {edge_id: UNGROUPED for edge_id in edge_ids}

    @lru_cache(maxsize=None)
    def calculate_ink(edge_ids):
        """Calculates the total ink for a set of edges."""
        total_length = 0
        for eid in edge_ids:
            source, target = edge_map[eid]
            start = np.array(positions[source])
            end = np.array(positions[target])
            total_length += np.linalg.norm(end - start)
        return total_length

    def bundle_edges(edge_ids):
        """Creates a bundled Bezier curve for a group of edges."""
        # Calculate the average positions of sources and targets
        sources = [np.array(positions[edge_map[eid][0]]) for eid in edge_ids]
        targets = [np.array(positions[edge_map[eid][1]]) for eid in edge_ids]
        source_mean = np.mean(sources, axis=0)
        target_mean = np.mean(targets, axis=0)

        # Control point for the Bezier curve
        control = source_mean + (target_mean - source_mean) * bundle_strength

        t = np.linspace(0, 1, num_curve_points)
        curve_points = quadratic_bezier_curve(source_mean, control, target_mean, t)
        edge_x.extend(curve_points[:, 0].tolist() + [None])
        edge_y.extend(curve_points[:, 1].tolist() + [None])

        # For hover text, we can list the number of edges in the bundle
        edge_text.append(f"Bundled Edges: {len(edge_ids)}")

    def quadratic_bezier_curve(p0, p1, p2, t_values):
        """Returns points along a quadratic bezier curve."""
        return (
            np.outer((1 - t_values) ** 2, p0)
            + 2 * np.outer((1 - t_values) * t_values, p1)
            + np.outer(t_values**2, p2)
        )

    # Main iterative bundling process
    iteration = 0
    while iteration < max_iters:
        gain = 0
        k = 0
        group = {edge_id: UNGROUPED for edge_id in edge_ids}
        for u in edge_ids:
            if group[u] == UNGROUPED:
                best_gain = 0
                best_v = None
                u_neighbors = edge_proximity[u]
                ink_u = calculate_ink((u,))
                for v in u_neighbors:
                    if group[v] == UNGROUPED:
                        ink_v = calculate_ink((v,))
                        # Combined ink if u and v are bundled
                        bundled_ink = calculate_ink((u, v))
                        gain_uv = (ink_u + ink_v) - bundled_ink
                        if gain_uv > best_gain:
                            best_gain = gain_uv
                            best_v = v
                if best_gain > 0 and best_v is not None:
                    # Bundle u and best_v
                    gain += best_gain
                    if group[best_v] != UNGROUPED:
                        group[u] = group[best_v]
                    else:
                        group[u] = k
                        group[best_v] = k
                        k += 1
                else:
                    group[u] = k
                    k += 1
        if gain <= 0:
            break  # No further gain, exit the loop
        total_gain += gain

        # Coalesce edges in the same group
        new_edge_map = {}
        new_edge_ids = []
        new_edge_proximity = defaultdict(set)
        group_to_edge_ids = defaultdict(list)
        for edge_id, grp in group.items():
            group_to_edge_ids[grp].append(edge_id)

        for grp, grp_edge_ids in group_to_edge_ids.items():
            new_edge_id = len(new_edge_ids)
            new_edge_ids.append(new_edge_id)
            new_edge_map[new_edge_id] = (
                grp_edge_ids  # Map new edge ID to list of old edge IDs
            )

            # Update proximity graph
            neighbor_groups = set()
            for eid in grp_edge_ids:
                for neighbor in edge_proximity[eid]:
                    neighbor_grp = group[neighbor]
                    if neighbor_grp != grp:
                        neighbor_groups.add(neighbor_grp)
            for neighbor_grp in neighbor_groups:
                new_edge_proximity[new_edge_id].add(neighbor_grp)

        # Update edge data structures
        edge_ids = new_edge_ids
        edge_proximity = new_edge_proximity
        edge_group_map = new_edge_map

        # Update function to calculate ink for new groups
        @lru_cache(maxsize=None)
        def calculate_ink(edge_id_tuple):
            total_length = 0
            for eid in edge_id_tuple:
                if isinstance(eid, int) and eid in edge_map:
                    source, target = edge_map[eid]
                    start = np.array(positions[source])
                    end = np.array(positions[target])
                    total_length += np.linalg.norm(end - start)
                elif isinstance(eid, int):
                    # It's a bundled edge, need to sum lengths of constituent edges
                    for sub_eid in edge_group_map[eid]:
                        source, target = edge_map[sub_eid]
                        start = np.array(positions[source])
                        end = np.array(positions[target])
                        total_length += np.linalg.norm(end - start)
            return total_length

        iteration += 1

    # After bundling, draw the bundled edges
    for edge_id in edge_ids:
        if edge_id in edge_map:
            # It's a single edge
            source, target = edge_map[edge_id]
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # Retrieve edge data
            link_type = G.get_edge_data(source, target).get("link_type", "Undefined")
            edge_text.append(
                f"<b>{source}</b> → <b>{target}</b><br>Link Type: {link_type}"
            )
        else:
            # It's a bundled edge
            constituent_edges = edge_group_map[edge_id]
            sources = [
                np.array(positions[edge_map[eid][0]]) for eid in constituent_edges
            ]
            targets = [
                np.array(positions[edge_map[eid][1]]) for eid in constituent_edges
            ]
            source_mean = np.mean(sources, axis=0)
            target_mean = np.mean(targets, axis=0)
            control = source_mean + (target_mean - source_mean) * bundle_strength
            t = np.linspace(0, 1, num_curve_points)
            curve_points = quadratic_bezier_curve(source_mean, control, target_mean, t)
            edge_x.extend(curve_points[:, 0].tolist() + [None])
            edge_y.extend(curve_points[:, 1].tolist() + [None])
            # For hover text, list the number of edges and optionally some of the node pairs
            num_edges = len(constituent_edges)
            edge_descriptions = []
            for eid in constituent_edges[:5]:  # Limit to first 5 edges for brevity
                source, target = edge_map[eid]
                edge_descriptions.append(f"{source} → {target}")
            if num_edges > 5:
                edge_descriptions.append("...and more")
            hover_text = f"Bundled Edges: {num_edges}<br>" + "<br>".join(
                edge_descriptions
            )
            edge_text.append(hover_text)

    return edge_x, edge_y, edge_text


def visualize_graph_with_priorities(
    G,
    grouping_order,
    max_path_length=10,
    max_paths_per_node=3,
    vertical_spacing=20,
    horizontal_spacing=10,
    x_spacing=10,
    max_nodes_per_row=15,
    title="Network Graph",
    color_palette=px.colors.qualitative.Alphabet,
    show_warnings=True,
    fig_width=1200,
    fig_height=1000,
    show_legend=True,
    highlight_nodes=None,
    output_file=None,
    path_highlighting=False,
    filter_nodes=None,
    filter_edges=None,
    jitter=0.2,
    bundle_strength=0.95,
    num_curve_points=100,
    max_iters=10,
    bundle_edges=False,
):
    """
    Visualizes the graph G prioritizing node types specified in grouping_order.

    Parameters:
    - G: NetworkX graph.
    - grouping_order: List of node types defining the order of tiers.
    - max_path_length: Maximum length for path calculations.
    - max_paths_per_node: Maximum number of paths to display per node.
    - vertical_spacing: Spacing between tiers.
    - horizontal_spacing: Spacing between nodes within the same tier.
    - x_spacing: Initial spacing for root nodes.
    - max_nodes_per_row: Maximum number of nodes per row in a tier.
    - title: Title of the graph.
    - color_palette: Color palette for node groups.
    - show_warnings: If True, show warnings about unknown node types.
    - fig_width, fig_height: Figure dimensions.
    - show_legend: If True, display the legend.
    - highlight_nodes: List of nodes to highlight.
    - output_file: If provided, save the figure to this file.
    - path_highlighting: If True, highlight paths in hover text.
    - filter_nodes: Function to filter nodes.
    - filter_edges: Function to filter edges.
    - jitter: Amount of jitter in node positions to avoid overlap.
    - bundle_strength, num_curve_points, max_iters: Parameters for edge bundling.
    - bundle_edges: If True, perform edge bundling.
    """
    # Make a copy of the graph to avoid modifying the original
    G = G.copy()

    # Apply node and edge filters if provided
    if filter_nodes is not None:
        nodes_to_keep = [node for node in G.nodes() if filter_nodes(G.nodes[node])]
        G = G.subgraph(nodes_to_keep)
    if filter_edges is not None:
        edges_to_keep = [
            (u, v) for u, v, data in G.edges(data=True) if filter_edges((u, v, data))
        ]
        G = G.edge_subgraph(edges_to_keep).copy()

    # Prepare grouping order and map node types to levels
    grouping_order_lower = [item.strip().lower() for item in grouping_order]
    type_levels = {node_type: idx for idx, node_type in enumerate(grouping_order_lower)}
    max_level = len(grouping_order_lower)

    # Assign node types and check for unknown node types
    unknown_node_types = set()
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "Unknown")
        node_type = str(node_type).strip().lower()
        G.nodes[node]["node_type"] = node_type
        if node_type not in type_levels:
            unknown_node_types.add(node_type)
    if show_warnings and unknown_node_types:
        print(
            "Warning: The following node types are not in grouping_order and will be assigned to a separate tier:",
            unknown_node_types,
        )

    # Identify root nodes (nodes with in-degree zero)
    roots = [n for n, d in G.in_degree() if d == 0]
    num_roots = len(roots)

    # Position root nodes horizontally
    if num_roots > 1:
        x_positions = np.linspace(
            -x_spacing * (num_roots - 1) / 2, x_spacing * (num_roots - 1) / 2, num_roots
        )
    else:
        x_positions = [0]
    positions = {}
    assigned_nodes = set()
    for i, node in enumerate(roots):
        x_pos = x_positions[i]
        y_pos = 0
        positions[node] = (x_pos, y_pos)
        assigned_nodes.add(node)

    # Organize nodes by level based on node type
    nodes_by_level = defaultdict(list)
    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        level = type_levels.get(node_type)
        if level is None:
            level = max_level  # Assign unknown node types to a separate tier
        nodes_by_level[level].append(node)

    # Position nodes by tiers and rows
    for level in sorted(nodes_by_level.keys()):
        nodes_at_level = nodes_by_level[level]
        if not nodes_at_level:
            continue  # Skip empty levels
        num_nodes = len(nodes_at_level)
        num_rows = int(np.ceil(num_nodes / max_nodes_per_row))
        tier_vertical_space = vertical_spacing
        if num_rows > 1:
            row_spacing = tier_vertical_space / num_rows
            row_vertical_offsets = np.linspace(
                (row_spacing * (num_rows - 1)) / 2,
                -(row_spacing * (num_rows - 1)) / 2,
                num_rows,
            )
        else:
            row_vertical_offsets = [0]
        base_y_pos = -level * vertical_spacing
        for row_idx in range(num_rows):
            start_idx = row_idx * max_nodes_per_row
            end_idx = min(start_idx + max_nodes_per_row, num_nodes)
            nodes_in_row = nodes_at_level[start_idx:end_idx]
            nodes_in_row_num = len(nodes_in_row)
            total_width = (nodes_in_row_num - 1) * horizontal_spacing
            tier_left_bound = -total_width / 2
            tier_right_bound = total_width / 2
            x_offsets = np.linspace(tier_left_bound, tier_right_bound, nodes_in_row_num)
            y_pos = base_y_pos + row_vertical_offsets[row_idx]
            for idx, node in enumerate(nodes_in_row):
                # Position nodes under their predecessors if possible
                predecessors = list(G.predecessors(node))
                if predecessors:
                    pred_positions = [
                        positions[pred][0] for pred in predecessors if pred in positions
                    ]
                    if pred_positions:
                        x_pos = np.mean(pred_positions)
                        x_pos = max(tier_left_bound, min(x_pos, tier_right_bound))
                    else:
                        x_pos = x_offsets[idx]
                else:
                    x_pos = x_offsets[idx]
                positions[node] = (x_pos, y_pos)
                assigned_nodes.add(node)

    # Position any unassigned nodes
    unassigned_nodes = [node for node in G.nodes() if node not in positions]
    if unassigned_nodes:
        current_y = min(y for x, y in positions.values()) - vertical_spacing
        num_nodes = len(unassigned_nodes)
        unassigned_total_width = (num_nodes - 1) * horizontal_spacing
        x_offsets = np.linspace(
            -unassigned_total_width / 2, unassigned_total_width / 2, num_nodes
        )
        for idx, node in enumerate(unassigned_nodes):
            x_pos = x_offsets[idx]
            positions[node] = (x_pos, current_y)
            assigned_nodes.add(node)

    # Compute node sizes based on degree
    degrees = np.array([G.degree(node) for node in G.nodes()])
    if degrees.max() > 0:
        node_sizes = 10 + 5 * np.log1p(degrees) / np.log1p(degrees).max()
    else:
        node_sizes = np.full_like(degrees, 10)

    # Assign nodes to groups and colors
    groups = []
    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        group = node_type if node_type in grouping_order_lower else "other"
        groups.append(group)
        G.nodes[node]["group"] = group  # Store the group in node attributes
    # Define unique groups and create color map
    unique_groups = grouping_order_lower + ["other"]
    color_map = {
        group: color_palette[i % len(color_palette)]
        for i, group in enumerate(unique_groups)
    }
    node_colors = [color_map[group] for group in groups]

    # Optionally highlight certain nodes
    if highlight_nodes is not None:
        highlight_set = set(highlight_nodes)
        node_colors = [
            "red" if node in highlight_set else color
            for node, color in zip(G.nodes(), node_colors)
        ]
        node_sizes = [
            size * 1.5 if node in highlight_set else size
            for node, size in zip(G.nodes(), node_sizes)
        ]

    # Add jitter to node positions to avoid overlap
    def add_jitter(pos_dict, jitter_amount=0.5):
        for node in pos_dict:
            x, y = pos_dict[node]
            jitter_x = np.random.uniform(-jitter_amount, jitter_amount)
            jitter_y = np.random.uniform(-jitter_amount, jitter_amount)
            pos_dict[node] = (x + jitter_x, y + jitter_y)

    add_jitter(positions, jitter_amount=jitter)

    # Prepare node coordinates and hover text
    node_x = [positions[node][0] for node in G.nodes()]
    node_y = [positions[node][1] for node in G.nodes()]
    node_text = []
    predecessors = {node: set(G.predecessors(node)) for node in G.nodes()}

    @lru_cache(maxsize=None)
    def get_paths(node, depth):
        if depth > max_path_length:
            return []
        if node in roots:
            return [(node,)]
        paths = []
        for pred in predecessors[node]:
            previous_paths = get_paths(pred, depth + 1)
            for p in previous_paths:
                if len(paths) >= max_paths_per_node:
                    break
                paths.append(p + (node,))
            if len(paths) >= max_paths_per_node:
                break
        return paths

    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        group = G.nodes[node]["group"]
        degree = G.degree(node)
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        # Prepare hover text
        neighbors = list(G.neighbors(node))[:5]
        connections = (
            "<br>".join([f"→ {neighbor}" for neighbor in neighbors])
            if neighbors
            else "No outgoing connections"
        )
        paths = get_paths(node, 0)
        block_type = G.nodes[node].get("block_type", "")
        layer_number = G.nodes[node].get("layer_number", "")
        layer_number_str = ""
        if layer_number:
            layer_number_str = f"Layer_Number: {layer_number}<br>"
        block_type_str = ""
        if block_type:
            block_type_str = f"Block_Type: {block_type}<br>"
        if paths:
            origin_paths_formatted = []
            for path in paths[:max_paths_per_node]:
                if len(path) > 5:
                    path_chunks = [
                        " → ".join(path[i : i + 5]) for i in range(0, len(path), 5)
                    ]
                    path_str = "<br>".join(path_chunks)
                else:
                    path_str = " → ".join(path)
                if path_highlighting:
                    path_str = f"<span style='color:blue'>{path_str}</span>"
                origin_paths_formatted.append(path_str)
            if len(paths) > max_paths_per_node:
                origin_paths_formatted.append("...and more paths")
            origin = "<br><br>".join(origin_paths_formatted)
        else:
            origin = "No origin path found"
        hover_text = (
            f"<b>{node}</b><br>"
            f"Type: {node_type}<br>"
            f"Group: {group}<br>"
            f"Degree: {degree}<br>"
            f"{block_type_str}"
            f"{layer_number_str}"
            f"In-Degree: {in_degree}<br>"
            f"Out-Degree: {out_degree}<br>"
            f"<b>Connections:</b><br>{connections}<br>"
            f"<b>Origin Paths:</b><br>{origin}"
        )
        node_text.append(hover_text)

    # Edge x, y coordinates and hover text
    if bundle_edges:
        # Use the edge bundling algorithm
        edge_x, edge_y, edge_text = create_bundled_edges(
            G,
            positions,
            bundle_strength=bundle_strength,
            num_curve_points=num_curve_points,
            max_iters=max_iters,
        )
    else:
        # Create edges in the standard way
        edge_x = []
        edge_y = []
        edge_text = []
        for u, v, data in G.edges(data=True):
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # Retrieve edge data for hover text
            link_type = data.get("link_type", "Undefined")
            edge_text.append(f"<b>{u}</b> → <b>{v}</b><br>Link Type: {link_type}")

    # Edge trace with adjusted line width and color
    edge_line_width = 0.3 if bundle_edges else 0.05
    edge_line_color = "rgba(150,150,150,0.4)" if bundle_edges else "gray"
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=edge_line_width, color=edge_line_color),
        hoverinfo="text",
        hovertemplate="%{text}",
        text=edge_text,
        mode="lines",
        showlegend=False,
    )

    # Node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertemplate="%{text}",
        text=node_text,
        marker=dict(color=node_colors, size=node_sizes, line_width=0),
        showlegend=False,
    )

    # Combine edge and node traces
    data = [edge_trace, node_trace]

    # Create the figure
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=40, l=40, r=40, t=40),
            width=fig_width,
            height=fig_height,
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, visible=False
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, visible=False
            ),
        ),
    )

    # Add legends if requested
    if show_legend:
        # Create legend entries for node groups, including hover text
        group_traces = []
        for group in unique_groups:
            indices = [i for i, grp in enumerate(groups) if grp == group]
            trace = go.Scatter(
                x=[node_x[i] for i in indices],
                y=[node_y[i] for i in indices],
                mode="markers",
                marker=dict(
                    color=color_map[group],
                    size=[node_sizes[i] for i in indices],
                    line_width=0,
                ),
                name=group.capitalize(),
                showlegend=True,
                text=[node_text[i] for i in indices],
                hoverinfo="text",
                hovertemplate="%{text}",
            )
            group_traces.append(trace)
        # Remove the existing node trace and add group traces
        data = [edge_trace] + group_traces
        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=True,
                hovermode="closest",
                margin=dict(b=40, l=40, r=40, t=40),
                width=fig_width,
                height=fig_height,
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, visible=False
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, visible=False
                ),
            ),
        )

    # Save or display the figure
    if output_file:
        fig.write_image(output_file)
        print(f"Figure saved to {output_file}")
    else:
        fig.show()

    return fig
