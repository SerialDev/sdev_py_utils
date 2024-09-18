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
def visualize_graph_with_priorities(
    G,
    grouping_order,
    max_path_length=10,
    max_paths_per_node=3,
    vertical_spacing=20,  # Spacing between tiers
    horizontal_spacing=10,  # Spacing between nodes within a row
    x_spacing=10,
    max_nodes_per_row=15,  # Maximum nodes per row within a tier
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
    jitter=0,
):
    """
    Visualize a NetworkX graph with nodes arranged according to a specified grouping order.
    Nodes are placed in their own tiers based on 'node_type', with adequate spacing to prevent overlap.
    Multiple paths from root nodes are displayed in the hover text.

    Parameters:
    - G: NetworkX graph
    - grouping_order: List of node types to prioritize in the sorting and grouping
    - Other parameters: As previously defined, with adjusted defaults
    """
    import pandas as pd
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np
    import plotly.express as px
    from collections import defaultdict
    from functools import lru_cache

    # Copy the graph to avoid modifying the original
    G = G.copy()

    # Apply node filtering if provided
    if filter_nodes is not None:
        nodes_to_keep = [node for node in G.nodes() if filter_nodes(G.nodes[node])]
        G = G.subgraph(nodes_to_keep)

    # Apply edge filtering if provided
    if filter_edges is not None:
        edges_to_keep = [
            (u, v) for u, v, data in G.edges(data=True) if filter_edges((u, v, data))
        ]
        G = G.edge_subgraph(edges_to_keep).copy()

    # Normalize and validate node_type values
    grouping_order_lower = [item.strip().lower() for item in grouping_order]
    type_levels = {node_type: idx for idx, node_type in enumerate(grouping_order_lower)}
    max_level = len(grouping_order_lower)

    unknown_node_types = set()

    # Ensure node types are normalized and identify unknown types
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "Unknown")
        node_type = str(node_type).strip().lower()
        G.nodes[node]["node_type"] = node_type  # Update node_type

        if node_type not in type_levels:
            unknown_node_types.add(node_type)

    # Warn about unknown node types
    if show_warnings and unknown_node_types:
        print(
            "Warning: The following node types are not in grouping_order and will be assigned to a separate tier:",
            unknown_node_types,
        )

    # Identify root nodes (nodes with no incoming edges)
    roots = [n for n, d in G.in_degree() if d == 0]

    # Assign x positions to root nodes
    num_roots = len(roots)
    if num_roots > 1:
        x_positions = np.linspace(
            -x_spacing * (num_roots - 1) / 2, x_spacing * (num_roots - 1) / 2, num_roots
        )
    else:
        x_positions = [0]  # Single root node at center

    positions = {}
    assigned_nodes = set()

    # Map root nodes to x positions
    for i, node in enumerate(roots):
        x_pos = x_positions[i]
        y_pos = 0  # Top level (Tier 0)
        positions[node] = (x_pos, y_pos)
        assigned_nodes.add(node)

    # Collect nodes by their levels based on node_type
    nodes_by_level = defaultdict(list)
    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        level = type_levels.get(node_type)
        if level is None:
            level = max_level  # Assign unknown types to a separate tier
        nodes_by_level[level].append(node)

    # Assign positions to nodes at each level
    for level in sorted(nodes_by_level.keys()):
        nodes_at_level = nodes_by_level[level]

        # Skip if no nodes at this level
        if not nodes_at_level:
            continue

        num_nodes = len(nodes_at_level)

        # Calculate the number of rows needed
        num_rows = int(np.ceil(num_nodes / max_nodes_per_row))

        # Total vertical space allocated for this tier
        tier_vertical_space = vertical_spacing

        # Calculate the vertical positions for rows within the tier
        if num_rows > 1:
            # Ensure rows stay within the tier's vertical space
            row_spacing = tier_vertical_space / num_rows
            row_vertical_offsets = np.linspace(
                (row_spacing * (num_rows - 1)) / 2,
                -(row_spacing * (num_rows - 1)) / 2,
                num_rows,
            )
        else:
            row_vertical_offsets = [0]

        # Base vertical position for this tier
        base_y_pos = -level * vertical_spacing

        # Now, for each row, position the nodes
        for row_idx in range(num_rows):
            start_idx = row_idx * max_nodes_per_row
            end_idx = min(start_idx + max_nodes_per_row, num_nodes)
            nodes_in_row = nodes_at_level[start_idx:end_idx]
            nodes_in_row_num = len(nodes_in_row)

            # Determine horizontal range for this row
            total_width = (nodes_in_row_num - 1) * horizontal_spacing
            tier_left_bound = -total_width / 2
            tier_right_bound = total_width / 2

            # Create x_offsets for the nodes in this row
            x_offsets = np.linspace(tier_left_bound, tier_right_bound, nodes_in_row_num)

            # Vertical position for this row within the tier
            y_pos = base_y_pos + row_vertical_offsets[row_idx]

            for idx, node in enumerate(nodes_in_row):
                # Position each node regardless of whether it already has a position
                predecessors = list(G.predecessors(node))
                # If the node has predecessors, align it under their average x position within constraints
                if predecessors:
                    pred_positions = [
                        positions[pred][0] for pred in predecessors if pred in positions
                    ]
                    if pred_positions:
                        x_pos = np.mean(pred_positions)
                        # Ensure x_pos stays within the row bounds
                        x_pos = max(tier_left_bound, min(x_pos, tier_right_bound))
                    else:
                        x_pos = x_offsets[idx]
                else:
                    x_pos = x_offsets[idx]
                positions[node] = (x_pos, y_pos)
                assigned_nodes.add(node)

    # Identify unassigned nodes (should be none, but just in case)
    unassigned_nodes = [node for node in G.nodes() if node not in positions]

    # Assign positions to unassigned nodes
    if unassigned_nodes:
        current_y = (
            min(y for x, y in positions.values()) - vertical_spacing
        )  # Start below the lowest assigned node
        num_nodes = len(unassigned_nodes)
        unassigned_total_width = (num_nodes - 1) * horizontal_spacing
        x_offsets = np.linspace(
            -unassigned_total_width / 2, unassigned_total_width / 2, num_nodes
        )
        for idx, node in enumerate(unassigned_nodes):
            x_pos = x_offsets[idx]
            positions[node] = (x_pos, current_y)
            assigned_nodes.add(node)

    # Compute node sizes and colors
    degrees = np.array([G.degree(node) for node in G.nodes()])
    if degrees.max() > 0:
        node_sizes = 10 + 20 * np.log1p(degrees) / np.log1p(degrees).max()
    else:
        node_sizes = np.full_like(degrees, 10)

    # Optionally adjust node sizes to prevent overlaps
    node_sizes = node_sizes * 0.8  # Reduce sizes by 20%

    # Map groups to colors
    groups = []
    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        group = node_type if node_type in grouping_order_lower else "other"
        groups.append(group)
    unique_groups = grouping_order_lower + ["other"]  # Preserve order
    color_map = {
        group: color_palette[i % len(color_palette)]
        for i, group in enumerate(unique_groups)
    }
    node_colors = [color_map[group] for group in groups]

    # Highlight specified nodes
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

    # Add jitter to node positions
    def add_jitter(pos_dict, jitter_amount=0.5):
        for node in pos_dict:
            x, y = pos_dict[node]
            jitter_x = np.random.uniform(-jitter_amount, jitter_amount)
            jitter_y = np.random.uniform(-jitter_amount, jitter_amount)
            pos_dict[node] = (x + jitter_x, y + jitter_y)

    # Apply jitter to positions
    add_jitter(positions, jitter_amount=jitter)

    # Get node positions
    node_x = [positions[node][0] for node in G.nodes()]
    node_y = [positions[node][1] for node in G.nodes()]

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertemplate="%{text}",
        text=[],  # Will populate later
        marker=dict(color=node_colors, size=node_sizes, line_width=0),
    )

    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        if edge[0] in positions and edge[1] in positions:
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            link_type = edge[2].get("link_type", "Undefined")
            edge_text.append(
                f"<b>{edge[0]}</b> → <b>{edge[1]}</b><br>Link Type: {link_type}"
            )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="text",
        hovertemplate="%{text}",
        text=edge_text,
        mode="lines",
    )

    # Generate hover text for nodes, including all origin paths
    node_text = []

    # Optimized Path Computation
    # For each node, we'll store a set of predecessors
    predecessors = {node: set(G.predecessors(node)) for node in G.nodes()}

    # Function to reconstruct paths using predecessors
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
                    break  # Short-circuit if we've found enough paths
                paths.append(p + (node,))
            if len(paths) >= max_paths_per_node:
                break  # Short-circuit if we've found enough paths
        return paths

    # Compute paths for nodes
    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        group = node_type if node_type in grouping_order_lower else "other"
        degree = G.degree(node)
        # Get up to 5 connections
        neighbors = list(G.neighbors(node))[:5]
        connections = (
            "<br>".join([f"→ {neighbor}" for neighbor in neighbors])
            if neighbors
            else "No outgoing connections"
        )

        # Get all origin paths
        paths = get_paths(node, 0)
        if paths:
            # Limit the number of paths displayed to avoid clutter
            origin_paths_formatted = []
            for path in paths[:max_paths_per_node]:
                path = path  # Tuple of nodes
                if len(path) > 5:
                    # Insert <br> every 5 nodes
                    path_chunks = [
                        " → ".join(path[i : i + 5]) for i in range(0, len(path), 5)
                    ]
                    path_str = "<br>".join(path_chunks)
                else:
                    path_str = " → ".join(path)

                # Highlight the path if enabled
                if path_highlighting:
                    path_str = f"<span style='color:blue'>{path_str}</span>"

                origin_paths_formatted.append(path_str)
            if len(paths) > max_paths_per_node:
                origin_paths_formatted.append("...and more paths")
            origin = "<br><br>".join(origin_paths_formatted)
        else:
            origin = "No origin path found"

        # Assemble hover text
        hover_text = (
            f"<b>{node}</b><br>"
            f"Type: {node_type}<br>"
            f"Group: {group}<br>"
            f"Degree: {degree}<br>"
            f"<b>Connections:</b><br>{connections}<br>"
            f"<b>Origin Paths:</b><br>{origin}"
        )
        node_text.append(hover_text)

    # Set the node text
    node_trace.text = node_text

    # Create legend for groups
    legend_traces = []
    if show_legend:
        for group in unique_groups:
            legend_traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color_map[group]),
                    legendgroup=group,
                    showlegend=True,
                    name=group,
                )
            )

    # Assemble all traces
    data = [edge_trace] + legend_traces + [node_trace]

    # Create figure
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=show_legend,
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

    # Save the figure to a file if output_file is specified
    if output_file:
        fig.write_image(output_file)
        print(f"Figure saved to {output_file}")

    return fig
