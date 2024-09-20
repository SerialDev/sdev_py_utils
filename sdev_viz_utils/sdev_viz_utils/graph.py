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
