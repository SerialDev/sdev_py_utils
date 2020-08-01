import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """
	# source code is from Colin Raffel, GitHub repo: https://github.com/craffel

    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size):
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 4.0,
                color="w",
                ec="k",
                zorder=4,
            )
            # Add texts
            if n == 0:
                plt.text(
                    left - 0.130,
                    layer_top - m * v_spacing,
                    r"$X_{" + str(m + 1) + "}$",
                    fontsize=15,
                )
            elif (n_layers == 3) & (n == 1):
                plt.text(
                    n * h_spacing + left + 0.00,
                    layer_top - m * v_spacing + (v_spacing / 8.0 + 0.01 * v_spacing),
                    r"$H_{" + str(m + 1) + "}$",
                    fontsize=15,
                )
            elif n == n_layers - 1:
                plt.text(
                    n * h_spacing + left + 0.10,
                    layer_top - m * v_spacing,
                    r"$y_{" + str(m + 1) + "}$",
                    fontsize=15,
                )

            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    c="k",
                )

                ax.add_artist(line)

    # Input-Arrows
    layer_top_0 = v_spacing * (layer_sizes[0] - 1) / 2.0 + (top + bottom) / 2.0
    for m in range(layer_sizes[0]):
        plt.arrow(
            left - 0.18,
            layer_top_0 - m * v_spacing,
            0.12,
            0,
            lw=1,
            head_width=0.01,
            head_length=0.02,
            color="k",
        )
    # Output-Arrows
    layer_top_0 = v_spacing * (layer_sizes[-1] - 1) / 2.0 + (top + bottom) / 2.0
    for m in range(layer_sizes[-1]):
        plt.arrow(
            right + 0.015,
            layer_top_0 - m * v_spacing,
            0.16 * h_spacing,
            0,
            lw=1,
            head_width=0.01,
            head_length=0.02,
            color="k",
        )


def root_mean_sqr(x, flag="norm"):
    if flag == "norm":
        return np.sqrt(x.dot(x) / x.size)
    if flag == "complex":
        return np.sqrt(np.vdot(x, x) / x.size)


# --------------{Flare Js}--------------#


from collections import OrderedDict
import json


def next_node(node, key):
    result = None
    children = node.setdefault("children", [])
    for child in children:
        if child["name"] == key:
            result = child
    if not result:
        result = OrderedDict(name=key)
        children.append(result)
    return result


def process_nodes(csv_data):
    """
    fake_csv = [ 'A,AA,AAA',
                 'A,AB,ABA',
                 'A,AB,ABB',
                 'B,BA,BA'
               ]
    """
    tree_root = OrderedDict(name="flare")

    for line in csv_data:
        categories = line.split(",")
        node = tree_root
        for cat in categories:
            node = next_node(node, cat)

    return json.dumps(tree_root, indent=2)


def flatten(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def pd_to_flare(df):
    return process_nodes(flatten(df.as_matrix()))


# ------------{for vikz}------------#


def timeseries_to_vizk(x, y):
    return [{x[i]: y[i] for j in range(len(x))} for i in range(len(x))]
