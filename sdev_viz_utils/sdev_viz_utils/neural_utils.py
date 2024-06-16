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
    '''
    * ---------------Function---------------
    * Calculates the root mean square of the input array x, 
      depending on the specified flag, either in normal or complex mode.
    * ----------------Returns---------------
    * -> result :: numpy.float64
    * ----------------Params----------------
    * x :: numpy.ndarray
            + Input array for calculation of root mean square.
    * flag :: str, optional (default="norm")
            + Flag to specify the mode of calculation. Either "norm" for normal mode or "complex" for complex mode.
    * ----------------Usage-----------------
    * >>> root_mean_sqr(x) 
    * Calculate the root mean square of array x in normal mode.
    * >>> root_mean_sqr(x, "complex") 
    * Calculate the root mean square of array x in complex mode.
    '''
    if flag == "norm":
        return np.sqrt(x.dot(x) / x.size)
    if flag == "complex":
        return np.sqrt(np.vdot(x, x) / x.size)


# --------------{Flare Js}--------------#


from collections import OrderedDict
import json


def next_node(node, key):
    """
    * ---------------Function---------------
    * next_node searches for a child node with a given key in the input node's
    * children.
    * ----------------Returns---------------
    * -> result :: OrderedDict or None
    * 'Success' if the operation was successful, 'Failure' otherwise
    * ----------------Params----------------
    * node :: dict
    * A dictionary containing the current node data.
    * key :: str
    * The name of the child node to search for.
    * ----------------Usage-----------------
    * To search for a child node with a given key, provide a node and key as
    * arguments:
    * 
    * ────────────────────
    * node = {
    *     "children": [
    *         {"name": "child1"},
    *         {"name": "child2"},
    *     ]
    * }
    * 
    * result = next_node(node, "child1")
    * print(result)
    * ────────────────────
    * 
    * • In this example, 'child1' is found in the node's children, and the resulting
    *   OrderedDict is returned. If a child node with the provided key does not exist,
    *   a new OrderedDict will be created and appended to the 'children' list with the
    *   specified key.
    * • The 'children' list within the node argument will be altered in case the key
    *   does not exist in the original list. A new OrderedDict with the specified key
    *   will be appended and set as the result.
    * 
    * ────────────────────
    * node = {
    *     "children": [
    *         {"name": "child1"},
    *     ]
    * }
    * 
    * result = next_node(node, "child3")
    * print(result)
    * print(node)
    * ────────────────────
    * 
    * • In this example, 'child3' does not exist in the original list, a new
    *   OrderedDict will be created with the specified key, and appended to the
    *   'children' list.
    * • As a result:
    * 
    * ────────────────────
    * result = OrderedDict(name='child3')
   n* ode = {
    *    "children": [
    *       {"name": "child1"},
    *       OrderedDict(name='child3'),
    *  ]
    * }
    * ────────────────────

    """
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
    * ------------Function---------------
    * Converts a CSV string containing a list of categorized items into a nested
      dictionary structure, where each nested dictionary represents a category
      and its subcategories. The result is returned as a JSON string.
    * ----------------Returns---------------
    * -> result ::str |'Success' if the operation was successful, 'Failure' otherwise
    * ----------------Params----------------
    * csv_data :: <any> | The input CSV data as a string.
    * ----------------Usage-----------------
    * process_nodes('A,AA,AAA\nA,AB,ABA\nA,AB,ABB\nB,BA,BA')
    """
    tree_root = OrderedDict(name="flare")

    for line in csv_data:
        categories = line.split(",")
        node = tree_root
        for cat in categories:
            node = next_node(node, cat)

    return json.dumps(tree_root, indent=2)


def flatten(l):
    """
    *  ---------------Function---------------
    *  Flattens a list of lists into a single list
    *  ----------------Returns---------------
    *  -> result ::list|The flattened list
    *  ----------------Params----------------
    *  l ::list|A list containing other lists
    *  ----------------Usage-----------------
    * flatten([[1, 2], [3, 4], [5, 6]]) ->
    *  [1, 2, 3, 4, 5, 6]

    """
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def pd_to_flare(df):
    '''
* ---------------Function---------------
* Converts a Pandas DataFrame to a Flare JSON format
* ----------------Returns---------------
* -> str: 'Success' if the operation was successful, 'Failure' otherwise
* ----------------Params----------------
* df :: pandas.DataFrame: The input DataFrame to be converted
* ----------------Usage-----------------
* >>> pd_to_flare(my_dataframe)
* This function can be used to prepare data for visualization in Flare.
    '''
    return process_nodes(flatten(df.as_matrix()))


# ------------{for vikz}------------#


def timeseries_to_vizk(x, y):
    '''
    * ---------------Function---------------
    * Converts a timeseries into a format suitable for visualization.
    * ----------------Returns---------------
    * -> list[dict] : a list of dictionaries where each dictionary represents a data point
    * ----------------Params----------------
    * x : list : the x-coordinates of the timeseries
    * y : list : the y-coordinates of the timeseries
    * ----------------Usage-----------------
    * >>> timeseries_to_vizk([1, 2, 3], [4, 5, 6])
    * [{1: 4}, {1: 4}, {1: 4}]
    '''
    return [{x[i]: y[i] for j in range(len(x))} for i in range(len(x))]
