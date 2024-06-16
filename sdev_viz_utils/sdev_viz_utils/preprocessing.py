from itertools import chain, count

_attrs = dict(id="id", children="children")


def timeline_label_datapoint(ordinal_string, start_time, end_time):
    '''
* ---------------Function---------------
* Creates a timeline label datapoint
* ----------------Returns---------------
* -> dict
* ----------------Params---------------
* ordinal_string :: str
* start_time :: <any>
* end_time :: <any>
* ----------------Usage---------------
* timeline_label_datapoint("Label", 1, 2)

    '''
    return {"timeRange": [start_time, end_time], "val": ordinal_string}


def timeline_group_datapoint(label_name, label_datapoints):
    '''
* ---------------Function---------------
* Creates a timeline group datapoint
* ----------------Returns---------------
* -> dict
* ----------------Params---------------
* label_name :: str
* label_datapoints :: list
* ----------------Usage---------------
* timeline_group_datapoint("Label", [...])
    '''
    return {"label": label_name, "data": label_datapoints}


def timeline_chart_datapoint(group_name, group_datapoints):
    '''
* ---------------Function---------------
* Creates a timeline chart datapoint
* ----------------Returns---------------
* -> dict
* ----------------Params---------------
* group_name :: str
* group_datapoints :: list
* ----------------Usage---------------
* timeline_chart_datapoint("Group", [...])
    '''
    return {"group": group_name, "data": group_datapoints}


def tree_data(G, root, attrs=_attrs):
    """Return data in tree format that is suitable for JSON serialization
    and use in Javascript documents.

    Parameters
    ----------
    G : NetworkX graph
       G must be an oriented tree

    root : node
       The root of the tree

    attrs : dict
        A dictionary that contains two keys 'id' and 'children'. The
        corresponding values provide the attribute names for storing
        NetworkX-internal graph data. The values should be unique. Default
        value: :samp:`dict(id='id', children='children')`.

        If some user-defined graph data use these attribute names as data keys,
        they may be silently dropped.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If values in attrs are not unique.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1,2)])
    >>> data = json_graph.tree_data(G,root=1)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Node attributes are stored in this format but keys
    for attributes must be strings if you want to serialize with JSON.

    Graph and edge attributes are not stored.

    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    tree_graph, node_link_data, node_link_data
    """
    if not G.is_directed():
        raise TypeError("G is not directed.")

    id_ = attrs["id"]
    children = attrs["children"]
    if id_ == children:
        raise nx.NetworkXError("Attribute names are not unique.")

    def add_children(n, G):
            """
    * ---------------Function---------------
    * Recursively generates a tree-like data structure from a graph, 
      where each node's value is a dictionary containing the node's attributes 
      and a list of its children.
    * ----------------Returns---------------
    * -> list[dict] : A list of dictionaries, where each dictionary represents a node 
                      in the graph, containing the node's attributes and its children.
    * ----------------Params----------------
    * n : <any> : The current node in the graph.
    * G : <any> : The graph data structure.
    * ----------------Usage-----------------
    * To generate a tree-like data structure from a graph, where each node's value 
      is a dictionary containing the node's attributes and a list of its children.
    """

        nbrs = G[n]
        if len(nbrs) == 0:
            return []
        children_ = []
        for child in nbrs:
            d = dict(chain(G.node[child].items(), [(id_, child)]))
            c = add_children(child, G)
            if c:
                d[children] = c
            children_.append(d)
        return children_

    data = dict(chain(G.node[root].items(), [(id_, root)]))
    data[children] = add_children(root, G)
    return data


def tree_graph(data, attrs=_attrs):
    """Return graph from tree data format.

    Parameters
    ----------
    data : dict
        Tree formatted graph data

    Returns
    -------
    G : NetworkX DiGraph

    attrs : dict
        A dictionary that contains two keys 'id' and 'children'. The
        corresponding values provide the attribute names for storing
        NetworkX-internal graph data. The values should be unique. Default
        value: :samp:`dict(id='id', children='children')`.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1,2)])
    >>> data = json_graph.tree_data(G,root=1)
    >>> H = json_graph.tree_graph(data)

    Notes
    -----
    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    tree_graph, node_link_data, adjacency_data
    """
    graph = nx.DiGraph()
    id_ = attrs["id"]
    children = attrs["children"]

    def add_children(parent, children_):
        '''
Here is the converted docstring:

* ---------------add_children Function---------------
* Recursively adds children nodes to a graph, creating edges and nodes as necessary.
* ----------------Returns---------------
* -> None
* ----------------Params----------------
* parent : <any>
* children_ : list of dictionaries
    where each dictionary represents a node with the following keys:
    - id_ : <any> (required)
    - children : list of dictionaries (optional)
    - other attributes : passed as attributes to the node
* ----------------Usage-----------------
* Call this function with a parent node and a list of child nodes to add to the graph.

        '''
        for data in children_:
            child = data[id_]
            graph.add_edge(parent, child)
            grandchildren = data.get(children, [])
            if grandchildren:
                add_children(child, grandchildren)
            nodedata = dict(
                (make_str(k), v) for k, v in data.items() if k != id_ and k != children
            )
            graph.add_node(child, attr_dict=nodedata)

    root = data[id_]
    children_ = data.get(children, [])
    nodedata = dict(
        (make_str(k), v) for k, v in data.items() if k != id_ and k != children
    )
    graph.add_node(root, attr_dict=nodedata)
    add_children(root, children_)
    return graph


def pandas_df_to_markdown_table(df):
    """
* ---------------Function---------------
* Converts a pandas DataFrame to a markdown table
* ----------------Returns---------------
* -> result :: Markdown object
* ----------------Params----------------
* df :: pandas.DataFrame | The input DataFrame to be converted to markdown table
* ----------------Usage-----------------
Example usage:
```
df = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': [4, 5, 6]})
pandas_df_to_markdown_table(df)
```
This will display the markdown table in the output.
    """
    
    from IPython.display import Markdown, display

    fmt = ["---" for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))


def to_markdown(df):
    '''
    * ---------------Function---------------
    * Converts a pandas DataFrame to a Markdown table
    * ----------------Returns---------------
    * -> display :: None | Displays the Markdown table
    * ----------------Params----------------
    * df :: pandas.DataFrame | The input DataFrame to be converted
    * ----------------Usage-----------------
    * Call the function with a pandas DataFrame as an argument, and it will display the table in Markdown format.
    '''
    from subprocess import Popen, PIPE

    s = df.to_latex()
    p = Popen("pandoc -f latex -t markdown", stdin=PIPE, stdout=PIPE, shell=True)
    stdoutdata, _ = p.communicate(input=s.encode("utf-8"))
    return stdoutdata.decode("utf-8")


def df_to_markdown(df, float_format="%.2g"):
    """
    * ---------------Function---------------
    * Converts a pandas DataFrame to markdown-formatted text. DataFrame should not contain any `|` characters.
    * ----------------Returns---------------
    * -> str
    * ----------------Params----------------
    * df :: pandas.DataFrame
    * float_format :: str (default is "%.2g")
    * ----------------Usage-----------------
    * Use this function to convert a pandas DataFrame into a markdown-formatted string. The resulting string can be used to generate markdown tables.
    """
    from os import linesep

    return linesep.join(
        [
            "|".join(df.columns),
            "|".join(4 * "-" for i in df.columns),
            df.to_csv(sep="|", index=False, header=False, float_format=float_format),
        ]
    ).replace("|", " | ")
