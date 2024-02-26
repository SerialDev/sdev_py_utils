import numpy as np
import pandas as pd


def rot_euler(v, xyz):
    """
    * type-def ::np.ndarray ::Tuple[float, float, float] -> np.ndarray
    * ---------------{Function}---------------
        * Rotate vector v (or array of vectors) by the Euler angles xyz.
    * ----------------{Returns}---------------
        * -> rotated_v ::np.ndarray | The rotated vector(s)
    * ----------------{Params}----------------
        * : v   ::np.ndarray               | The input vector(s) to be rotated
        * : xyz ::Tuple[float, float, float]| The Euler angles for rotation
    * ----------------{Usage}-----------------
        * >>> v = np.array([1, 0, 0])
        * >>> xyz = (0, np.pi/2, 0)
        * >>> rotated_v = rot_euler(v, xyz)
    * ----------------{Notes}-----------------
        * This function rotates the input vector(s) by the given Euler angles using a rotation matrix.
    """
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    from scipy.linalg import expm

    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis * -theta)))
    return v


def pd_distance_matrix(df):
    """
    * type-def ::pd.DataFrame -> pd.DataFrame
    * ---------------{Function}---------------
        * Computes the distance matrix of a DataFrame containing numeric data.
    * ----------------{Returns}---------------
        * -> distance_df ::pd.DataFrame | The distance matrix as a DataFrame
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The input DataFrame containing numeric data
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        * >>> distance_df = pd_distance_matrix(df)
    * ----------------{Notes}-----------------
        * This function computes the distance matrix of a DataFrame containing numeric data using the scipy.spatial.distance_matrix function.
    """
    from scipy.spatial import distance_matrix

    df = df.select_dtypes("number")
    return pd.DataFrame(
        distance_matrix(df.values, df.values), index=df.index, columns=df.index
    )


def pmi(df, positive=True):
    """
    * type-def ::pd.DataFrame ::bool -> pd.DataFrame
    * ---------------{Function}---------------
        * Computes the pointwise mutual information (PMI) matrix of a given DataFrame.
    * ----------------{Returns}---------------
        * -> pmi_df ::pd.DataFrame | The PMI matrix as a DataFrame
    * ----------------{Params}----------------
        * : df      ::pd.DataFrame | The input DataFrame
        * : positive::bool         | Whether to compute positive PMI (default: True)
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        * >>> pmi_df = pmi(df)
    * ----------------{Notes}-----------------
        * This function computes the pointwise mutual information (PMI) matrix of a given DataFrame, optionally making it positive PMI.
        * Pointwise mutual information:
        *  https://stackoverflow.com/questions/
        * 58701337/how-to-construct-ppmi-matrix-from-a-text-corpus
    """
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide="ignore"):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


def reachability_matrix(G):
    """
    * type-def ::nx.DiGraph -> np.ndarray
    * ---------------{Function}---------------
        * Computes the reachability matrix of a given directed graph.
    * ----------------{Returns}---------------
        * -> R ::np.ndarray | The reachability matrix
    * ----------------{Params}----------------
        * : G ::nx.DiGraph | The input directed graph
    * ----------------{Usage}-----------------
        * >>> import networkx as nx
        * >>> G = nx.DiGraph()
        * >>> G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        * >>> R = reachability_matrix(G)
    * ----------------{Notes}-----------------
        * This function computes the reachability matrix for a directed graph using NetworkX's all_pairs_shortest_path_length function.
        * The reachability matrix indicates whether there is a path from one node to another in the graph.
    """
    import networkx as nx
    import numpy as np

    # G = nx.DiGraph(c)
    np.random.seed(42)
    c = np.random.rand(4, 4)
    length = dict(nx.all_pairs_shortest_path_length(G))
    R = np.array(
        [[length.get(m, {}).get(n, 0) > 0 for m in G.nodes] for n in G.nodes],
        dtype=np.int32,
    )
    return R



def get_mtf_map(timeseries, mtf, step_size=0, colormap='jet', reversed_cmap=False):
    import tsia

    # discretized = tsia.markov.discretize(cb.tp_revenue)
    # transition_matrix = tsia.markov.markov_transition_matrix(discretized[0])
    # transition_probs = tsia.markov.markov_transition_probabilities(transition_matrix)
    # transition_mapped = get_mtf_map(cb.tp_revenue, transition_matrix)
    # tsia.plot.plot_mtf_metrics(transition_matrix)
    # tsia.markov.get_multivariate_mtf([cb.tp_revenue])
    # tsia.plot.plot_timeseries_quantiles(cb.tp_revenue, discretized[1])

    image_size = mtf.shape[0]
    mtf_min = np.min(mtf)
    mtf_max = np.max(mtf)
    mtf_range = mtf_max - mtf_min
    mtf_colors = (np.diag(mtf, k=step_size) - mtf_min) / mtf_range
    
    if reversed_cmap:
        colormap = plt.cm.get_cmap(colormap).reversed()
    else:
        colormap = plt.cm.get_cmap(colormap)
    
    mtf_map = []
    sequences_width = timeseries.shape[0] / image_size
    for i in range(image_size - step_size):
        c = colormap(mtf_colors[i])
        start = int(i * sequences_width)
        end = int((i+1) * sequences_width)
        data = timeseries.iloc[start:end]
        
        current_map = {
            'color': c,
            'slice': data
        }
        mtf_map.append(current_map)
        
    for i in range(image_size - step_size, image_size):
        c = '#DDDDDD'
        start = int(i * sequences_width)
        end = int((i+1) * sequences_width)
        data = timeseries.iloc[start:end]

        current_map = {
            'color': c,
            'slice': data
        }
        mtf_map.append(current_map)
        
    return mtf_map

