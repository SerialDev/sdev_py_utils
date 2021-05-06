import numpy as np
import pandas as pd


def rot_euler(v, xyz):
    """ Rotate vector v (or array of vectors) by the euler angles xyz """
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    from scipy.linalg import expm

    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis * -theta)))
    return v


def pd_distance_matrix(df):
    from scipy.spatial import distance_matrix

    df = df.select_dtypes("number")
    return pd.DataFrame(
        distance_matrix(df.values, df.values), index=df.index, columns=df.index
    )


def pmi(df, positive=True):
    # Pointwise mutual information:
    # https://stackoverflow.com/questions/
    # 58701337/how-to-construct-ppmi-matrix-from-a-text-corpus
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
