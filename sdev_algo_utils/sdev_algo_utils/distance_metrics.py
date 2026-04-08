from math import sqrt, isinf
import numpy as np
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from scipy.ndimage import minimum_filter1d, maximum_filter1d

_INF = float("inf")


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w) : min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if isinf(w) or (max(0, i - w) <= j <= min(c, i + w)):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    if warp == 1:
        for i in range(r):
            min_da = np.minimum(D0[i, :c], D0[i, 1 : c + 1])
            i1 = min(i + 1, r)
            for j in range(c):
                D1[i, j] += min(min_da[j], D0[i1, j])
    else:
        for i in range(r):
            for j in range(c):
                min_list = [D0[i, j]]
                for k in range(1, warp + 1):
                    min_list += [D0[min(i + k, r), j], D0[i, min(j + k, c)]]
                D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    """
    * type-def ::(np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    * ---------------{Function}---------------
        * Computes the traceback path for dynamic time warping (DTW) based on the accumulated cost matrix D.
    * ----------------{Returns}---------------
        * : p ::np.ndarray | The path indices for the first sequence.
        * : q ::np.ndarray | The path indices for the second sequence.
    * ----------------{Params}----------------
        * : D ::np.ndarray | The accumulated cost matrix for dynamic time warping.
    * ----------------{Usage}-----------------
        * >>> D = np.array([[0, 1, 2], [1, 1, 2], [2, 1, 1]])
        * >>> p, q = _traceback(D)
    * ----------------{Output}----------------
        * The traceback path indices for both sequences.
    * ----------------{Dependencies}---------
        * This function requires the following libraries:
          * numpy
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the size of the input matrix D. For large
          * matrices, consider using more efficient traceback algorithms or reducing the size of the input matrix.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input matrix D.
    """
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.append(i)
        q.append(j)
    # Reverse once at the end: O(n) total instead of O(n^2) from insert(0).
    p.reverse()
    q.reverse()
    return np.array(p), np.array(q)


if __name__ == "__main__":
    w = inf
    s = 1.0
    if 1:  # 1-D numeric
        from sklearn.metrics.pairwise import manhattan_distances

        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_fun = manhattan_distances
        w = 1
        # s = 1.2
    elif 0:  # 2-D numeric
        from sklearn.metrics.pairwise import euclidean_distances

        x = [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [4, 3],
            [2, 3],
            [1, 1],
            [2, 2],
            [0, 1],
        ]
        y = [
            [1, 0],
            [1, 1],
            [1, 1],
            [2, 1],
            [4, 3],
            [4, 3],
            [2, 3],
            [3, 1],
            [1, 2],
            [1, 0],
        ]
        dist_fun = euclidean_distances
    else:  # 1-D list of strings
        from nltk.metrics.distance import edit_distance

        # x = ['we', 'shelled', 'clams', 'for', 'the', 'chowder']
        # y = ['class', 'too']
        x = ["i", "soon", "found", "myself", "muttering", "to", "the", "walls"]
        y = ["see", "drown", "himself"]
        # x = 'we talked about the situation'.split()
        # y = 'we talked about the situation'.split()
        dist_fun = edit_distance
    dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)

    # Vizualize
    from matplotlib import pyplot as plt

    plt.imshow(cost.T, origin="lower", cmap=plt.cm.Reds, interpolation="nearest")
    plt.plot(path[0], path[1], "-o")  # relation
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("tight")
    if isinf(w):
        plt.title("Minimum distance: {}, slope weight: {}".format(dist, s))
    else:
        plt.title(
            "Minimum distance: {}, window widht: {}, slope weight: {}".format(
                dist, w, s
            )
        )
    plt.show()


def DTWDistance(s1, s2):
    """Compute the Dynamic Time Warping distance between two 1-D sequences.

    Uses a 2D Python list for the cost matrix instead of a dict with tuple
    keys, eliminating per-cell hashing overhead.  The outer-loop value
    ``s1_i`` is cached to halve list index lookups in the inner loop.

    Usage::

        from sklearn.metrics.pairwise import pairwise_distances
        p = pairwise_distances(test, metric=DTWDistance)
    """
    n, m = len(s1), len(s2)
    DTW = [[_INF] * (m + 1) for _ in range(n + 1)]
    DTW[0][0] = 0.0
    for i in range(1, n + 1):
        s1_i = s1[i - 1]
        DTW_i = DTW[i]
        DTW_prev = DTW[i - 1]
        for j in range(1, m + 1):
            cost = (s1_i - s2[j - 1]) ** 2
            DTW_i[j] = cost + min(DTW_prev[j], DTW_i[j - 1], DTW_prev[j - 1])
    return sqrt(DTW[n][m])


def euclid_dist(t1, t2):
    return sqrt(sum((t1 - t2) ** 2))


def LB_Keogh(s1, s2, r):
    """
    https://nbviewer.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb

    Vectorized using scipy rolling min/max filters (O(n) C-level passes)
    instead of per-element Python loops with slice min/max.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    lower = minimum_filter1d(s2, size=2 * r, mode="nearest")
    upper = maximum_filter1d(s2, size=2 * r, mode="nearest")
    above = np.maximum(s1 - upper, 0.0)
    below = np.maximum(lower - s1, 0.0)
    return float(np.sqrt(np.sum(above**2 + below**2)))


def k_means_clust(data, num_clust, num_iter, w=5):
    import random

    centroids = random.sample(list(data), num_clust)
    for n in range(num_iter):
        print("\033[33m" + "iteration {}".format(n + 1) + "\033[0m")
        assignments = {}
        for ind, i in enumerate(data):
            min_dist = _INF
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust is not None:
                if closest_clust in assignments:
                    assignments[closest_clust].append(ind)
                else:
                    assignments[closest_clust] = [ind]

        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    return centroids


def DTWDistance_w(s1, s2, w):
    """Windowed Dynamic Time Warping distance using a 2D Python list.

    Only cells within a Sakoe-Chiba band of width *w* are evaluated,
    reducing complexity from O(n*m) to O(n*w).  Uses a 2D list instead
    of a dict to eliminate per-cell hashing overhead.
    """
    n, m = len(s1), len(s2)
    w = max(w, abs(n - m))
    DTW = [[_INF] * (m + 1) for _ in range(n + 1)]
    DTW[0][0] = 0.0

    for i in range(1, n + 1):
        s1_i = s1[i - 1]
        DTW_i = DTW[i]
        DTW_prev = DTW[i - 1]
        for j in range(max(1, i - w), min(m + 1, i + w)):
            cost = (s1_i - s2[j - 1]) ** 2
            DTW_i[j] = cost + min(DTW_prev[j], DTW_i[j - 1], DTW_prev[j - 1])

    return sqrt(DTW[n][m])
