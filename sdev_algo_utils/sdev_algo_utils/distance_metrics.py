from math import sqrt
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf


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
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


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
    """
    from sklearn.metrics.pairwise import pairwise_distances
    p = pairwise_distances(test, metric = DTWDistance)
    """
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float("inf")
    for i in range(len(s2)):
        DTW[(-1, i)] = float("inf")
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(
                DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
            )

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


def euclid_dist(t1, t2):
    return sqrt(sum((t1 - t2) ** 2))


def LB_Keogh(s1, s2, r):
    """
    https://nbviewer.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb

    """
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0) : (ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0) : (ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return sqrt(LB_sum)


def k_means_clust(data, num_clust, num_iter, w=5):
    import random

    centroids = random.sample(data, num_clust)
    counter = 0
    for n in range(num_iter):
        counter += 1
        print(counter)
        assignments = {}
        # assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float("inf")
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    return centroids


def DTWDistance_w(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float("inf")
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(
                DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
            )

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])
