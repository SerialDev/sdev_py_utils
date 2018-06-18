
from collections import namedtuple
from collections import deque
import random
import numpy as np
import heapq


class NDPoint(object):
    """
    *  NDPoint 
    * -----------{returns}------------
    * a point in n-dimensional space . . . 
    """
    def __init__(self, x, idx=None):
        self.x = np.array(x)
        self.idx = idx
        def __repr__(self):
            return "NDPoint(idx={idx}, x={x})".format(idx = self.idx, x = self.x)

class VPTree(object):
    """
    * Function: Efficient data structure to perform nearest-Neighbor search 
    * -----------{returns}------------
    *  . . . Vantage point tree
    """

    def __init__(self, points, dist_fn=None):
        self.left = None
        self.right = None
        self.mu = None
        self.dist_fn = dist_fn if dist_fn is not None else l2

        # Choose a better vantage point selection process
        self.vp = points.pop(random.randrange(len(points)))

        if len(points) < 1 :
            return

        # Choose division boundary at median of distances
        distances = [self.dist_fn(self.vp, p) for p in points]
        self.mu = np.median(distances)

        left_points = [] # all points inside mu radious
        right_points = [] # all points outside mu radious
        for i, p in enumerate(points):
            d = distances[i]
            if d >= self.mu:
                right_points.append(p)
            else:
                left_points.append(p)

        if len(left_points) > 0:
            self.left = VPTree(points=left_points, dist_fn=self.dist_fn)
        if len(right_points) > 0:
            self.right = VPTree(points = right_points, dist_fn=self.dist_fn)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

class PriorityQueue(object):
    def __init__(self, size = None):
        self.queue = []
        self.queue.sort()
        self.size = size

    def push(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()
        if self.size is not None and len(self.queue) > self.size:
            self.queue.pop()


##-------------{Distance functions}-----

def l2(p1, p2):
    return np.sqrt(np.sum(np.power(p2.x - p1.x, 2)))



##==========================={Operations}=======================================

def get_nearest_neighbors(tree, q, k=1):
    """
    * Function: find k nearest neighbor(s) of q 
    * -----------{returns}------------
    * k nearest neighbors . . . 
    * -----------{params}-------------
      :param tree: vp-tree
      :param q: a query point
      :param k: number of nearest neighboors
    """

    # buffer for nearest neighbors
    neighbors = PriorityQueue(k)

    # list of nodes
    visit_stack = deque([tree])

    #disance of n-nearest neighbors so far
    tau = np.inf

    while len(visit_stack) > 0:
        node = visit_stack.popleft()
        if node is None:
            continue

        d = tree.dist_fn(q, node.vp)
        if d < tau:
            neighbors.push(d, node.vp)
            tau, _ = neighbors.queue[-1]

        if node.is_leaf():
            continue

        if d < node.mu:
            if d < node.mu + tau:
                visit_stack.append(node.left)
            if d >= node.mu - tau:
                visit_stack.append(node.right)

        else:
            if d >= node.mu - tau:
                visit_stack.append(node.right)
            if d < node.mu + tau:
                visit_stack.append(node.left)
    return neighbors.queue


def get_all_in_range(tree, q, tau):
    """
    * Function: Find all points within a given radious of point q 
    * -----------{returns}------------
    * points . . . 
    * -----------{params}=------------
    * tree: vp-tree
    * q: a query point
    * tau: the maximum distance from point q
    """

    # buffer for nearest neighbors
    neighbors = []

    visit_stack = deque([tree])

    while len(visit_stack) > 0:
        node = visit_stack.popleft()
        if node is None:
            continue

        d = tree.dist_fn(q, node.vp)
        if d < tau:
            neighbors.append((d, node.vp))
        if node.is_leaf():
            continue

        if d < node.mu:
            if d < node.mu + tau:
                visit_stack.append(node.left)
            if d >= node.mu - tau:
                visit_stack.append(node.right)
        else:
            if d >=  node.mu -tau:
                visit_stack.append(node.right)
            if d < node.mu + tau:
                visit_stack.append(node.left)
    return neighbors

#     
#   if __name__ == '__main__':  
#   X = np.random.uniform(0, 100000, size=10000)  
#   Y = np.random.uniform(0, 100000, size=10000)  
#   points = [NDPoint(x,i) for i, x in enumerate(zip(X,Y))]  
#   tree = VPTree(points)  
#   q = NDPoint([300,300])  
#   neighbors = get_nearest_neighbors(tree, q, 5)  
#     
#   print "query:"  
#   print "\t", q  
#   print "nearest neighbors: "  
#   for d, n in neighbors:  
#   print "\t", n  
#






    
