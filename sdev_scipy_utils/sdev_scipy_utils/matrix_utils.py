import numpy as np
from scipy.linalg import expm


def rot_euler(v, xyz):
    """ Rotate vector v (or array of vectors) by the euler angles xyz """
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis * -theta)))
    return v
