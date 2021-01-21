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
    df = df.select_dtypes('number')
    return pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)

