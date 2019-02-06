"""
Utility / helper functions.
"""
from pprint import pprint
import numpy as np


def get_centroid(points):
    return np.mean(points, 0)


def get_sse(points):
    centroid = get_centroid(points)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)