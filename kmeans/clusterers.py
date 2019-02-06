"""
Definition of k-means clusterer classes.
"""
from pprint import pprint

import numpy as np

from . import defaults as D
from .utils import get_centroid, get_sse


class KMeans:
    def __init__(self, k=D.K):
        self.k = k

    def fit(self, points, epochs=D.EPOCHS, max_iter=D.MAX_ITER, verbose=False):
        # Convert data into numpy array
        points = np.array(points)
        if len(points.shape) == 1:
            points = np.expand_dims(points, -1)

        self.sse = np.inf
        for ep in range(epochs):
            # Randomly initialize k centroids
            np.random.shuffle(points)
            centroids = points[0:self.k, :]

            last_sse = np.inf
            for it in range(max_iter):
                # Cluster assignment
                clusters = [None] * self.k
                for p in points:
                    index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
                    if clusters[index] is None:
                        clusters[index] = np.expand_dims(p, 0)
                    else:
                        clusters[index] = np.vstack((clusters[index], p))

                # Centroid update
                centroids = [get_centroid(c) for c in clusters]

                # SSE calculation
                sse = np.sum([get_sse(c) for c in clusters])
                gain = last_sse - sse
                if verbose:
                    print(f'Epoch: {ep:3d}, Iter: {it:4d}, SSE: {sse:12.4f}, Gain: {gain:12.4f}')
                if sse < self.sse:
                    self.clusters, self.centroids = clusters, centroids
                    self.sse = sse
                if np.isclose(gain, 0, atol=0.0001):
                    break
                last_sse = sse
