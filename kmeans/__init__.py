"""Implementation of k-means clustering algorithm. 

These functions are designed to work with cartesian data points
"""

import numpy as np
from matplotlib import pyplot as plt


def convert_to_2d_array(points):
    """
    Converts `points` to a 2-D numpy array.
    """
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, -1)
    return points


def visualize_clusters(clusters):
    """
    Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
    """
    plt.figure()
    for cluster in clusters:
        points = convert_to_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:,0], points[:,1], 'o')
    plt.show()


def SSE(points):
    """
    Calculates the sum of squared errors for the given list of data points.

    Args:
        points: array-like
            Data points

    Returns:
        sse: float
            Sum of squared errors
    """
    points = convert_to_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)


def kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    """
    Clusters the list of points into `k` clusters using k-means clustering
    algorithm.

    Args:
        points: array-like
            Data points
        k: int
            Number of output clusters
        epochs: int
            Number of random starts (to find global optima)
        max_iter: int
            Max iteration per epoch
        verbose: bool
            Display progress on every iteration

    Returns:
        clusters: list with size = k
            List of clusters, where each cluster is a list of data points
    """
    points = convert_to_2d_array(points)
    assert len(points) >= k, "Number of data points can't be less than k"

    best_sse = np.inf
    for ep in range(epochs):
        # Randomly initialize k centroids
        np.random.shuffle(points)
        centroids = points[0:k, :]

        last_sse = np.inf
        for it in range(max_iter):
            # Cluster assignment
            clusters = [None] * k
            for p in points:
                index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
                if clusters[index] is None:
                    clusters[index] = np.expand_dims(p, 0)
                else:
                    clusters[index] = np.vstack((clusters[index], p))

            # Centroid update
            centroids = [np.mean(c, 0) for c in clusters]

            # SSE calculation
            sse = np.sum([SSE(c) for c in clusters])
            gain = last_sse - sse
            if verbose:
                print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                       f'SSE: {sse:12.4f}, Gain: {gain:12.4f}'))

            # Check for improvement
            if sse < best_sse:
                best_clusters, best_sse = clusters, sse

            # Epoch termination condition
            if np.isclose(gain, 0, atol=0.00001):
                break
            last_sse = sse

    return best_clusters


def bisecting_kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    """
    Clusters the list of points into `k` clusters using bisecting k-means
    clustering algorithm. Internally, it uses the standard k-means with k=2 in
    each iteration.

    Args:
        points: array-like
            Data points
        k: int
            Number of output clusters
        epochs: int
            Number of random starts (to find global optima)
        max_iter: int
            Max iteration per epoch
        verbose: bool
            Display progress on every iteration

    Returns:
        clusters: list with size = k
            List of clusters, where each cluster is a list of data points
    """
    points = convert_to_2d_array(points)
    clusters = [points]
    while len(clusters) < k:
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        cluster = clusters.pop(max_sse_i)
        two_clusters = kmeans(
            cluster, k=2, epochs=epochs, max_iter=max_iter, verbose=verbose)
        clusters.extend(two_clusters)
    return clusters
