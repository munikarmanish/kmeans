#!/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np

from kmeans import BisectingKMeansClusterer, KMeansClusterer


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Run the k-means clustering algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--datafile', type=str, default='data.txt',
        help='The data file containing m x n matrix')
    parser.add_argument(
        '-k', '--k', type=int, default=10,
        help='Number of clusters')
    parser.add_argument(
        '-g', '--min-gain', type=float, default=0.1,
        help='Minimum gain to keep iterating')
    parser.add_argument(
        '-t', '--max-iter', type=int, default=100,
        help='Maximum number of iterations per epoch')
    parser.add_argument(
        '-e', '--epoch', type=int, default=10,
        help='Number of random starts, for global optimum')
    parser.add_argument(
        '-v', '--verbose', default=0, action='store_true',
        help='Show verbose info')
    args = parser.parse_args()

    # prepare data
    data = np.loadtxt(args.datafile)

    # initialize clusterer
    c = KMeansClusterer(
        data, k=args.k, max_iter=args.max_iter, max_epoch=args.epoch,
        verbose=args.verbose)

    # the result
    plt.figure(1)
    # plot the clusters in different colors
    for i in range(c.k):
        plt.plot(c.C[i][:, 0], c.C[i][:, 1], 'x')
    # plot the centroids in black squares
    plt.plot(c.u[:, 0], c.u[:, 1], 'ks')
    plt.show()


if __name__ == '__main__':
    main()
