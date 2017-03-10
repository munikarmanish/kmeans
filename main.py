#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeansClusterer, BisectingKMeansClusterer


def main():
    # prepare data
    data = np.loadtxt('D.txt')

    # initialize clusterer
    #c = KMeansClusterer(data, k=k, max_iter=100, max_epoch=10)
    c = BisectingKMeansClusterer(data, max_k=10, min_gain=0.1)

    # plot result
    plt.figure(1)
    for i in range(c.k):
        plt.plot(c.C[i][:,0], c.C[i][:,1], 'x')
    plt.plot(c.u[:,0], c.u[:,1], 'ks')
    plt.show()


if __name__ == '__main__':
    main()
