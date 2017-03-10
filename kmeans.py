
import numpy as np


def centroid(data):
    return np.mean(data, 0)


def sse(data):
    u = centroid(data)
    return np.sum(np.linalg.norm(data - u, 2, 1))


class KMeansClusterer:

    def __init__(self, data=None, k=2, min_gain=0.01, max_iter=100,
                 max_epoch=10, verbose=True):
        if data is not None:
            self.cluster(data, k, min_gain, max_iter, max_epoch)

    def cluster(self, data, k=2, min_gain=0.01, max_iter=100, max_epoch=10,
                verbose=True):
        # Pre-process
        self.data = np.matrix(data)
        self.k = k
        self.min_gain = min_gain

        # Perform multiple random init for global optimum
        min_sse = np.inf
        for epoch in range(max_epoch):
        
            # Randomly initialize k centroids
            indices = np.random.choice(len(data), k, replace=False)
            u = self.data[indices,:]

            # Loop
            t = 0
            old_sse = np.inf
            while True:
                t += 1

                # Cluster assignment
                C = [None] * k
                for x in self.data:
                    j = np.argmin(np.linalg.norm(x-u, 2, 1))
                    C[j] = x if C[j] is None else np.vstack((C[j], x))

                # Centroid update
                for j in range(k):
                    u[j] = np.mean(C[j], 0)

                # Loop termination condition
                if t >= max_iter:
                    break
                new_sse = np.sum([sse(C[j]) for j in range(k)])
                gain = old_sse - new_sse
                if verbose:
                    line = "Epoch {:2d} Iter {:2d}: SSE={:10.4f}, GAIN={:10.4f}"
                    print(line.format(epoch, t, new_sse, gain))
                if gain < self.min_gain:
                    if new_sse < min_sse:
                        min_sse, self.C, self.u = new_sse, C, u
                    break
                else:
                    old_sse = new_sse

            if verbose:
                print('') # blank line between every epoch

        return self


class BisectingKMeansClusterer:

    def __init__(self, data, max_k=10, min_gain=0.1, verbose=True):
        if data is not None:
            self.cluster(data, max_k, min_gain)

    def cluster(self, data, max_k=10, min_gain=0.1, verbose=True):
        self.kmeans = KMeansClusterer()
        self.C = [data,]
        self.k = len(self.C)
        self.u = np.reshape([centroid(self.C[i]) for i in range(self.k)],
                            (self.k,2))

        if verbose:
            print("k={}, SSE={}, GAIN={}".format(self.k, sse(data), '-'))

        while True:
            # pick a cluster to bisect
            sse_list = [sse(data) for data in self.C]
            old_sse = np.sum(sse_list)
            data = self.C.pop(np.argmax(sse_list))
            # bisect it
            self.kmeans.cluster(data, k=2, verbose=False)
            # add bisected clusters to our list
            self.C.append(self.kmeans.C[0])
            self.C.append(self.kmeans.C[1])
            self.k += 1
            self.u = np.reshape([centroid(self.C[i]) for i in range(self.k)],
                                (self.k,2))
            # check sse or k
            sse_list = [sse(data) for data in self.C]
            new_sse = np.sum(sse_list)
            gain = old_sse - new_sse
            if verbose:
                print("k={}, SSE={}, GAIN={}".format(self.k, new_sse, gain))
            if gain < min_gain or self.k >= max_k:
                break

        print(self.u.shape)

        return self

