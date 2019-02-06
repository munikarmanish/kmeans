"""
Default values
"""

from .utils import euclidean

# number of clusters
K = 2

# Number of random initializations to try
EPOCHS = 10

# Max no. of cluster_assignment / centroid_update iterations
MAX_ITER = 100

# Distance function
DISTANCE = euclidean