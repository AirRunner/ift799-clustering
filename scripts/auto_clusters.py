import numpy as np
from fastdtw import fastdtw

from scripts.silhouette import silhouette, silhouette_noreps
from scripts.algorithms import dist_matrix


def auto_clusters(X, algo, min_k=2, max_k=10):
    s_values = []
    best_s = -1

    np.random.shuffle(X)

    for k in range(min_k, max_k + 1):
        centers, y = algo(X, k)

        s = silhouette(X, y)
        s_values.append(s)
        
        if s > best_s:
            best_centers, best_y, best_k = centers, y, k
            best_s = s

    return best_y, best_k, best_centers, s_values


def auto_clusters_noreps(X, algo, min_k=2, max_k=10):
    s_values = []
    best_s = -1

    dist_mat = dist_matrix(X, dist=fastdtw)

    np.random.shuffle(X)
    for k in range(min_k, max_k + 1):
        y = algo(X, k, dist_mat=dist_mat)

        s = silhouette_noreps(X, y, dist_mat)
        s_values.append(s)
        
        if s > best_s:
            best_y, best_k = y, k
            best_s = s

    return best_y, best_k, s_values
