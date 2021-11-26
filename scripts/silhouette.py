import numpy as np

from scripts.distances import dist_euclide, dtw_to_clust



# Validity index for nb_clusters
def silhouette(X, y):
    s = []
    clusters = np.unique(y)

    for p in range(len(X)):
        i = y[p]
        Ci = X[y == i]

        if len(Ci) == 1:
            s.append(0)
            continue

        a = sum(dist_euclide(X[p], Ci)) / (len(Ci) - 1)

        bj = []
        for j in clusters:
            if j != i:
                Cj = X[y == j]
                bj.append(sum(dist_euclide(X[p], Cj)) / len(Cj))
        b = min(bj)

        s.append((b - a) / max(a, b))
    
    return np.asarray(s).mean()


def silhouette_noreps(X, y, dist_mat):
    s = []
    
    for p in range(len(X)):
        i = y[p]
        c_idxs = np.where(y == i)[0]
        
        if len(c_idxs) <= 1:
            s.append(0)
            continue

        a = dtw_to_clust(p, c_idxs, dist_mat)

        bj = []
        for j in np.unique(y):
            if j != i:
                c_idxs = np.where(y == j)[0]
                bj.append(dtw_to_clust(p, c_idxs, dist_mat))
        b = min(bj)
        
        s.append((b - a) / max(a, b))
    
    return np.asarray(s).mean()
