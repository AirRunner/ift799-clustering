import numpy as np


def dist_euclide(x, y):
    return np.linalg.norm(x - y, axis=1)


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
