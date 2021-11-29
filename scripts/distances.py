import numpy as np


def dist_euclide(x, y):
    return np.linalg.norm(x - y, axis=1)


def dtw_score(s, t):  # unused
    n, m = len(s), len(t)

    dtw = np.ones((n + 1, m + 1)) * np.inf
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            dtw[i, j] = cost + np.min([
                dtw[i - 1, j],     # insertion
                dtw[i, j - 1],     # deletion
                dtw[i - 1, j - 1]  # match
            ])

    # dtw[i,j] is the distance between s[1:i] and t[1:j] with the best alignment
    return dtw[n, m], dtw


def dist_matrix(X, dist):
    N = X.shape[0]
    dist_mat = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i <= j:
                continue
            dtw_dist = dist(X[i], X[j])[0]
            dist_mat[i, j] = dist_mat[j, i] = dtw_dist

    return dist_mat


def dtw_to_clust(x_idx, c_idxs, dist_mat):
    if len(c_idxs) == 0:
        return np.inf
    return np.asarray([dist_mat[x_idx, ci] for ci in c_idxs]).mean()
