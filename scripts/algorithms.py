import numpy as np
from fcmeans import FCM
from fastdtw import fastdtw

from scripts.distances import dist_euclide, dist_matrix, dtw_to_clust


def k_means(X, k, threshold=1e-4, max_iter=100):
    # Initialize random cluster centers
    rand_indices = np.random.choice(X.shape[0], k, replace=False)
    centers = X[rand_indices]

    for _ in range(max_iter):
        # Assign points to clusters
        y = np.asarray([dist_euclide(p, centers).argmin() for p in X])

        # Recalculate centers
        new_centers = centers.copy()
        for c in range(k):
            new_centers[c] = X[y == c].mean(axis=0)

        # Stop if centers are stable
        if (dist_euclide(centers, new_centers) < threshold).all():
            break

        centers = new_centers

    return new_centers, y


def fc_means(X, k):
    fcm = FCM(n_clusters=k)
    fcm.fit(X)

    return fcm.centers, fcm.predict(X)


def k_meanoid(X, k, dist_mat=None, dist=fastdtw):
    # Assign clusters randomly
    N = X.shape[0]
    y = np.arange(N) % k

    # Compute distance matrix
    if dist_mat is None:
        dist_mat = dist_matrix(X, dist=dist)

    # Run algorithm
    old_y = y.copy()
    for _ in range(100):  # Max iterations
        for i in range(N):
            curr_clust = y[i]
            best_clust = curr_clust
            best_score = np.inf

            # Assign to the point the closest cluster
            for c in range(k):
                c_idxs = np.where(y == c)[0]
                c_idxs = c_idxs[c_idxs != i]
                score = dtw_to_clust(i, c_idxs, dist_mat)

                if score < best_score:
                    best_score = score
                    best_clust = c

            y[i] = best_clust

        if (y != old_y).sum() < 1e-2 * N:  # less than 1% changes
            break
        old_y = y.copy()

    return y
