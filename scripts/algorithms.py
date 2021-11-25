import numpy as np
from random import choice
from fcmeans import FCM

from scripts.distances import dist_euclide, dtw_score, dist_to_clust


def k_means(X, k, threshold=1e-3):
    # Initialize random cluster centers
    centers = X[:k]
    
    while True:
        # Assign points to clusters
        y = []
        for p in X:
            y.append(dist_euclide(p, centers).argmin())
        y = np.asarray(y)

        # Recalculate centers
        new_centers = centers.copy()
        for c in range(k):
            mask = y == c
            new_centers[c] = X[mask].mean(axis=0)

        # Stop if centers are stable
        if (dist_euclide(centers, new_centers) < threshold).all():
            return new_centers, y

        centers = new_centers


def fc_means(X, k):
    fcm = FCM(n_clusters=k)
    fcm.fit(X)
    
    centers = fcm.centers
    y = fcm.predict(X)

    return centers, y


def k_meanoid(X, k, dist=dtw_score):
    rem = lambda A, i: np.delete(A, i, axis=0)
    
    # Assign clusters randomly
    N = X.shape[0]
    y = np.asarray([choice(range(k)) for _ in range(N)])
    
    # Run algorithm
    old_y = y.copy()
    while True:
        for i in range(N):
            curr_clust = y[i]
            best_clust = curr_clust
            best_score = np.inf
            
            # Assign to the point the closest cluster
            for c in range(k):
                cluster = rem(X, i)[rem(y, i) == c]
                if (score := dist_to_clust(X[i], cluster, dist)) < best_score:
                    best_score = score
                    best_clust = c
                    
            y[i] = best_clust
            
        if (y == old_y).all():
            break
        old_y = y.copy()
    
    return y
