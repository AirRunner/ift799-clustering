import numpy as np
from scripts.silhouette import dist_euclide, silhouette


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

def k_means_auto_clusters(X, min_k=2, max_k=10):
    s_values = []
    best_s = -1

    np.random.shuffle(X)

    for k in range(min_k, max_k + 1):
        centers, y = k_means(X, k)

        s = silhouette(X, y)
        s_values.append(s)
        
        if s > best_s:
            best_centers, best_y, best_k = centers, y, k
            best_s = s

    return best_centers, best_y, best_k, s_values