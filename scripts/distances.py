import numpy as np


def dist_euclide(x, y):
    return np.linalg.norm(x - y, axis=1)

def dtw_score(s, t):
    n, m = len(s), len(t)
    
    dtw = np.ones((n+1, m+1)) * np.inf
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            dtw[i, j] = cost + np.min([
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            ])
    
    # dtw[i,j] is the distance between s[1:i] and t[1:j] with the best alignment
    return dtw[n, m]

def dist_matrix(X, dist):
    print(">> Distance matrix")

    N = X.shape[0]
    dist_mat = np.zeros((N, N))

    for i in range(N):
        if (i/N*100) % 10 == 0:
            print(f"{int(i/N*100)}%...", end=' ')
        
        for j in range(N):
            if i == j:
                continue
            dist_mat[i, j] = dist(X[i], X[j])[0]
    print()

    return dist_mat
