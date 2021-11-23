import numpy as np

def dtw_dist(s, t, w=1):
    n, m = len(s), len(t)
    w = np.max([w, abs(n-m)])
    
    dtw = np.ones((n+1, m+1)) * np.inf
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w]) + 1):
            dtw[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w]) + 1):
            cost = abs(s[i-1] - t[j-1])
            dtw[i, j] = cost + np.min([
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            ])
    
    return dtw

if __name__ == "__main__":
    from fastdtw import fastdtw
    from dtw import dtw
    
    # A noisy sine wave as query
    idx = np.linspace(0, 6.28, num=3)
    query = np.sin(idx) + np.random.uniform(size=3) / 10.0
    # A cosine is for template; sin and cos are offset by 25 samples
    template = np.cos(idx)
    
    print(dtw_dist(query, template))
    
    print(fastdtw(query, template))
    
    alignment = dtw(query, template, keep_internals=True)
    alignment.plot(type="threeway")
