import numpy as np
from tqdm import tqdm
from fcmeans import FCM
from silhouette import silhouette


def fc_means_auto_clusters(X, min_k=2, max_k=10):
    s_values = []
    best_s = -1

    np.random.shuffle(X)

    for k in tqdm(range(min_k, max_k + 1)):
        fcm = FCM(n_clusters=k)
        fcm.fit(X)
        
        centers = fcm.centers
        y = fcm.predict(X)
        
        s = silhouette(X, y)
        s_values.append(s)
        
        if s > best_s:
            best_centers, best_y, best_k = centers, y, k
            best_s = s

    return best_centers, best_y, best_k, s_values