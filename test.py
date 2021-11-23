import sys
import numpy as np
from matplotlib import pyplot as plt
from scripts.k_means import k_means_auto_clusters
from scripts.fc_means import fc_means_auto_clusters


def main():
    n_samples = 1000
    min_k = 2
    max_k = 10

    X = np.concatenate((
        np.random.normal((-2, -2), size=(n_samples, 2)),
        np.random.normal((2, 2), size=(n_samples, 2)),
        np.random.normal((-4, 4), size=(n_samples, 2))
    ))

    if sys.argv[1] == "km":
        centers, y, k, s_values = k_means_auto_clusters(X, min_k, max_k)
    elif sys.argv[1] == "fcm":
        centers, y, k, s_values = fc_means_auto_clusters(X, min_k, max_k)
    else:
        raise NotImplementedError(f"The algorithm {sys.argv[1]} is not supported")

    print("Best k:", k)

    plt.title("Silhouette value in function of k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette value")
    plt.plot(range(2, 11), s_values)
    plt.show()

    plt.title(f"Result of {'k-means' if sys.argv[1] == 'km' else 'fc-means'} clustering")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X[:,0], X[:,1], alpha=0.2, c=y)
    plt.scatter(centers[:,0], centers[:,1], marker="+", s=500, c='r')
    plt.show()

def help():
    print("\n Usage: python test.py algo\
           \n\t algo : 'km' for k-means of 'fcm' for fc-means")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        help()