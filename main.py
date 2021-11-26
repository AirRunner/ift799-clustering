import sys
from tqdm import tqdm

from scripts.auto_clusters import auto_clusters, auto_clusters_noreps
from scripts.algorithms import k_means, fc_means, k_meanoid
from scripts.plot import plot_k_values, plot_max_cluster_size
from scripts.processing import prepare_data, create_windows


def main():
    algo = sys.argv[1]
    window_size = int(sys.argv[2])
    window_shift = int(sys.argv[3])

    years = max_vars = None
    if len(sys.argv) > 4:
        years = (int(sys.argv[4]), int(sys.argv[5]))
        max_vars = int(sys.argv[6])

    auto_clustering = auto_clusters
    if algo == "km":
        clustering = k_means
    elif algo == "fcm":
        clustering = fc_means
    elif algo == "dtw":
        clustering = k_meanoid
        auto_clustering = auto_clusters_noreps
    else:
        raise NotImplementedError(f"The algorithm {algo} is not supported")

    df = prepare_data(years, max_vars)
    dates, windows = create_windows(df, window_size, window_shift)

    k_values = []
    max_cluster_size = []

    for win in tqdm(windows):
        y, k = auto_clustering(win, clustering)[:2]
        k_values.append(k)
        
        max_size = 0
        for c in range(k):
            size = sum(y == c)
            if size > max_size:
                max_size = size
        max_cluster_size.append(max_size)

    plot_k_values(dates, k_values, algo, window_size, window_shift, years, max_vars)
    plot_max_cluster_size(dates, max_cluster_size, algo, window_size, window_shift, years, max_vars)

def help():
    print("\n Usage: python main.py algo window_size window_shift [min_year, max_year, max_vars]\
           \n\t algo : 'km' for k-means of 'fcm' for fc-means\
           \n\t window_size : 21 for a month, 63 for 3 months\
           \n\t window_shift : equal to window_size if no superposition, else smaller\
           \n\n\t ex : python main.py fcm 63 21")

if __name__ == "__main__":
    if len(sys.argv) == 4 or len(sys.argv) == 7:
        main()
    else:
        help()
