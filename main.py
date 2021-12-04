import sys
from tqdm import tqdm

from scripts.processing import create_windows, prepare_data
from scripts.auto_clusters import auto_clusters, auto_clusters_noreps
from scripts.algorithms import k_means, fc_means, k_meanoid
from scripts.export_results import ExportResults
from scripts.rand import rand


def main():
    algo = sys.argv[1]
    window_size = int(sys.argv[2])
    window_shift = int(sys.argv[3])

    years = nb_vars = None
    if len(sys.argv) > 4:
        years = (int(sys.argv[4]), int(sys.argv[5]))
        nb_vars = int(sys.argv[6])

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

    df = prepare_data(years, nb_vars)
    dates, windows = create_windows(df, window_size, window_shift)

    k_values = []
    max_cluster_size = []
    rand_scores = []

    y_old = None
    for win in tqdm(windows):
        y, k = auto_clustering(win, clustering)[:2]
        k_values.append(k)

        max_size = 0
        for c in range(k):
            size = sum(y == c)
            if size > max_size:
                max_size = size
        max_cluster_size.append(max_size)

        if y_old is not None:
            rand_scores.append(rand(y, y_old))
        y_old = y

    export_results = ExportResults(dates, k_values, max_cluster_size, rand_scores, algo, window_size, window_shift, years, nb_vars)
    export_results.plot_k_values()
    export_results.plot_max_cluster_size()
    export_results.plot_rand_values()


def help():
    print("\n Usage: python main.py algo window_size window_shift [min_year max_year nb_vars]\
           \n\t algo : 'km' for k-means or 'fcm' for fc-means or 'dtw' for time series k-meanoid\
           \n\t window_size : 21 for a month, 63 for 3 months\
           \n\t window_shift : equal to window_size if no superposition, else smaller\
           \n\t min_year, max_year : restrict the time interval to work with\
           \n\t nb_vars : restrict the number of variables to use (100 <= nb_vars <= 287)\
           \n\n\t ex : python main.py fcm 63 21")


if __name__ == "__main__":
    if len(sys.argv) in (4, 7):
        main()
    else:
        help()
