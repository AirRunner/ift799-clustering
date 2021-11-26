import sys
from tqdm import tqdm

from scripts.k_means import k_means_auto_clusters
from scripts.fc_means import fc_means_auto_clusters
from scripts.export_results import identify_outliers, plot_k_values, plot_max_cluster_size
from scripts.processing import create_windows, prepare_data


def main():
    algo = sys.argv[1]
    window_size = int(sys.argv[2])
    window_shift = int(sys.argv[3])

    if algo == "km":
        clustering = k_means_auto_clusters
    elif algo == "fcm":
        clustering = fc_means_auto_clusters
    else:
        raise NotImplementedError(f"The algorithm {algo} is not supported")

    df = prepare_data()
    dates, windows = create_windows(df, window_size, window_shift)

    k_values = []
    max_cluster_size = []

    for win in tqdm(windows):
        _, y, k, _ = clustering(win)
        k_values.append(k)

        max_size = 0
        for c in range(k):
            size = sum(y == c)
            if size > max_size:
                max_size = size
        max_cluster_size.append(max_size)

    upper_band, lower_band = identify_outliers(dates, max_cluster_size, algo, window_size, window_shift)

    plot_k_values(dates, k_values, algo, window_size, window_shift)
    plot_max_cluster_size(dates, max_cluster_size, algo, window_size, window_shift, upper_band, lower_band)


def help():
    print("\n Usage: python main.py algo window_size window_shift\
           \n\t algo : 'km' for k-means or 'fcm' for fc-means\
           \n\t window_size : 21 for a month, 63 for 3 months\
           \n\t window_shift : equal to window_size if no superposition, else smaller\
           \n\n\t ex : python main.py fcm 63 21")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main()
    else:
        help()
