import numpy as np
from matplotlib import pyplot as plt


def plot_max_cluster_size(dates, max_cluster_size, algo, window_size, window_shift, upper_band, lower_band):
    plt.figure(figsize=(12, 8))

    plt.title("Taille du plus gros cluster en fonction du temps\n"
              f"({'k-means' if algo == 'km' else 'FCM'} "
              f"avec fenêtres de {window_size} jours et déplacement de {window_shift} jours)")
    plt.xlabel("Date")
    plt.ylabel("Taille du plus gros cluster")
    plt.plot(dates, max_cluster_size)
    plt.plot(dates, np.full(len(dates), upper_band), c='r')
    plt.plot(dates, np.full(len(dates), lower_band), c='r')

    plt.savefig(f"output/images/{algo}_max_{window_size}_{window_shift}.png")


def plot_k_values(dates, k_values, algo, window_size, window_shift):
    plt.figure(figsize=(12, 8))

    plt.title("Nombre de clusters en fonction du temps\n"
              f"({'k-means' if algo == 'km' else 'FCM'} "
              f"avec fenêtres de {window_size} jours et déplacement de {window_shift} jours)")
    plt.xlabel("Date")
    plt.ylabel("Nombre de clusters")
    plt.plot(dates, k_values)

    plt.savefig(f"output/images/{algo}_k_{window_size}_{window_shift}.png")


def identify_outliers(dates, max_cluster_size, algo, window_size, window_shift):
    q1 = np.quantile(max_cluster_size, 0.25)
    q3 = np.quantile(max_cluster_size, 0.75)
    iqr = q3 - q1
    upper_band = q3 + 1.5 * iqr
    lower_band = q1 - 1.5 * iqr

    d = np.asarray(dates)
    m = np.asarray(max_cluster_size)

    upper_outliers = d[m > upper_band].astype(str)
    lower_outliers = d[m < lower_band].astype(str)

    with open(f"output/outliers/{algo}_{window_size}_{window_shift}.txt", "w") as f:
        f.write("Upper outliers:\n")
        for date in upper_outliers:
            f.write(date[:10] + "\n")

        f.write("\nLower outliers:\n")
        for date in lower_outliers:
            f.write(date[:10] + "\n")

    return upper_band, lower_band
