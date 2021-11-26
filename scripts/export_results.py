from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


class ExportResults():
    def __init__(self, dates, k_values, max_cluster_size, algo, window_size, window_shift, years=None, max_vars=None):
        self.dates = dates
        self.k_values = k_values
        self.max_cluster_size = max_cluster_size

        if algo == "km":
            self.algo = "k-means"
        elif algo == "fcm":
            self.algo = "fc-means"
        elif algo == "dtw":
            self.algo = "k-meanoid"

        if years is not None and max_vars is not None:
            filter = f"{years[0]}-{years[1]}_{max_vars}"
        else:
            filter = "full"

        self.subtitle = f"({self.algo} avec fenêtres de {window_size} jours et déplacement de {window_shift} jours)"
        self.filename_params = f"{window_size}_{window_shift}_{filter}"

        self.init_output_folder_tree()
        self.upper_band, self.lower_band = self.identify_outliers()

    def init_output_folder_tree(self):
        for subfolder in ("images", "outliers"):
            Path(f"output/{subfolder}/k-means").mkdir(parents=True, exist_ok=True)
            Path(f"output/{subfolder}/fc-means").mkdir(parents=True, exist_ok=True)
            Path(f"output/{subfolder}/k-meanoid").mkdir(parents=True, exist_ok=True)

    def plot_max_cluster_size(self):
        plt.figure(figsize=(12, 8))

        plt.title("Taille du plus gros cluster en fonction du temps\n" + self.subtitle)
        plt.xlabel("Date")
        plt.ylabel("Taille du plus gros cluster")
        plt.plot(self.dates, self.max_cluster_size)
        plt.plot(self.dates, np.full(len(self.dates), self.upper_band), c='r')
        plt.plot(self.dates, np.full(len(self.dates), self.lower_band), c='r')

        plt.savefig(f"output/images/{self.algo}/max_{self.filename_params}.png")

    def plot_k_values(self):
        plt.figure(figsize=(12, 8))

        plt.title("Nombre de clusters en fonction du temps\n" + self.subtitle)
        plt.xlabel("Date")
        plt.ylabel("Nombre de clusters")
        plt.plot(self.dates, self.k_values)

        plt.savefig(f"output/images/{self.algo}/k_{self.filename_params}.png")

    def identify_outliers(self, alpha=1.5):
        q1 = np.quantile(self.max_cluster_size, 0.25)
        q3 = np.quantile(self.max_cluster_size, 0.75)
        iqr = q3 - q1
        upper_band = q3 + alpha * iqr
        lower_band = q1 - alpha * iqr

        d = np.asarray(self.dates)
        m = np.asarray(self.max_cluster_size)

        upper_outliers = d[m > upper_band].astype(str)
        lower_outliers = d[m < lower_band].astype(str)

        with open(f"output/outliers/{self.algo}/{self.filename_params}.txt", "w") as f:
            f.write("Upper outliers:\n")
            for date in upper_outliers:
                f.write(date[:10] + "\n")

            f.write("\nLower outliers:\n")
            for date in lower_outliers:
                f.write(date[:10] + "\n")

        return upper_band, lower_band
