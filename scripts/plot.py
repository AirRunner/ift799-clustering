from matplotlib import pyplot as plt


def plot_max_cluster_size(dates, max_cluster_size, algo, window_size, window_shift):
    plt.figure(figsize=(12,8))

    plt.title("Max cluster size in function of time")
    plt.xlabel("Date")
    plt.ylabel("Max cluster size")
    plt.plot(dates, max_cluster_size)
    plt.savefig(f"images/{algo}_max_{window_size}_{window_shift}.png")
    plt.show()

def plot_k_values(dates, k_values, algo, window_size, window_shift):
    plt.figure(figsize=(12,8))

    plt.title("Number of clusters in function of time")
    plt.xlabel("Date")
    plt.ylabel("k")
    plt.plot(dates, k_values)
    plt.savefig(f"images/{algo}_k_{window_size}_{window_shift}.png")
    plt.show()
