# sh elbow_plot.sh CS1200385_generated_dataset_7D.dat 7 q3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

def cal_sqrd_dist(dataset, labels, centroids):
    sqrd_dist = 0
    for i, centroid in enumerate(centroids):
        cluster_points = dataset[labels == i]
        if len(cluster_points) == 0:
            continue
        cluster_dist = np.linalg.norm(cluster_points - centroid, axis=1)
        sqrd_dist += np.sum(cluster_dist**2)
    return sqrd_dist  

def elbow_plot(dataset, dimension, plot_name):
    sqrd_dists = []

    for k in range(1, 16):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(dataset)
        sqrd_dists.append(cal_sqrd_dist(dataset, kmeans.labels_, kmeans.cluster_centers_))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 16), sqrd_dists, marker='o', linestyle='-', color='b')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.grid(True)

    # Determine the optimal value of k using the elbow method
    final_k = None
    min_slope = np.inf
    min_k = None
    for i in range(1, len(sqrd_dists)):
        slope = abs(sqrd_dists[i] - sqrd_dists[i - 1])
        if slope < 1:         # Threshold is set to 1 currently
            final_k = i
            break
        if slope < min_slope:
            min_slope = slope
            min_k = i

    if final_k is None:
        final_k = min_k

    plt.axvline(x=final_k, color='r', linestyle='--', label=f'Optimal k = {final_k}')
    plt.legend()

    plt.savefig(f"{plot_name}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("dimension", type=int)
    parser.add_argument("plot_name")

    args = parser.parse_args()
    file_path = args.dataset
    dimension = args.dimension
    plot_name = args.plot_name

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        numbers = [float(x) for x in line.strip().split()]
        data.append(numbers)
    dataset = np.array(data)

    elbow_plot(dataset, dimension, plot_name)
