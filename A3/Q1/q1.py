import numpy as np
import matplotlib.pyplot as plt


NUM_PTS = 1000000
PRECISION = 100000
DIMS = [1, 2, 4, 8, 16, 32, 64]


def analyze():
    L1_ratio = []
    L2_ratio = []
    L_inf_ratio = []

    for dim in DIMS:
        data_pts = np.unique(np.trunc(np.random.default_rng().uniform(
            0, 1, (NUM_PTS, dim)) * PRECISION) / PRECISION, axis=0)
        query_idx = np.random.choice(data_pts.shape[0], 100, replace=False)
        query_pts = data_pts[query_idx]

        # Using L1 distance
        sum_ratio = 0
        for pt in query_pts:
            dist = np.sum(np.abs(data_pts - pt), axis=1)
            sum_ratio += np.max(dist) / np.min(dist[dist != 0])
        L1_ratio.append(sum_ratio / len(query_pts))

        # Using L2 distance
        sum_ratio = 0
        for pt in query_pts:
            dist = np.sqrt(np.sum(np.square(data_pts - pt), axis=1))
            sum_ratio += np.max(dist) / np.min(dist[dist != 0])
        L2_ratio.append(sum_ratio / len(query_pts))

        # Using L_inf distance
        sum_ratio = 0
        for pt in query_pts:
            dist = np.max(np.abs(data_pts - pt), axis=1)
            sum_ratio += np.max(dist) / np.min(dist[dist != 0])
        L_inf_ratio.append(sum_ratio / len(query_pts))

        print(f"Finished analyzing dimension {dim}")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.plot(DIMS, L1_ratio, label="L1 dist", marker="o")
    ax.plot(DIMS, L2_ratio, label="L2 dist", marker="s")
    ax.plot(DIMS, L_inf_ratio, label="L_inf dist", marker="D")
    plt.xlabel("Dimension")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig("q1.png")


if __name__ == "__main__":
    analyze()
