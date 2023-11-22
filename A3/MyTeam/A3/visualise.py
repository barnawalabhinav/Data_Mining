import os
import argparse
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(graph, data_dir):
    DATA_DIR = data_dir

    edges_df = pd.read_csv(os.path.join(
        DATA_DIR, 'edges.csv.gz'), compression='gzip', header=None)
    num_edges_df = pd.read_csv(os.path.join(
        DATA_DIR, 'num_edges.csv.gz'), compression='gzip', header=None)
    NUM_GRAPHS = len(num_edges_df)

    edges_done = 0
    for i in range(NUM_GRAPHS):
        num_edges = num_edges_df.iloc[i, 0]
        edges_done += num_edges

        edges = edges_df.iloc[edges_done-num_edges:edges_done, :].values
        graph.add_edges_from(edges)
    
    return graph

def main():
    parser = argparse.ArgumentParser(description="Visualising Graph")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    
    graph = nx.Graph()

    graph = load_data(graph, args.dataset_path)

    nx.draw(graph, with_labels = True)
    plt.savefig(args.save_path, bbox_inches='tight')


if __name__ == "__main__":
    main()