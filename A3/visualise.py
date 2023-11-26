import os
import sys
import math
import torch
import argparse
import matplotlib
import pandas as pd
import networkx as nx
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def load_graph(data_dir):
    DATA_DIR = data_dir
    graph_list = []
    data_list = []

    # Read the csv files into dataframes
    graph_labels_df = pd.read_csv(os.path.join(
        DATA_DIR, 'graph_labels.csv.gz'), compression='gzip', header=None)
    edges_df = pd.read_csv(os.path.join(
        DATA_DIR, 'edges.csv.gz'), compression='gzip', header=None)
    num_nodes_df = pd.read_csv(os.path.join(
        DATA_DIR, 'num_nodes.csv.gz'), compression='gzip', header=None)
    num_edges_df = pd.read_csv(os.path.join(
        DATA_DIR, 'num_edges.csv.gz'), compression='gzip', header=None)
    node_features_df = pd.read_csv(os.path.join(
        DATA_DIR, 'node_features.csv.gz'), compression='gzip', header=None)
    edge_features_df = pd.read_csv(os.path.join(
        DATA_DIR, 'edge_features.csv.gz'), compression='gzip', header=None)

    NUM_GRAPHS = len(num_nodes_df)

    nodes_done = 0
    edges_done = 0
    for i in range(NUM_GRAPHS):
        num_nodes = num_nodes_df.iloc[i, 0]
        num_edges = num_edges_df.iloc[i, 0]
        nodes_done += num_nodes
        edges_done += num_edges

        if math.isnan(graph_labels_df.iloc[i, 0]):
            continue

        # -----------------------------------------------------
        #                 WITHOUT ENCODING
        # -----------------------------------------------------
        node_features = torch.tensor(
            node_features_df.iloc[nodes_done-num_nodes:nodes_done, :].values, dtype=torch.float)
        edges = torch.tensor(
            edges_df.iloc[edges_done-num_edges:edges_done, :].values)
        edge_features = torch.tensor(
            edge_features_df.iloc[edges_done-num_edges:edges_done, :].values, dtype=torch.float)

        data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)
        data.y = torch.tensor([graph_labels_df.iloc[i, 0]], dtype=torch.float)  # Set the label
        graph_list.append([to_networkx(data, to_undirected=True), data.y])
        data_list.append(data)

    '''
    DataLoader is a collection of batches. Each batch is a Batch object, which is a collection of <BATCH_SIZE> graphs as follows:

    for data in dataloader:
        graph_idx = 0  # The index of the graph you want to access: 0 to BATCH_SIZE-1
        subgraph = data[graph_idx]
    '''
    return graph_list, data_list


def main():
    parser = argparse.ArgumentParser(description="Visualising Graph")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    
    # graph = nx.Graph()

    # visualise(args.dataset_path, args.save_path, args.model_path)

    graph_list = load_graph(args.dataset_path)
    cnt_1, cnt_0 = 0, 0
    for graph in graph_list:
        dir_path = os.path.join(args.save_path, str(int(graph[1].item())))
        if int(graph[1].item()) == 1:
            save_path = os.path.join(dir_path, str(cnt_1)+'.png')
            cnt_1 += 1
        else:
            save_path = os.path.join(dir_path, str(cnt_0)+'.png')
            cnt_0 += 1
        nx.draw(graph[0], with_labels = True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    main()