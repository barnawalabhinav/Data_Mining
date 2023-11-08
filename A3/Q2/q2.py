import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import pandas as pd

DATA_DIR = "../dataset/dataset_1/train/"

# Read the csv files into dataframes
edges_df = pd.read_csv(DATA_DIR + 'edges.csv.gz', compression='gzip')
num_nodes_df = pd.read_csv(DATA_DIR + 'num_nodes.csv.gz', compression='gzip')
num_edges_df = pd.read_csv(DATA_DIR + 'num_edges.csv.gz', compression='gzip')
graph_labels_df = pd.read_csv(DATA_DIR + 'graph_labels.csv.gz', compression='gzip')
node_features_df = pd.read_csv(DATA_DIR + 'node_features.csv.gz', compression='gzip')
edge_features_df = pd.read_csv(DATA_DIR + 'edge_features.csv.gz', compression='gzip')

NUM_GRAPHS = len(graph_labels_df)
data_list = []

nodes_done = 0
edges_done = 0
for i in range(NUM_GRAPHS):
    num_nodes = num_nodes_df.iloc[i, 0]
    num_edges = num_edges_df.iloc[i, 0]
    node_features = torch.tensor(node_features_df.iloc[nodes_done:nodes_done+num_nodes, :].values)
    edges = torch.tensor(edges_df.iloc[edges_done:edges_done+num_edges, :].values)
    edge_features = torch.tensor(edge_features_df.iloc[edges_done:edges_done+num_edges, :].values)
    nodes_done += num_nodes
    edges_done += num_edges

    data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)
    data.y = torch.tensor([graph_labels_df.iloc[i, 0]], dtype=torch.double)  # Set the label
    data_list.append(data)


class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


dataset = CustomDataset(data_list)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
