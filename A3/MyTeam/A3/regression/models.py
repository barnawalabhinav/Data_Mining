import os
import math
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader

from encoder import *


class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class Random_Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def predict(self, data, low=-2, high=5) -> torch.Tensor:
        return torch.rand(data.num_graphs) * (high - low) + low


class Linear_Regressor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear_Regressor, self).__init__()
        self.regressor = torch.nn.Linear(in_channels, out_channels)

        torch.nn.init.xavier_normal_(self.regressor.weight)
        # torch.nn.init.constant_(self.regressor.bias, 0.1)

    def forward(self, data) -> torch.Tensor:
        x = global_add_pool(data.x, data.batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.regressor(x).squeeze(dim=1)
        return x

    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        return self.forward(data)


class Custom_Regressor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(Custom_Regressor, self).__init__()
        torch.manual_seed(12345)
        hidden_channels_01 = 32
        hidden_channels_02 = 64
        hidden_channels_03 = 128
        hidden_channels_1 = 256
        hidden_channels_2 = 128
        hidden_channels_3 = 64
        hidden_channels_4 = 32

        self.conv1 = GINEConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels_01), torch.nn.LeakyReLU(
                            ), torch.nn.Linear(hidden_channels_01, hidden_channels_02)), edge_dim=edge_dim)
        self.conv2 = GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels_02, hidden_channels_03), torch.nn.LeakyReLU(
                            ), torch.nn.Linear(hidden_channels_03, hidden_channels_1)), edge_dim=edge_dim)

        # self.conv1 = GATConv(in_channels, hidden_channels_1)
        # self.conv2 = GATConv(hidden_channels_1, hidden_channels_2)

        self.fc0 = torch.nn.Linear(hidden_channels_1, hidden_channels_2)
        self.fc1 = torch.nn.Linear(hidden_channels_2, hidden_channels_3)
        self.fc2 = torch.nn.Linear(hidden_channels_3, hidden_channels_4)
        self.regressor = torch.nn.Linear(hidden_channels_4, out_channels)

        # torch.nn.init.xavier_normal_(self.regressor.weight)
        # torch.nn.init.constant_(self.regressor.bias, 0.1)

    def forward(self, data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = global_add_pool(x, data.batch)
        x = self.fc0(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.regressor(x).squeeze(dim=1)
        return x
    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        return self.forward(data)


def load_data(data_dir, batch_size=-1, num_edims=5, num_ndims=5, load_labels=True, shuffle=True):
    NUM_EDIMS = num_edims
    NUM_NDIMS = num_ndims
    DATA_DIR = data_dir
    BATCH_SIZE = batch_size

    # Initialize the Encoders
    node_encoder = NodeEncoder(NUM_NDIMS)
    edge_encoder = EdgeEncoder(NUM_EDIMS)

    # Read the csv files into dataframes
    if load_labels:
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
    data_list = []
    if BATCH_SIZE == -1:
        BATCH_SIZE = NUM_GRAPHS

    nodes_done = 0
    edges_done = 0
    for i in range(NUM_GRAPHS):
        num_nodes = num_nodes_df.iloc[i, 0]
        num_edges = num_edges_df.iloc[i, 0]
        nodes_done += num_nodes
        edges_done += num_edges

        if load_labels:
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

        # ******************** ENCODEING FEATURES ********************
        # node_features = node_encoder(node_features.long())
        # edge_features = edge_encoder(edge_features.long())
        # ************************************************************

        data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)

        # -----------------------------------------------------
        #                   WITH ENCODING
        # -----------------------------------------------------
        # node_features = torch.tensor(node_features_df.iloc[nodes_done-num_nodes:nodes_done, :].values, dtype=torch.float)
        # edges = torch.tensor(edges_df.iloc[edges_done-num_edges:edges_done, :].values)
        # edge_features = torch.tensor(edge_features_df.iloc[edges_done-num_edges:edges_done, :].values, dtype=torch.float)

        # encoded_edge_features = [torch.empty(3 * NUM_EDIMS) for _ in range(num_edges)]

        # # Initialize a list to hold the combined edge features for each node
        # combined_edge_features = [torch.empty((0, edge_features.shape[1])) for _ in range(num_nodes)]

        # # Combine edge features for each node
        # for idx, edge in enumerate(edges):
        #     combined_edge_features[edge[0]] = torch.cat((combined_edge_features[edge[0]], edge_features[idx].unsqueeze(0)))
        #     combined_edge_features[edge[1]] = torch.cat((combined_edge_features[edge[1]], edge_features[idx].unsqueeze(0)))

        #     encoded_edge_features[idx] = edge_encoder(edge_features[idx].long().reshape(1, -1)).view(-1).detach()

        # encoded_node_features = [torch.empty(9 * NUM_NDIMS + 3 * NUM_EDIMS) for _ in range(num_nodes)]
        # # Encode the combined edge features and append to node features
        # for node in range(num_nodes):
        #     edge_encoding = edge_encoder(combined_edge_features[node].long())
        #     node_encoding = node_encoder(node_features[node].long().reshape(1, -1)).view(-1)
        #     encoded_node_features[node] = torch.cat((node_encoding, edge_encoding.mean(dim = 0))).detach()

        # data = Data(x=encoded_node_features, edge_index=edges.t().contiguous(), edge_attr=encoded_edge_features)

        if load_labels:
            data.y = torch.tensor(
                [graph_labels_df.iloc[i, 0]], dtype=torch.float)  # Set the label
        data_list.append(data)

    def custom_collate(batch):
        batch = Batch.from_data_list(batch)
        graph_num_nodes = [graph.num_nodes for graph in batch]
        batch.batch = torch.cat([torch.full((num_nodes,), graph_idx)
                                for graph_idx, num_nodes in enumerate(graph_num_nodes)])
        return batch

    dataset = CustomDataset(data_list)
    dataloader = DataLoader(data_list, batch_size=BATCH_SIZE,
                            shuffle=shuffle, collate_fn=custom_collate)
    '''
    DataLoader is a collection of batches. Each batch is a Batch object, which is a collection of <BATCH_SIZE> graphs as follows:

    for data in dataloader:
        graph_idx = 0  # The index of the graph you want to access: 0 to BATCH_SIZE-1
        subgraph = data[graph_idx]
    '''
    return dataset, dataloader
