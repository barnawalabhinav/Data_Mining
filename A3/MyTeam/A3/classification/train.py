import argparse
import os
import math
import numpy as np
import pandas as pd

import torch
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


class Random_Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        length = data.num_graphs
        return torch.randint(0, self.num_classes, (length,), dtype=torch.float)


class Logistic_Regressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Logistic_Regressor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

        torch.nn.init.xavier_normal_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias, 0.1)

    def forward(self, data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x).squeeze(dim=1)
        return x


class Custom_Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Custom_Classifier, self).__init__()
        self.conv1 = GINEConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(
        ), torch.nn.Linear(hidden_channels, hidden_channels)), edge_dim=3)
        self.conv2 = GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(
        ), torch.nn.Linear(hidden_channels, hidden_channels)), edge_dim=3)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

        torch.nn.init.xavier_normal_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias, 0.1)

    def forward(self, data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x).squeeze(dim=1)
        return x


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
        graph_labels_df = pd.read_csv(os.path.join(DATA_DIR, 'graph_labels.csv.gz'), compression='gzip', header=None)

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

        data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)
        
        # -----------------------------------------------------
        #                   WITH ENCODING
        # -----------------------------------------------------
        # node_features = torch.tensor(node_features_df.iloc[nodes_done-num_nodes:nodes_done, :].values, dtype=torch.float)
        # edges = torch.tensor(edges_df.iloc[edges_done-num_edges:edges_done, :].values)
        # edge_features = torch.tensor(edge_features_df.iloc[edges_done-num_edges:edges_done, :].values, dtype=torch.float)
        
        # encoded_edge_features = [torch.empty(3 * NUM_EDIMS) for _ in range(num_edges)]

        # # # Initialize a list to hold the combined edge features for each node
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
            data.y = torch.tensor([graph_labels_df.iloc[i, 0]], dtype=torch.float)  # Set the label
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


# model = Custom_Classifier(in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model_path, train_data_path, val_data_path, num_epochs=200, batch_size=32, model='custom'):
    dataset, dataloader = load_data(train_data_path, batch_size)
    _, val_dataloader = load_data(val_data_path, -1)
    val_data = next(iter(val_dataloader))

    if model == 'custom':
        MODEL = Custom_Classifier(
            in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
    else:
        MODEL = Logistic_Regressor(
            in_channels=dataset.num_features, hidden_channels=32, out_channels=1)

    # optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=1e-3)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_output = 0
        num_batches = 0
        num_graphs = 0
        for batch in dataloader:
            num_graphs += batch.num_graphs
            optimizer.zero_grad()
            output = MODEL(batch).squeeze(dim=1)
            labels = torch.where(output < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            # gold = F.one_hot(batch.y, num_classes=2).to(dtype=torch.float)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss
            num_batches += 1
            correct_output += torch.sum(batch.y == labels).item()

        if epoch % 1 == 0:
            output = MODEL.predict(val_data)
            labels = torch.where(output < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            correct_val = torch.sum(val_data.y == labels).item()
            print('-------------------------------------------')
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/num_batches:.4f}')
            print(f'Train Accuracy: {correct_output / num_graphs * 100 :.2f} %')
            print(f'Val Accuracy: {correct_val / val_data.num_graphs * 100 :.2f} %')

    torch.save(MODEL.state_dict(), model_path)


def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--num_epochs", required=False, default=200, type=int)
    parser.add_argument("--batch_size", required=False, default=32, type=int)
    parser.add_argument("--model", required=False, default='custom', type=str)
    args = parser.parse_args()
    print(
        f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")

    train(args.model_path, args.dataset_path, args.val_dataset_path,
          args.num_epochs, args.batch_size, args.model)


if __name__ == "__main__":
    main()
