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

DATA_DIR = "../dataset/dataset_2/train/"
BATCH_SIZE = 32
NUM_EPOCHS = 200

# Possible BaseLines: 'Random', 'Logistic', 'Custom'
MODEL = 'Custom'

NUM_EDIMS = 6

# Initialize the EdgeEncoder
edge_encoder = EdgeEncoder(NUM_EDIMS)

# Read the csv files into dataframes
edges_df = pd.read_csv(DATA_DIR + 'edges.csv.gz', compression='gzip', header=None)
num_nodes_df = pd.read_csv(DATA_DIR + 'num_nodes.csv.gz', compression='gzip', header=None)
num_edges_df = pd.read_csv(DATA_DIR + 'num_edges.csv.gz', compression='gzip', header=None)
graph_labels_df = pd.read_csv(DATA_DIR + 'graph_labels.csv.gz', compression='gzip', header=None)
node_features_df = pd.read_csv(DATA_DIR + 'node_features.csv.gz', compression='gzip', header=None)
edge_features_df = pd.read_csv(DATA_DIR + 'edge_features.csv.gz', compression='gzip', header=None)

NUM_GRAPHS = len(graph_labels_df)
data_list = []

nodes_done = 0
edges_done = 0
for i in range(NUM_GRAPHS):
    num_nodes = num_nodes_df.iloc[i, 0]
    num_edges = num_edges_df.iloc[i, 0]
    nodes_done += num_nodes
    edges_done += num_edges

    if math.isnan(graph_labels_df.iloc[i, 0]):
        continue

    node_features = torch.tensor(node_features_df.iloc[nodes_done-num_nodes:nodes_done, :].values, dtype=torch.float)
    edges = torch.tensor(edges_df.iloc[edges_done-num_edges:edges_done, :].values)
    edge_features = torch.tensor(edge_features_df.iloc[edges_done-num_edges:edges_done, :].values, dtype=torch.float)
    
    # # Initialize a list to hold the combined edge features for each node
    # combined_edge_features = [torch.empty((0, edge_features.shape[1])) for _ in range(num_nodes)]

    # # Combine edge features for each node
    # for idx, edge in enumerate(edges):
    #     combined_edge_features[edge[0]] = torch.cat((combined_edge_features[edge[0]], edge_features[idx].unsqueeze(0)))
    #     combined_edge_features[edge[1]] = torch.cat((combined_edge_features[edge[1]], edge_features[idx].unsqueeze(0)))

    # # Encode the combined edge features and append to node features
    # for node in range(num_nodes):
    #     edge_encoding = edge_encoder(combined_edge_features[node])
    #     node_features[node] = torch.cat((node_features[node], edge_encoding.mean(dim=0)))
        
    data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)
    data.y = torch.tensor([graph_labels_df.iloc[i, 0]], dtype=torch.float)  # Set the label
    data_list.append(data)


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

    def forward(self, x: torch.Tensor, length) -> torch.Tensor:
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
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


class Custom_Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Custom_Classifier, self).__init__()
        self.conv1 = GINEConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels)), edge_dim=3)
        self.conv2 = GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels)), edge_dim=3)
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


def custom_collate(batch):
    batch = Batch.from_data_list(batch)
    graph_num_nodes = [graph.num_nodes for graph in batch]
    batch.batch = torch.cat([torch.full((num_nodes,), graph_idx) for graph_idx, num_nodes in enumerate(graph_num_nodes)])
    return batch

dataset = CustomDataset(data_list)
dataloader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

'''
DataLoader is a collection of batches. Each batch is a Batch object, which is a collection of <BATCH_SIZE> graphs as follows:

for data in dataloader:
    graph_idx = 0  # The index of the graph you want to access: 0 to BATCH_SIZE-1
    subgraph = data[graph_idx]

'''

# model = Custom_Classifier(in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if MODEL == 'Custom':
    custom_model = Custom_Classifier(in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
    # optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=1e-3)
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.001)
else:
    logistic_model = Logistic_Regressor(in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
    # optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=1e-3)
    optimizer = torch.optim.Adam(logistic_model.parameters(), lr=0.001)

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()
# criterion = torch.nn.BCEWithLogitsLoss()


def train(num_epochs, model_type='Custom'):
    model = custom_model if model_type == 'Custom' else logistic_model
    for epoch in range(num_epochs):
        total_loss = 0
        correct_output = 0
        num_batches = 0
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch).squeeze(dim=1)
            labels = torch.where(output < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            # gold = F.one_hot(batch.y, num_classes=2).to(dtype=torch.float)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss
            num_batches += 1
            correct_output += torch.sum(batch.y == labels).item()

        # print(loss)

        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/num_batches:.4f}, Accuracy: {correct_output / NUM_GRAPHS * 100 :.2f}')


if MODEL == 'Random':
    random_model = Random_Classifier(num_classes=dataset.num_classes)
    correct_output = 0
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        predicted_output = random_model.forward(batch.x, length=len(batch.y))
        # print(predicted_output)
        # print(batch.y)
        loss = criterion(predicted_output, batch.y)
        total_loss += loss
        correct_output += torch.sum(batch.y == predicted_output).item()
        num_batches += 1
    
    print(f"Random Accuracy: {correct_output / NUM_GRAPHS * 100 :.2f}")
    print(f"Random BCE Loss {total_loss/num_batches}")

elif MODEL == 'Logistic':
    train(NUM_EPOCHS, 'logistic')

else:
    train(NUM_EPOCHS)

# torch.save(class_model.state_dict(), "q2_class_model.pt")